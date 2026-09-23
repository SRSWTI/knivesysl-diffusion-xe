#!/usr/bin/env python3
"""AXE-Diffusion KSL-XE server (GoedelMachines W4A16 only).

One uvicorn process, one model instance, two APIs:

  * Diffusion Studio UI:                 GET /        POST /chat     GET /health
  * OpenAI-compatible (router/cockpit):  GET /v1/models
                                         POST /v1/chat/completions
                                         (OpenAI SSE stream + non-stream, usage)

Serves GoedelMachines/diffusiongemma-26B-A4B-w4a16 via the checkpoint's fused
W4 loader (`load_fast`, tier=turbo) on a single Intel XPU GPU. Generation runs
block diffusion with live revisable drafts; the OpenAI endpoint streams
committed deltas and reports denoising stats under the "axe" response key.
"""
import asyncio
import json
import os
import queue
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

REPO = "GoedelMachines/diffusiongemma-26B-A4B-w4a16"
REVISION = "0fede6e3e65d258d444822cfb6a79092cdc5c0af"
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "diffusiongemma-w4a16")
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL",
    "http://172.17.0.1:8000" if Path("/.dockerenv").exists() else "http://127.0.0.1:8000",
).rstrip("/")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "RedHatAI/diffusiongemma-26B-A4B-it-NVFP4")

state = {}
generation_lock = threading.Lock()


class CancelledGeneration(Exception):
    pass


def split_channels(raw):
    if "<|channel>thought\n" in raw:
        before, body = raw.split("<|channel>thought\n", 1)
        if "<channel|>" in body:
            reasoning, answer = body.split("<channel|>", 1)
            return reasoning.strip(), (before + answer).strip()
        return body.strip(), before.strip()
    return "", raw.strip()


class DraftStreamer:
    def __init__(self, tokenizer, eos_ids, events, cancelled):
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.events = events
        self.cancelled = cancelled
        self.prompt = True
        self.committed = []
        self.steps = 0
        self.started = time.perf_counter()
        self.first_draft = None

    def check_cancel(self):
        if self.cancelled.is_set():
            raise CancelledGeneration

    def trim(self, value):
        tokens = value.reshape(-1).tolist()
        end = next((i for i, token in enumerate(tokens) if token in self.eos_ids), len(tokens))
        return tokens[:end]

    def emit(self, kind, tokens):
        raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
        for marker in ("<|turn>", "<turn|>", "", "<pad>"):
            raw = raw.replace(marker, "")
        reasoning, answer = split_channels(raw)
        self.events.put({"type": kind, "reasoning": reasoning, "text": answer,
                         "steps": self.steps, "elapsed": time.perf_counter() - self.started})

    def put(self, value):
        self.check_cancel()
        if self.prompt:
            self.prompt = False
            return
        self.committed.extend(self.trim(value))
        self.emit("commit", self.committed)

    def put_draft(self, value, **kwargs):
        self.check_cancel()
        self.steps += 1
        if self.first_draft is None:
            self.first_draft = time.perf_counter() - self.started
        self.emit("draft", self.committed + self.trim(value))

    def end(self):
        pass


@asynccontextmanager
async def lifespan(app):
    os.environ["MOE_CUDA_ALIGN"] = "0"
    if os.environ.get("LOAD_LOCAL", "1") == "1":
        root = snapshot_download(REPO, revision=REVISION, local_files_only=True)
        sys.path.insert(0, root + "/kernels")
        from load_w4_checkpoint import load_fast
        state["model"], state["tokenizer"] = load_fast(root, device="xpu:0", tier="turbo")
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5)) as client:
        state["nvidia_client"] = client
        yield


app = FastAPI(lifespan=lifespan)


def run_generation(messages, thinking, max_tokens, events, cancelled):
    """Shared generation core: studio /chat and /v1/chat/completions both use it."""
    try:
        model, tokenizer = state["model"], state["tokenizer"]
        inputs = tokenizer.apply_chat_template(
            [m if isinstance(m, dict) else m.model_dump() for m in messages],
            enable_thinking=thinking, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )["input_ids"].to("xpu:0")
        context_limit = model.config.text_config.max_position_embeddings
        if inputs.shape[1] + max_tokens > context_limit:
            raise ValueError(f"Prompt plus output budget exceeds {context_limit} tokens.")
        eos = model.generation_config.eos_token_id
        streamer = DraftStreamer(tokenizer, set(eos if isinstance(eos, list) else [eos]),
                                 events, cancelled)
        with torch.no_grad():
            model.generate(input_ids=inputs, max_new_tokens=max_tokens, streamer=streamer)
        torch.xpu.synchronize()
        events.put({"type": "done", "tokens": len(streamer.committed),
                    "prompt_tokens": int(inputs.shape[1]),
                    "elapsed": time.perf_counter() - streamer.started,
                    "first_draft": streamer.first_draft, "steps": streamer.steps,
                    "memory_gib": torch.xpu.memory_allocated() / 2**30})
    except CancelledGeneration:
        pass
    except Exception as error:
        events.put({"type": "error", "message": str(error)})
    finally:
        generation_lock.release()
        events.put(None)


def drain_sync(events):
    """Collect the final text/reasoning/done event from an event queue."""
    result = {"text": "", "reasoning": "", "done": None}
    while True:
        event = events.get()
        if event is None:
            break
        if event.get("type") == "error":
            raise RuntimeError(event["message"])
        if event.get("type") in ("draft", "commit"):
            result["text"], result["reasoning"] = event.get("text", ""), event.get("reasoning", "")
        elif event.get("type") == "done":
            result["done"] = event
    return result


# ───────────────────────────── Studio API ─────────────────────────────


class StudioMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100000)


class StudioChatRequest(BaseModel):
    messages: list[StudioMessage] = Field(min_length=1, max_length=100)
    max_tokens: Literal[256, 512, 1024, 2048, 4096, 8192] = 512
    thinking: bool = False
    backend: Literal["intel", "nvidia"] = "intel"


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("web.html"))


@app.get("/health")
async def health(backend: Literal["intel", "nvidia"] = "intel"):
    if backend == "intel":
        return {"ready": "model" in state, "busy": generation_lock.locked(),
                "device": "Intel XPU 0", "model": SERVED_MODEL_NAME}
    try:
        response = await state["nvidia_client"].get(NVIDIA_BASE_URL + "/health", timeout=3)
        ready = response.is_success
    except httpx.HTTPError:
        ready = False
    return {"ready": ready, "busy": False, "device": "NVIDIA RTX 5090"}


async def nvidia_stream(body):
    started = time.perf_counter()
    text, reasoning = "", ""
    first_output = None
    chunks = 0
    usage = None
    finish_reason = None
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [message.model_dump() for message in body.messages],
        "max_completion_tokens": body.max_tokens,
        "chat_template_kwargs": {"enable_thinking": body.thinking},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with state["nvidia_client"].stream(
            "POST", NVIDIA_BASE_URL + "/v1/chat/completions", json=payload
        ) as response:
            if not response.is_success:
                detail = (await response.aread()).decode(errors="replace")
                raise RuntimeError(f"NVIDIA server HTTP {response.status_code}: {detail[:1000]}")
            data = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                    continue
                if line or not data:
                    continue
                raw = "\n".join(data)
                data.clear()
                if raw == "[DONE]":
                    if finish_reason is None:
                        raise RuntimeError("NVIDIA stream ended without a finish reason.")
                    yield "data: " + json.dumps({
                        "type": "done", "backend": "nvidia",
                        "elapsed": time.perf_counter() - started,
                        "first_output": first_output, "chunks": chunks,
                        "tokens": usage.get("completion_tokens") if usage else None,
                        "finish_reason": finish_reason,
                    }) + "\n\n"
                    return
                event = json.loads(raw)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    content = delta.get("content") or ""
                    thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if not content and not thought:
                        continue
                    text += content
                    reasoning += thought
                    chunks += 1
                    elapsed = time.perf_counter() - started
                    if first_output is None:
                        first_output = elapsed
                    yield "data: " + json.dumps({
                        "type": "commit", "backend": "nvidia", "text": text,
                        "reasoning": reasoning, "chunks": chunks, "elapsed": elapsed,
                    }) + "\n\n"
            raise RuntimeError("NVIDIA stream disconnected before completion.")
    except Exception as error:
        yield "data: " + json.dumps({"type": "error", "message": str(error)}) + "\n\n"


@app.post("/chat")
async def chat(body: StudioChatRequest, request: Request):
    if body.messages[-1].role != "user":
        raise HTTPException(400, "The last message must be from the user.")
    if body.backend == "nvidia":
        return StreamingResponse(nvidia_stream(body), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(409, "The Intel GPU is busy. Wait for the current generation to finish.")
    events = queue.Queue()
    cancelled = threading.Event()
    threading.Thread(target=run_generation,
                     args=(body.messages, body.thinking, body.max_tokens, events, cancelled),
                     daemon=True).start()

    async def stream():
        try:
            while True:
                if await request.is_disconnected():
                    cancelled.set()
                    break
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.015)
                    continue
                if event is None:
                    break
                yield "data: " + json.dumps(event) + "\n\n"
        finally:
            cancelled.set()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ──────────────────────── OpenAI-compatible API ────────────────────────


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100000)


class OpenAIChatRequest(BaseModel):
    model: str = SERVED_MODEL_NAME
    messages: list[OpenAIMessage] = Field(min_length=1, max_length=100)
    max_tokens: int = Field(default=512, ge=1, le=131072)
    temperature: float | None = None  # accepted for OpenAI compatibility
    top_p: float | None = None
    stream: bool = False
    stream_options: dict | None = None
    thinking: bool = False  # non-standard; also honored via chat_template_kwargs
    chat_template_kwargs: dict = Field(default_factory=dict)


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{
        "id": SERVED_MODEL_NAME, "object": "model", "created": 0,
        "owned_by": "axe", "max_model_len": 262144,
        "architecture": "DiffusionGemma 26B-A4B W4A16 (Intel XPU)",
    }]}


@app.post("/v1/chat/completions")
async def openai_chat(body: OpenAIChatRequest, request: Request):
    if body.messages[-1].role != "user":
        raise HTTPException(400, "The last message must be from the user.")
    thinking = body.thinking
    if body.chat_template_kwargs.get("enable_thinking") is not None:
        thinking = bool(body.chat_template_kwargs["enable_thinking"])

    if not generation_lock.acquire(blocking=False):
        raise HTTPException(409, "The Intel GPU is busy. Wait for the current generation to finish.")
    events = queue.Queue()
    cancelled = threading.Event()
    threading.Thread(
        target=run_generation,
        args=(body.messages, thinking, body.max_tokens, events, cancelled),
        daemon=True,
    ).start()

    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    started_at = time.perf_counter()

    async def stream():
        sent_text, sent_reasoning, first_output, chunks, final = "", "", None, 0, None

        def chunk(delta, finish=None, usage=None):
            payload = {"id": completion_id, "object": "chat.completion.chunk",
                       "created": created, "model": body.model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                payload["usage"] = usage
            return payload

        try:
            yield "data: " + json.dumps(chunk({"role": "assistant"})) + "\n\n"
            while True:
                if await request.is_disconnected():
                    cancelled.set()
                    break
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.015)
                    continue
                if event is None:
                    break
                if event.get("type") == "error":
                    raise RuntimeError(event["message"])
                if event.get("type") in ("draft", "commit"):
                    text, reasoning = event.get("text", ""), event.get("reasoning", "")
                    if first_output is None and (text or reasoning):
                        first_output = time.perf_counter() - started_at
                    text_delta = text[len(sent_text):]
                    reasoning_delta = reasoning[len(sent_reasoning):]
                    if text_delta or reasoning_delta:
                        chunks += 1
                        delta = {}
                        if reasoning_delta:
                            delta["reasoning_content"] = reasoning_delta
                        if text_delta:
                            delta["content"] = text_delta
                        sent_text, sent_reasoning = text, reasoning
                        yield "data: " + json.dumps(chunk(delta)) + "\n\n"
                elif event.get("type") == "done":
                    final = event
                    break
            usage = {
                "prompt_tokens": final.get("prompt_tokens", 0) if final else 0,
                "completion_tokens": final.get("tokens", 0) if final else 0,
                "total_tokens": (final.get("prompt_tokens", 0) + final.get("tokens", 0)) if final else 0,
            }
            yield "data: " + json.dumps(chunk({}, finish="stop", usage=usage)) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            cancelled.set()

    if body.stream:
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    result = drain_sync(events)
    done = result["done"] or {}
    finish = "length" if (done.get("tokens") or 0) >= body.max_tokens else "stop"
    message = {"role": "assistant", "content": result["text"]}
    if result["reasoning"]:
        message["reasoning_content"] = result["reasoning"]
    return {"id": completion_id, "object": "chat.completion", "created": created,
            "model": body.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": done.get("prompt_tokens", 0),
                      "completion_tokens": done.get("tokens", 0),
                      "total_tokens": done.get("prompt_tokens", 0) + done.get("tokens", 0)},
            "axe": {"elapsed": done.get("elapsed"), "first_draft": done.get("first_draft"),
                     "denoising_steps": done.get("steps")}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
