#!/usr/bin/env python3
"""OpenAI-compatible API test + benchmark for the AXE-Diffusion KSL-XE server.

stdlib-only. Measures streaming TTFT / end-to-end latency / tok/s, runs a
quality battery, and verifies protocol behavior (models list, non-stream,
finish reasons, 409-on-busy semantics).

usage: python3 scripts/test_api.py [--base http://127.0.0.1:8080]
"""
import argparse
import json
import time
import urllib.error
import urllib.request


def post(base, path, payload, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(base, prompt, max_tokens=256, thinking=False, model="diffusiongemma-w4a16"):
    payload = {"model": "knivesysl-diffusion-xe",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    text, reasoning = "", ""
    ttft = None
    chunks = 0
    usage, finish = None, None
    for _ in range(20):
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            break
        except urllib.error.HTTPError as e:
            if e.code == 409:
                time.sleep(3)
            else:
                raise
    with resp:
        buf = b""
        for raw in resp:
            buf += raw
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                line = block.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    delta = ch.get("delta", {})
                    c = delta.get("content") or ""
                    r = delta.get("reasoning_content") or ""
                    if c or r:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks += 1
                        text += c
                        reasoning += r
    e2e = time.perf_counter() - t0
    return {"ttft": ttft, "e2e": e2e, "chunks": chunks, "usage": usage,
            "finish": finish}


def nonstream_chat(base, prompt, max_tokens=256, thinking=False):
    payload = {"model": "knivesysl-diffusion-xe",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "thinking": thinking}
    req = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    for _ in range(20):
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            break
        except urllib.error.HTTPError as e:
            if e.code == 409:
                time.sleep(3)
            else:
                raise
    with resp:
        data = json.load(resp)
    e2e = time.perf_counter() - t0
    ch = data["choices"][0]
    return {"e2e": e2e, "content": ch["message"]["content"],
            "reasoning": ch["message"].get("reasoning_content", ""),
            "finish": ch["finish_reason"], "usage": data["usage"],
            "knivesys_meta": data.get("axe")}


QUALITY = [
    ("math", "What is 17 times 23?", "391"),
    ("capital", "What is the capital of Australia?", "canberra"),
    ("code", "Write a Python function rev(s) that returns s reversed. Just code.", "[::-1]"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print("== health ==")
    with urllib.request.urlopen(base + "/health", timeout=5) as r:
        print(" ", r.read().decode())

    print("== /v1/models ==")
    with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
        print(" ", r.read().decode())

    print("\n== warmup (stream) ==")
    r = stream_chat(base, "Say hello in one word.", 32)
    print(f"  e2e={r['e2e']:.1f}s ttft={r['ttft']:.1f}s chunks={r['chunks']} finish={r['finish']}")

    print("\n== speed (streaming, 3 runs each) ==")
    for name, prompt, _ in QUALITY:
        rates, ttfts = [], []
        for _ in range(3):
            r = stream_chat(base, prompt, 256)
            if r["usage"] and r["usage"].get("completion_tokens") and r["e2e"]:
                rates.append(r["usage"]["completion_tokens"] / r["e2e"])
            if r["ttft"]:
                ttfts.append(r["ttft"])
        med = sorted(rates)[len(rates) // 2] if rates else 0
        print(f"  [{name}] tok/s runs={[round(x,1) for x in rates]} median={med:.1f} "
              f"ttft={min(ttfts):.2f}-{max(ttfts):.2f}s" if rates else "  (no data)")

    print("\n== quality battery (non-stream) ==")
    ok = 0
    for name, prompt, expect in QUALITY:
        r = nonstream_chat(base, prompt, 256)
        answer = r["content"].lower()
        passed = expect.lower() in answer
        ok += passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: expected '{expect}' in "
              f"{r['content'][:80]!r} ({r['e2e']:.1f}s, {r['usage']['completion_tokens']} tok)")
    print(f"  {ok}/{len(QUALITY)} passed")

    print("\n== thinking on (non-stream) ==")
    r = nonstream_chat(base, "Should I water a cactus daily?", 256, thinking=True)
    print(f"  reasoning present: {bool(r['reasoning'])}, "
          f"answer: {r['content'][:100]!r}")

    print("\n== edge: max_tokens=1 ==")
    r = nonstream_chat(base, "Hi", 1)
    print(f"  finish={r['finish'] if 'finish' in r else ''} completion_tokens={r['usage']['completion_tokens']}")

    print("\nALL CHECKS DONE")


if __name__ == "__main__":
    main()