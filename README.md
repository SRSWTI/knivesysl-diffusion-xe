# AXE-Diffusion KSL-XE

a 26b-a4b block-diffusion language model served on a single intel arc pro b70 —
one process, one model: an in-place code-editing demo, a browser studio with
live revisable drafts, and an openai-compatible endpoint.

the weights are 4-bit quantized, and the fast kernels that run them are ours:
we wrote and fused the kernels ourselves instead of relying on generic
inference-stack implementations. that choice is the whole speed story — the
same gpu serving the same model class through a generic engine runs 20–95x
slower.

## what the model is

this is a block-diffusion language model, not autoregressive. instead of
emitting one token at a time, it denoises a 256-token canvas over up to 48
steps, committing tokens once their entropy drops below a bound. 26b total
parameters with 4b active: a 128-expert moe routing top-8 per token, across 30
encoder and 30 decoder layers.

because generation is denoising rather than decoding, quantization and
throughput are coupled: quantization noise feeds the entropy bound that decides
how many tokens commit per step. the kernels we wrote target exactly that loop.

## the kernels we built

the expert weights are asymmetric uint4 with an fp16 scale and zero-point per
group. the gemm reads packed nibbles and dequantizes inline, streaming real
4-bit traffic — experts drop from 85 gib to ~11.5 gib, and since the expert
path is bandwidth-bound that accounts for most of the win.

on top of that:

- one fused triton launch handles all 128 experts instead of a 128-launch
  python loop, with the activation folded into the first gemm's epilogue
- rmsnorm collapses from roughly six elementwise launches into one, 180 times
  per forward
- the weighted combine uses a fixed-order reduction instead of index_add_,
  because in block diffusion a 1-ulp difference from atomics flips an
  acceptance threshold and forks the entire trajectory
- the sampler computes entropy once per denoising step in a single streaming
  pass and replaces softmax + multinomial with one gumbel-max kernel — on the
  stock loop the same 268 mib logits tensor was processed twice per step, and
  fixing that is worth ~1.4x by itself

## how the server works

one uvicorn process loads `srswti/axe-diffusion-ksl-xe` once (`load_fast`, turbo
tier) onto `xpu:0`. the loader comes from this repository's `kernels/`, mounted
at `/app/kernels`, not from a cached upstream checkpoint. generation follows
prompt → block-diffusion denoising → committed tokens. a `DraftStreamer` emits
revisable drafts as they happen. a shared generation lock limits the GPU to
one generation at a time.

```
code editor      →  GET /edit        (selection, cursor insertion, whole file)
editing sse      →  POST /edit       (full draft/commit snapshots, final source)
studio ui        →  GET /            (browser: live drafts, thinking toggle)
studio sse       →  POST /chat       (draft/commit events, per-backend history)
openai api       →  POST /v1/chat/completions   (openai sse or json)
                 →  GET  /v1/models
```

the openai endpoint speaks standard streaming (`chat.completion.chunk` +
`data: [DONE]`), carries thinking-channel text in `reasoning_content`, and
returns `usage` in the final chunk. non-stream responses add an `"axe"` stats
block: elapsed, first draft time, denoising steps.

## run

```bash
./scripts/xe.sh up        # build AXE image if absent, fetch weights, start + wait
./scripts/xe.sh fetch     # fetch checkpoint assets only; never overwrite kernels/
./scripts/xe.sh restart   # reload local Python/kernel edits
./scripts/xe.sh status    # container + gpu + health + /v1/models
./scripts/xe.sh test      # api benchmark + quality battery
./scripts/xe.sh logs      # follow logs
./scripts/xe.sh gpu       # xpu-smi snapshot
./scripts/xe.sh down      # stop
```

open **http://localhost:8080/edit** for the code demo, or `/` for Studio.

the image is `local/axe-diffusion-ksl-xe:triton38`. its Dockerfile uses the
existing `local/gemma-w4-intel:tested` runtime base and installs
`triton-xpu==3.8.0`. it embeds the server, both UIs, and kernels. compose mounts
the workspace versions read-only, so local optimizations take effect after
`./scripts/xe.sh restart`; no kernel re-download or image rebuild is needed.
to refresh the embedded standalone image too, run
`docker build -t local/axe-diffusion-ksl-xe:triton38 .`.

`hf download` fetches weights, config, and tokenizer assets from
`srswti/axe-diffusion-ksl-xe` into the HF cache. the server resolves those same
assets offline. `AXE_REVISION` defaults to `main`; set a commit hash to pin a
release. local `kernels/` remains authoritative in either case. to update kernel
code from the Hub, explicitly review and download it; startup never overwrites
your local changes.

env: `AXE_PORT` (8080), `AXE_IMAGE` (serving image), `AXE_BASE_IMAGE` (build base),
`AXE_REVISION` (main), `HF_HOME` (host HF cache root), `SERVED_MODEL_NAME`
(`axe-diffusion-ksl-xe`), `NVIDIA_BASE_URL` (optional existing Studio proxy).
this lifecycle is Intel XPU only; a CUDA lifecycle is not implemented yet.

openai example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "axe-diffusion-ksl-xe",
  "messages": [{"role":"user","content":"what is 17 times 23?"}],
  "max_tokens": 256
}'
```

## code-editing demo

- **edit selection / insert at cursor** is the default. select an expression,
  line, or function, or place the cursor where new text belongs. AXE generates
  only a replacement. the server constructs `prefix + replacement + suffix`,
  preserving all source outside the range exactly. explicit response delimiters
  preserve indentation and trailing newlines without JSON-escaping generated code.
- **revise whole file** is an explicit opt-in for broader changes such as adding
  documentation throughout a file. unrelated-code preservation is prompted,
  not guaranteed in this mode.
- both modes show real diffusion draft snapshots in the same editor. changed
  lines are highlighted; **review changed lines** shows removed and added text.
  **undo**, **redo**, **copy**, prompt chips, and an output budget are available.
- failed, cancelled, malformed, or token-exhausted responses restore the original
  editor contents. Python final documents must parse before being committed.
  parsing does not prove semantic correctness; other languages are not compiled
  or syntax-validated. submitted/generated code is never executed by the server.

this is instruction-conditioned generation with application-enforced edit
boundaries, **not native masked-canvas diffusion infilling**. there is one model
and one shared generation lock; concurrent generation receives HTTP 409.

`POST /edit` accepts `code`, `instruction`, `language`, `max_tokens` (1–8192),
and `mode` (`selection` or `whole`). selection mode also requires
`selection_start` and `selection_end`: zero-based **Unicode code-point** offsets,
end-exclusive. equal offsets mean insertion. the UI converts JavaScript UTF-16
selection offsets before sending. source is limited to 100,000 characters,
instructions to 8,000, and the tokenized edit prompt to 32,768 tokens.

SSE `draft` and `commit` events carry complete candidate documents in `code`.
only a terminal `done` event means a validated revision; `error` means retain the
original. timings and denoising steps come from the actual generation.

## editing verification

```bash
# real-tokenizer boundary, framing, cancellation, and completion regressions
docker compose run --rm --no-deps -v "$PWD/scripts:/app/scripts:ro" xe-diffusion -m unittest scripts.test_editing -v
# real-model selection, insertion, whole-file, busy/cancel, and error scenarios
python3 scripts/test_edit_api.py --base http://127.0.0.1:8080
```

the live tests evaluate only restricted Python fixtures in a separate,
resource-limited test process to check behavior. there are no mocked model
responses. the ordinary `./scripts/xe.sh test` remains the OpenAI API benchmark
and quality check; its thinking/token-limit lines are diagnostics, not assertions.

## measured performance (arc pro b70, triton 3.8.0 xpu)

| prompt | tok/s (median of 3) |
|---|---|
| math (391 correct) | ~44 |
| capital (canberra correct) | ~24 |
| explain diffusion lms | ~35 |
| cold warmup | 0.5 s |

run-to-run variance is inherent: block diffusion commits a different token
count per run depending on the entropy bound. short answers are latency-bound
not throughput-bound — the canvas is denoised whether the reply is 12 tokens
or 250, so time-to-answer is the metric that matters conversationally.

## requirements

Docker with Intel `/dev/dri` access, the existing runtime base image, the host
`hf` CLI, and enough HF-cache disk space for the checkpoint are required.

~18 gib vram for weights, ~20 gib peak (the fp32 logits tensor alone is
268 mib per denoising step). checkpoint format `dg-w4a16-v1`: asymmetric
uint4, fp16 scale and zero-point per group, two nibbles per byte. gate_up_proj
groups at 128 along the contraction dim, down_proj at 64 (its contraction dim
is 704 and 128 does not divide it). the vision tower rides at bf16 and
multimodal paths are untested.

the first forward jits the triton kernels — expect a one-time warmup on a
freshly started server.