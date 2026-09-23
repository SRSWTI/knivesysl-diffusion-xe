# knivesysl-diffusion-xe

a 26b-a4b block-diffusion language model served on a single intel arc pro b70 —
one process, one model, two apis: a browser studio with live revisable drafts
and an openai-compatible endpoint for routers and cockpits.

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

one uvicorn process loads the model once (`load_fast`, turbo tier) onto
`xpu:0` and serves both apis from that single instance. generation is
identical for both: prompt → block-diffusion denoising loop → committed
tokens, with a `draftstreamer` that emits revisable drafts as they happen and
locks the gpu to one generation at a time.

```
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
./scripts/xe.sh up        # start + wait for health (model load ~18 s)
./scripts/xe.sh status    # container + gpu + health + /v1/models
./scripts/xe.sh test      # api benchmark + quality battery
./scripts/xe.sh logs      # follow logs
./scripts/xe.sh gpu       # xpu-smi snapshot
./scripts/xe.sh down      # stop
```

env: `AXE_PORT` (8080), `AXE_IMAGE` (serving image), `SERVED_MODEL_NAME`
(`knivesysl-diffusion-xe`), `NVIDIA_BASE_URL` (optional nvidia backend for the
studio).

openai example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -h 'content-type: application/json' -d '{
  "model": "knivesysl-diffusion-xe",
  "messages": [{"role":"user","content":"what is 17 times 23?"}],
  "max_tokens": 256
}'
```

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

~18 gib vram for weights, ~20 gib peak (the fp32 logits tensor alone is
268 mib per denoising step). checkpoint format `dg-w4a16-v1`: asymmetric
uint4, fp16 scale and zero-point per group, two nibbles per byte. gate_up_proj
groups at 128 along the contraction dim, down_proj at 64 (its contraction dim
is 704 and 128 does not divide it). the vision tower rides at bf16 and
multimodal paths are untested.

the first forward jits the triton kernels — expect a one-time warmup on a
freshly started server.