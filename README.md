# axe-diffusion ksl-xe

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
steps, refining token positions before committing each completed canvas. 26b total
parameters with 4b active: a 128-expert moe routing top-8 per token, across 30
encoder and 30 decoder layers.

because generation is denoising rather than decoding, quantization and
throughput are coupled: quantization noise affects the entropy-based acceptance
and stopping decisions. the kernels we wrote target exactly that loop.

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
revisable drafts as they happen. a shared generation lock limits the gpu to
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

## diffusion settings and tuning

the diffusion controls can be tuned to compare output quality against latency,
but a larger value is not automatically better. distinguish ordinary generation
controls from changes to model dimensions, routing, or supported context.

### current defaults

these are the checkpoint defaults checked on 2026-09-23 for snapshot
`ac52f59d384f91708e3732337c54bc825d1fd748` of
`srswti/axe-diffusion-ksl-xe`. generation controls come from
`generation_config.json`; canvas length, context, and expert counts come from
`config.json`. the server loads this configuration with `load_fast`, using
`device="xpu:0"` and `tier="turbo"`.

| setting | configuration field | current default |
|---|---|---|
| canvas length | `canvas_length` | 256 tokens |
| maximum denoising steps | `max_denoising_steps` | 48 per canvas |
| entropy bound | `sampler_config.entropy_bound` | 0.1 |
| confidence threshold | `confidence_threshold` | 0.005 |
| stability threshold | `stability_threshold` | 1 |
| final / initial temperature | `t_min` / `t_max` | 0.4 / 0.8 |
| text context limit | `text_config.max_position_embeddings` | 262,144 tokens |
| total experts | `text_config.num_experts` | 128 |
| selected experts per token | `text_config.top_k_experts` | 8 |

the checkpoint also sets `max_new_tokens=256`, but the server overrides that
with each request's `max_tokens`. output length and canvas length are different
controls: a longer response can require multiple canvases.

### the five inference controls

the turbo kernels still use these controls; fusion does not bypass them.

| control | what it changes |
|---|---|
| `max_denoising_steps` | the maximum number of decoder refinement passes per canvas, not per response. early stopping can finish before this ceiling. |
| `entropy_bound` | how many proposed positions are accepted together, starting with the lowest-entropy positions. for the same logits, a larger bound accepts at least as many positions; a smaller bound is more restrictive. |
| `confidence_threshold` | early stopping requires mean token entropy below this value. lower is stricter; higher makes the confidence condition easier to satisfy. model confidence is not a correctness score. |
| `stability_threshold` | early stopping also requires the entire argmax canvas to match this many previous canvases. `2` requires two previous matches; `0` removes the stability requirement but retains the confidence check. |
| `t_min` / `t_max` | sampling temperature starts at `t_max` and moves toward `t_min`. lower temperatures sharpen the distribution; higher temperatures flatten it. neither direction guarantees better answers. |

accepting a token position during refinement is not the same as committing it
to the final response. the canvas continues through the denoising loop.
acceptance sorts token entropies and selects positions whose cumulative entropy,
excluding the current position, is within the bound. this is not an independent
per-token confidence cutoff.

### does increasing denoising steps improve accuracy?

not necessarily. `48 → 96` increases the available refinement budget, not a
guaranteed amount of useful reasoning or a guaranteed accuracy score.

- **it is a ceiling, not a mandatory count.** a canvas can stop early once both
  stability and confidence conditions are satisfied.
- **the maximum refinement work doubles.** actual request latency need not
  double: prompt processing, early stopping, and output length also contribute.
- **the temperature trajectory changes too.** the installed processor uses:

  ```text
  temperature = t_min + (t_max - t_min) * remaining_steps / max_denoising_steps
  ```

  increasing the maximum stretches this schedule across more passes. a 96-step
  run is not simply a 48-step run followed by 48 extra corrections. the loop
  counts down to 1, so the final scheduled value approaches `t_min` rather than
  evaluating the formula at zero remaining steps.

the experiment is whether additional refinement fixes errors often enough to
justify its latency. do not assume a confidently wrong answer becomes correct
just because it received more passes.

temperature and stopping also interact: entropy is measured after temperature
scaling. lowering temperature can satisfy the confidence condition more easily
without establishing that the answer is more accurate. change one control at a
time initially. use positive temperatures with `t_max > t_min` for experiments;
the installed validator rejects equal endpoints, and zero temperature is not a
greedy-mode switch in the division above.

### keep architecture experiments separate

| setting | what to know before changing it |
|---|---|
| canvas length | the generator reads it from model configuration rather than hardcoding 256 throughout the loop. changing it changes tensor shapes, per-step work, and block boundaries. treat other sizes as advanced experiments requiring kernel, memory, and quality validation; neither larger nor smaller is automatically faster or more accurate. |
| context limit | a smaller application limit is possible. raising the configuration number alone does not establish reliable longer-context support or prove that the context fits available vram. a larger configured limit does not improve a short prompt by itself. |
| total experts | 128 is part of the trained router and expert weight shapes, not an ordinary inference control. editing the number does not create additional trained experts. |
| selected experts | the router reads `top_k_experts` at forward time, so changing it is technically possible with coordinated configuration changes. it changes which expert contributions are used. fewer experts versus speed and more experts versus quality are hypotheses to measure, not guaranteed improvements. |

leave canvas length, context, and expert routing unchanged during the first
sampler experiments. changing selected experts is a model-altering experiment,
not equivalent to adjusting sampling temperature.

### what the server exposes today

the server forwards `input_ids`, `max_new_tokens`, and `streamer` to
`model.generate`. requests expose an output-token budget and thinking mode,
but not the five diffusion controls above. the openai request's `temperature`
and `top_p` fields are accepted for compatibility but are not forwarded to
local generation. adding these diffusion fields to an http request alone does
not enable tuning.

the installed generator supports per-call overrides and deep-copies generation
configuration before applying them. prefer per-request configuration over
mutating shared model defaults or editing the cached checkpoint. for example,
inside the python generation path with the existing model, inputs, budget, and
streamer available:

```python
from copy import deepcopy

config = deepcopy(model.generation_config)
config.max_denoising_steps = 64
config.confidence_threshold = 0.005
config.stability_threshold = 1
config.t_min = 0.4
config.t_max = 0.8
config.sampler_config.entropy_bound = 0.1

model.generate(
    input_ids=inputs,
    max_new_tokens=max_tokens,
    streamer=streamer,
    generation_config=config,
)
```

this is an integration example, not an implemented api feature or a tested
better preset. `sampler_config` is the typed sampler configuration loaded from
the checkpoint; the example changes its copied entropy bound.

### a useful first experiment

keep the current defaults as the baseline, then vary one control at a time:

1. compare step ceilings of `24`, `48`, `64`, and `96`.
2. at the chosen ceiling, compare confidence thresholds of `0.01`, `0.005`, and
   `0.0025`.
3. compare stability thresholds of `1` and `2`.
4. explore entropy bound and temperature afterward.

these are suggested experiment points, not validated presets. use the same
prompts and output budgets, with repeated seeded runs. turbo uses fused
gumbel-max sampling with a different random stream from native multinomial
sampling, so matching a seed does not make those two paths token-identical.

record correctness, tests passed, instruction compliance, end-to-end latency,
generated token count, and truncation or incomplete-edit failures. also measure
actual denoising passes, ideally per canvas, and how often the ceiling is hit.
the server's existing step counter totals draft callbacks across the response;
per-canvas ceiling-hit reporting would require additional instrumentation.

the goal is a measured correctness-versus-latency tradeoff. do not label a
higher-step preset “more accurate” until results support it.

## run

```bash
./scripts/xe.sh up        # build axe image if absent, fetch weights, start + wait
./scripts/xe.sh fetch     # fetch checkpoint assets only; never overwrite kernels/
./scripts/xe.sh restart   # reload local python/kernel edits
./scripts/xe.sh status    # container + gpu + health + /v1/models
./scripts/xe.sh test      # api benchmark + quality battery
./scripts/xe.sh logs      # follow logs
./scripts/xe.sh gpu       # xpu-smi snapshot
./scripts/xe.sh down      # stop
```

open **http://localhost:8080/edit** for the code demo, or `/` for studio.

the image is `local/axe-diffusion-ksl-xe:triton38`. its `Dockerfile` uses the
existing `local/gemma-w4-intel:tested` runtime base and installs
`triton-xpu==3.8.0`. it embeds the server, both uis, and kernels. compose mounts
the workspace versions read-only, so local optimizations take effect after
`./scripts/xe.sh restart`; no kernel re-download or image rebuild is needed.
to refresh the embedded standalone image too, run
`docker build -t local/axe-diffusion-ksl-xe:triton38 .`.

`hf download` fetches weights, config, and tokenizer assets from
`srswti/axe-diffusion-ksl-xe` into the hf cache. the server resolves those same
assets offline. `AXE_REVISION` defaults to `main`; set a commit hash to pin a
release. local `kernels/` remains authoritative in either case. to update kernel
code from the hub, explicitly review and download it; startup never overwrites
your local changes.

env: `AXE_PORT` (8080), `AXE_IMAGE` (serving image), `AXE_BASE_IMAGE` (build base),
`AXE_REVISION` (main), `HF_HOME` (host hf cache root), `SERVED_MODEL_NAME`
(`axe-diffusion-ksl-xe`), `NVIDIA_BASE_URL` (optional existing studio proxy).
this lifecycle is intel xpu only; a cuda lifecycle is not implemented yet.

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
  line, or function, or place the cursor where new text belongs. axe generates
  only a replacement. the server constructs `prefix + replacement + suffix`,
  preserving all source outside the range exactly. explicit response delimiters
  preserve indentation and trailing newlines without json-escaping generated code.
- **revise whole file** is an explicit opt-in for broader changes such as adding
  documentation throughout a file. unrelated-code preservation is prompted,
  not guaranteed in this mode.
- both modes show real diffusion draft snapshots in the same editor. changed
  lines are highlighted; **review changed lines** shows removed and added text.
  **undo**, **redo**, **copy**, prompt chips, and an output budget are available.
- failed, cancelled, malformed, or token-exhausted responses restore the original
  editor contents. python final documents must parse before being committed.
  parsing does not prove semantic correctness; other languages are not compiled
  or syntax-validated. submitted/generated code is never executed by the server.

this is instruction-conditioned generation with application-enforced edit
boundaries, **not native masked-canvas diffusion infilling**. there is one model
and one shared generation lock; concurrent generation receives http 409.

`POST /edit` accepts `code`, `instruction`, `language`, `max_tokens` (1–8192),
and `mode` (`selection` or `whole`). selection mode also requires
`selection_start` and `selection_end`: zero-based **unicode code-point** offsets,
end-exclusive. equal offsets mean insertion. the ui converts javascript utf-16
selection offsets before sending. source is limited to 100,000 characters,
instructions to 8,000, and the tokenized edit prompt to 32,768 tokens.

sse `draft` and `commit` events carry complete candidate documents in `code`.
only a terminal `done` event means a validated revision; `error` means retain the
original. timings and denoising steps come from the actual generation.

## editing verification

```bash
# real-tokenizer boundary, framing, cancellation, and completion regressions
docker compose run --rm --no-deps -v "$PWD/scripts:/app/scripts:ro" xe-diffusion -m unittest scripts.test_editing -v
# real-model selection, insertion, whole-file, busy/cancel, and error scenarios
python3 scripts/test_edit_api.py --base http://127.0.0.1:8080
```

the live tests evaluate only restricted python fixtures in a separate,
resource-limited test process to check behavior. there are no mocked model
responses. the ordinary `./scripts/xe.sh test` remains the openai api benchmark
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

docker with intel `/dev/dri` access, the existing runtime base image, the host
`hf` cli, and enough hf-cache disk space for the checkpoint are required.

~18 gib vram for weights, ~20 gib peak (the fp32 logits tensor alone is
268 mib per denoising step). checkpoint format `dg-w4a16-v1`: asymmetric
uint4, fp16 scale and zero-point per group, two nibbles per byte. gate_up_proj
groups at 128 along the contraction dim, down_proj at 64 (its contraction dim
is 704 and 128 does not divide it). the vision tower rides at bf16 and
multimodal paths are untested.

the first forward jits the triton kernels — expect a one-time warmup on a
freshly started server.