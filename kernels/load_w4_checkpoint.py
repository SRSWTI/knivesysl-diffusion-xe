"""load_w4_checkpoint.py — load the W4A16 checkpoint WITHOUT the 52GB BF16 base.

Meta-init (params on meta, buffers REAL so rotary inv_freq etc. survive) -> assign non-expert
weights from the shards -> re-tie lm_head -> attach packed expert buffers + bind the fused
Triton forward. Peak host memory ~= checkpoint size (~31GB), not 52+31.

Usage — `load_fast` is the entry point most callers want; it adds the fused sampler and RMSNorm
kernels on top of `load_w4`:

    from load_w4_checkpoint import load_fast
    model, tok = load_fast("..")               # repo root; cuda, eval, ready for model.generate

`load_w4` alone gives plain W4 with the fused MoE GEMM and no sampler/norm kernels.
"""
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ["MOE_V2"] = "1"                     # the deterministic v2 kernel (v1 = fallback)
import torch
from safetensors import safe_open
from safetensors.torch import load_file

EXPERT_BUFS = ("gu_q", "gu_s", "gu_z", "dn_q", "dn_s", "dn_z")


@torch.no_grad()
def load_w4(ckpt, device="cuda"):
    ckpt = pathlib.Path(ckpt).resolve()
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoTokenizer, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextExperts
    from pack_experts import fused_experts_forward

    assert (ckpt / "w4_meta.json").exists(), (
        f"{ckpt} is not a dg-w4a16 checkpoint (no w4_meta.json). Point this at the directory "
        f"holding model.safetensors + experts_*.safetensors, not at the bf16 base model."
    )
    cfg = AutoConfig.from_pretrained(str(ckpt))
    cfg._attn_implementation = "sdpa"
    with init_empty_weights(include_buffers=False):     # buffers real (inv_freq!), params meta
        model = DiffusionGemmaForBlockDiffusion._from_config(cfg, torch_dtype=torch.bfloat16)

    sd = {}
    for f in sorted(ckpt.glob("model*.safetensors")):   # save_pretrained shards (non-expert)
        sd.update(load_file(str(f), device=device))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    model.tie_weights()                                  # re-alias tied tensors post-assign
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    # `missing` = expert projections (packed below) + tied aliases; the final is_meta sweep is
    # the real completeness check once experts are attached.

    # Encoder and decoder layer N hold the SAME expert weights (verified byte-identical for all 30
    # pairs in the 60-shard export). Bind one loaded tensor to both modules: halves the expert VRAM
    # (~23 -> ~11.5 GiB) and lets the release ship 30 shards instead of 60. Inference is read-only,
    # so aliasing is safe. Files are keyed by layer index, not by enumeration order.
    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, DiffusionGemmaTextExperts)]
    by_layer = {}
    for name, mod in mods:
        m = re.search(r"\.layers\.(\d+)\.experts$", name)
        assert m, f"unrecognized experts module path: {name}"
        by_layer.setdefault(int(m.group(1)), []).append((name, mod))

    shard_of = {}                                        # layer idx -> shard path (first wins)
    for f in sorted(ckpt.glob("experts_*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            k0 = next(iter(h.keys()))
        i = int(re.search(r"\.layers\.(\d+)\.experts\.", k0).group(1))
        shard_of.setdefault(i, (f, k0.rsplit(".", 1)[0]))
    missing_layers = sorted(set(by_layer) - set(shard_of))
    assert not missing_layers, f"no expert shard for layer(s) {missing_layers}"

    for i, (f, src) in sorted(shard_of.items()):
        d = load_file(str(f), device=device)             # loaded ONCE per layer
        for name, mod in by_layer[i]:
            mod.gate_up_proj = None                      # meta placeholders out
            mod.down_proj = None
            for b in EXPERT_BUFS:
                mod.register_buffer(b, d[f"{src}.{b}"], persistent=False)   # same tensor, both modules
            mod.forward = fused_experts_forward.__get__(mod, mod.__class__)

    for n, p in model.named_parameters():
        assert not p.is_meta, f"param never materialized: {n}"

    # `_from_config` does NOT read generation_config.json (only `from_pretrained` does), so without
    # this every sampler field lands as None: confidence_threshold / stability_threshold / t_min /
    # t_max / max_denoising_steps / sampler_config. Denoising then never stops early and burns extra
    # steps per committed token -- same text out, ~38% slower (117.7 vs 188.8 tok/s on gsm8k:50).
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        DiffusionGemmaGenerationConfig,
    )
    model.generation_config = DiffusionGemmaGenerationConfig.from_pretrained(str(ckpt))
    assert model.generation_config.max_denoising_steps is not None, "generation_config failed to load"

    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    meta = json.loads((ckpt / "w4_meta.json").read_text())
    print(f"[load_w4] {meta['format']} | {len(shard_of)} expert shards -> {len(mods)} MoE modules "
          f"(encoder/decoder share weights) | device={device}", flush=True)
    return model, tok


@torch.no_grad()
def load_fast(ckpt, device="cuda", tier="turbo"):
    """One-line fast path: W4 checkpoint + fused kernels, ready for model.generate().

        from load_w4_checkpoint import load_fast
        model, tok = load_fast("..")           # the repo root

    tier:
      "turbo"    (default) fused entropy + Gumbel-max sampler + single-kernel RMSNorm.
                 Accuracy parity with the unmodified W4 model; different RNG stream.
      "bitexact" only the duplicate-entropy dedup, which cannot change output.
                 Token-identical to the unmodified W4 model. ~30% slower than turbo.
      "off"      plain W4, no sampler/norm kernels.
    """
    assert tier in ("turbo", "bitexact", "off"), f"unknown tier {tier!r}"
    model, tok = load_w4(ckpt, device)
    if tier != "off":
        import fast_sampler
        fast_sampler.install(parity=(tier == "turbo"))
        if tier == "turbo":
            from fused_rmsnorm import patch_rmsnorms
            patch_rmsnorms(model, verbose=False)
    print(f"[load_fast] tier={tier}", flush=True)
    return model, tok


if __name__ == "__main__":
    import time
    model, tok = load_w4(sys.argv[1] if len(sys.argv) > 1 else "..")
    ids = tok.apply_chat_template([{"role": "user", "content": "What is 17*23? Show your work."}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)["input_ids"].cuda()
    t = time.time()
    out = model.generate(input_ids=ids, max_new_tokens=256)
    seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
    txt = tok.decode(seq[ids.shape[1]:], skip_special_tokens=True)
    print(f"[smoke] {time.time()-t:.1f}s\n{txt[:400]}")
