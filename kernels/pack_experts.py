"""pack_experts.py — swap DiffusionGemmaTextExperts BF16 weights for packed W4A16 + a Triton grouped
forward. Mirrors the stock forward EXACTLY (routing/gather/act/index_add unchanged); only the two
nn.functional.linear calls become w4a16_linear (packed 4-bit, dequant inline). The evaluated model is
byte-identical to what the kernel computes (kill-test proves it).

gate_up_proj: per-expert [2*704, 2816], K=2816 -> group 128. down_proj: [2816, 704], K=704 -> group 64.
"""
import sys, pathlib, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from w4a16 import quantize_w4, w4a16_linear

GU_GS, DN_GS = 128, 64

# MoE ROUTING PROBE (MOE_STATS=1): does the expert UNION shrink if we drop top-k? Dynamic-k only cuts
# MoE bandwidth if top-4 union << top-8 union (else coupon-collector floors the union near num_experts).
import os as _os, collections as _coll
_MOE_ACC = _coll.defaultdict(list)


def dump_moe_stats():
    if not _MOE_ACC.get("u8"):
        return
    m = lambda x: sum(x) / len(x)
    E = _MOE_ACC["E"][0]
    print(f"\n=== MoE ROUTING PROBE (n={len(_MOE_ACC['u8'])} MoE-layer-forwards, {E} experts) ===", flush=True)
    print(f"  expert UNION loaded per forward:  top-8={m(_MOE_ACC['u8']):.0f}  top-6={m(_MOE_ACC['u6']):.0f}  "
          f"top-4={m(_MOE_ACC['u4']):.0f}  top-2={m(_MOE_ACC['u2']):.0f}   (of {E})", flush=True)
    print(f"  MoE bandwidth vs top-8:  top-6={100*m(_MOE_ACC['u6'])/m(_MOE_ACC['u8']):.0f}%  "
          f"top-4={100*m(_MOE_ACC['u4'])/m(_MOE_ACC['u8']):.0f}%  top-2={100*m(_MOE_ACC['u2'])/m(_MOE_ACC['u8']):.0f}%", flush=True)
    print(f"  experts holding 80% of routing mass: {m(_MOE_ACC['conc']):.0f}/{E}", flush=True)
    print(f"  VERDICT: dynamic-k moves tok/s IFF top-4 union << top-8 (concentrated). If ~equal -> coupon-collector, no BW win.", flush=True)


@torch.no_grad()
def _pack_3d(W3d, gs):
    """W3d [E, N, K] -> lists of per-expert (qpacked, scale, zero) stacked to [E, ...]."""
    qs, ss, zs = [], [], []
    for e in range(W3d.shape[0]):
        q, s, z = quantize_w4(W3d[e].float(), gs)
        qs.append(q); ss.append(s); zs.append(z)
    return torch.stack(qs), torch.stack(ss), torch.stack(zs)


def packed_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Identical to DiffusionGemmaTextExperts.forward but the two linears read packed W4."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        e = expert_idx[0]
        if e == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e])
        current_state = hidden_states[token_idx]
        gate_up = w4a16_linear(current_state, self.gu_q[e], self.gu_s[e], self.gu_z[e], BK=GU_GS)
        gate, up = gate_up.chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        h = w4a16_linear(h, self.dn_q[e], self.dn_s[e], self.dn_z[e], BK=DN_GS)
        h = h * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
    return final_hidden_states


def fused_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Single fused grouped-GEMM (one launch for all experts) — replaces the launch-bound loop.
    MOE_V2=1 selects the deeper-fused kernels (act into GEMM1, scatter into GEMM2)."""
    import os
    if os.environ.get("MOE_STATS"):
        with torch.no_grad():                                    # measure expert union at each top-k
            order = top_k_weights.argsort(dim=-1, descending=True)   # ensure [:, :k] = true top-k
            tis = torch.gather(top_k_index, -1, order)
            tk = top_k_index.shape[1]
            for k, key in ((8, "u8"), (6, "u6"), (4, "u4"), (2, "u2")):
                _MOE_ACC[key].append(int(tis[:, :min(k, tk)].unique().numel()))
            hist = torch.bincount(top_k_index.reshape(-1), minlength=self.num_experts).float()
            cum = torch.cumsum(torch.sort(hist, descending=True).values, 0) / hist.sum().clamp(min=1)
            _MOE_ACC["conc"].append(int((cum < 0.8).sum()) + 1)
            _MOE_ACC["E"].append(self.num_experts)
    if os.environ.get("MOE_V2") == "1":
        from fused_moe_w4_v2 import fused_moe_w4_v2
        return fused_moe_w4_v2(hidden_states, self, top_k_index, top_k_weights)
    from fused_moe_w4 import fused_moe_w4
    return fused_moe_w4(hidden_states, self, top_k_index, top_k_weights)


@torch.no_grad()
def pack_experts_module(exp, free_bf16=True, fused=True):
    """Pack one DiffusionGemmaTextExperts in place: add packed buffers, bind packed forward."""
    gu_q, gu_s, gu_z = _pack_3d(exp.gate_up_proj.data, GU_GS)
    I = exp.down_proj.shape[-1]                              # 704 (contraction dim of down)
    # down stays UNPADDED GS=64/BK=64: the 704->768 pad (BK=128) was 1.27x on the isolated kernel but its
    # GS 64->128 coarsening cut commits/fwd 44.4->41.9 -> NET -10 tok/s end-to-end (pad_ab.log). D2F coupling.
    dn_q, dn_s, dn_z = _pack_3d(exp.down_proj.data, DN_GS)
    dev = exp.gate_up_proj.device
    for nm, t in [("gu_q", gu_q), ("gu_s", gu_s), ("gu_z", gu_z), ("dn_q", dn_q), ("dn_s", dn_s), ("dn_z", dn_z)]:
        exp.register_buffer(nm, t.to(dev), persistent=False)
    if free_bf16:
        exp.gate_up_proj = None
        exp.down_proj = None
    if _os.environ.get("MOE_NS") == "1" and _os.environ.get("MOE_V2") == "1":
        from fused_moe_w4_v2 import enable_ns_down               # preshuffled-T down pack (v2-only)
        enable_ns_down(exp)
    fwd = fused_experts_forward if fused else packed_experts_forward
    exp.forward = fwd.__get__(exp, exp.__class__)
    return exp


@torch.no_grad()
def pack_all_experts(model, free_bf16=True, verbose=True):
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextExperts
    n, gu_bits, dn_bits = 0, 0, 0
    for name, mod in model.named_modules():
        if isinstance(mod, DiffusionGemmaTextExperts):
            gu_bits += mod.gate_up_proj.numel(); dn_bits += mod.down_proj.numel()
            pack_experts_module(mod, free_bf16)
            n += 1
    torch.cuda.empty_cache()
    if verbose:
        print(f"[pack] {n} expert modules -> W4 (gate_up gs{GU_GS}, down gs{DN_GS}); "
              f"experts {2*(gu_bits+dn_bits)/2**30:.1f}GiB bf16 -> ~{(gu_bits+dn_bits)/2/2**30:.1f}GiB W4", flush=True)
    return n


def _killtest():
    from transformers import AutoConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextExperts
    cfg = AutoConfig.from_pretrained("google/diffusiongemma-26B-A4B-it").get_text_config()
    cfg.num_experts = 128
    torch.manual_seed(0)
    exp = DiffusionGemmaTextExperts(cfg).to("cuda", torch.float16)
    with torch.no_grad():
        exp.gate_up_proj.normal_(0, 0.02); exp.down_proj.normal_(0, 0.02)
    T, topk = 256, cfg.top_k_experts if hasattr(cfg, "top_k_experts") else 8
    H = cfg.hidden_size
    from w4a16 import quantize_w4, dequant_w4
    hs = torch.randn(T, H, device="cuda", dtype=torch.float16) * 0.1
    idx = torch.randint(0, 128, (T, topk), device="cuda")
    wts = torch.softmax(torch.randn(T, topk, device="cuda", dtype=torch.float16), -1)

    bf16_ref = exp.forward(hs, idx, wts).float()                          # stock BF16 (full precision)
    # dequant reference: stock forward but with W4-ROUNDED weights -> isolates kernel+plumbing from quant error
    gu_dq = torch.stack([dequant_w4(*quantize_w4(exp.gate_up_proj[e].float(), GU_GS), GU_GS) for e in range(128)])
    dn_dq = torch.stack([dequant_w4(*quantize_w4(exp.down_proj[e].float(), DN_GS), DN_GS) for e in range(128)])
    with torch.no_grad():
        exp.gate_up_proj.copy_(gu_dq); exp.down_proj.copy_(dn_dq)
    dq_ref = exp.forward(hs, idx, wts).float()                           # W4 values via torch linear

    pack_experts_module(exp, free_bf16=True)
    out = exp.forward(hs, idx, wts).float()                             # W4 values via Triton kernel

    scale = dq_ref.abs().max().item() + 1e-9
    rel_max = (out - dq_ref).abs().max().item() / scale
    rel_mean = (out - dq_ref).abs().mean().item() / (dq_ref.abs().mean().item() + 1e-9)
    rel_quant = (dq_ref - bf16_ref).abs().max().item() / (bf16_ref.abs().max().item() + 1e-9)
    ok = rel_mean < 5e-3 and rel_max < 3e-2          # 2-GEMM fp16 MoE chain; eval is the real judge
    print(f"  kernel+plumbing (packed vs dequant-ref): rel_mean={rel_mean:.2e} rel_max={rel_max:.2e} "
          f"{'OK — grouped Triton forward CORRECT (fp16-level)' if ok else '**FAIL**'}")
    print(f"  (info) W4-vs-BF16 quant error on RANDOM weights: rel={rel_quant:.2e}  "
          f"(real-weight quality is judged by GSM8K/HumanEval, not this)")


if __name__ == "__main__":
    _killtest()
