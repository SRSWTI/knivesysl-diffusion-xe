"""fused_moe_w4.py — FUSED grouped W4A16 MoE GEMM (ONE Triton launch for all experts, not 128).

The per-expert Python loop is launch/overhead-bound at M/expert~16 (128 launches/layer swamp the 4x
bandwidth win). vLLM-style fix: sort the (token,expert) pairs by expert into block_m-aligned groups,
then a SINGLE grouped GEMM where each M-block knows its expert and gathers its tokens. Two grouped
GEMMs (gate_up, down) + act + weighted scatter = 2 launches/layer instead of 256.

Packed weights (from pack_experts): gu_q[E,2I,K//2] gu_s/gu_z[E,2I,K//gs]; dn_q[E,K,I//2] dn_s/dn_z[E,K,I//gs].
"""
import torch, triton, triton.language as tl

BLOCK_M = 64


@torch.no_grad()
def moe_align(topk_ids, block_m, E):
    """Sort expanded (token,slot) rows by expert into block_m-aligned groups. SYNC-FREE: fixed max-block
    capacity (no .item()/repeat_interleave), expert_ids via searchsorted (pad blocks -> sentinel E).
    Returns sorted_rows[cap] (expanded-row idx or sentinel=Nexp for pad), expert_ids[max_blocks], cap."""
    T, tk = topk_ids.shape
    dev = topk_ids.device
    flat = torch.clamp(topk_ids.reshape(-1), max=E)              # [Nexp]; dynamic-k marks dropped rows with E
    Nexp = flat.numel()
    valid = flat < E                                             # rows with a REAL expert (E = dropped sentinel)
    order = torch.argsort(flat, stable=True)                     # dropped (==E) sort to the end
    sorted_e = flat[order]
    sv = sorted_e < E                                            # valid mask in sorted order
    counts = torch.zeros(E, device=dev, dtype=torch.long).scatter_add_(   # count ONLY valid rows
        0, torch.where(valid, flat, torch.zeros_like(flat)).long(), valid.long())   # dropped add 0
    nblk_e = (counts + block_m - 1) // block_m                   # [E] blocks per expert
    max_blocks = (Nexp + block_m - 1) // block_m + E             # FIXED upper bound (from shapes, no sync)
    cap = max_blocks * block_m
    cum = torch.cumsum(nblk_e, 0)                                # [E] block-count prefix
    expert_ids = torch.searchsorted(cum, torch.arange(max_blocks, device=dev), right=True).to(torch.int64)
    dst_start = (cum - nblk_e) * block_m                         # [E] row-start per expert
    estart = torch.cumsum(counts, 0) - counts                   # [E]
    se_c = torch.clamp(sorted_e, max=E - 1)                      # safe index for dropped rows (excluded below)
    pos = torch.arange(Nexp, device=dev) - estart[se_c]
    dst = dst_start[se_c] + pos                                 # sorted position (garbage for dropped, unused)
    # CAPTURE-SAFE (CUDA graphs): bool-mask advanced indexing (dst[sv]) is a dynamic-shape op ->
    # cudaErrorStreamCaptureUnsupported. Equivalent FIXED-shape scatter: invalid rows write a sentinel
    # value into a trash slot (index cap / Nexp) that is sliced off. Bit-identical to the old assign.
    sorted_ext = torch.full((cap + 1,), Nexp, dtype=torch.long, device=dev)     # slot [cap] = trash
    sorted_ext.scatter_(0, torch.where(sv, dst, torch.full_like(dst, cap)),
                        torch.where(sv, order, torch.full_like(order, Nexp)))
    sorted_rows = sorted_ext[:cap]                              # sentinel Nexp = pad (semantics unchanged)
    inv_ext = torch.zeros(Nexp + 1, dtype=torch.long, device=dev)               # slot [Nexp] = trash
    inv_ext.scatter_(0, torch.where(sv, order, torch.full_like(order, Nexp)),
                     torch.where(sv, dst, torch.zeros_like(dst)))
    inv = inv_ext[:Nexp]                                        # inv[expanded_row] = its sorted position
    return sorted_rows, expert_ids, cap, inv


@triton.jit
def _grouped_w4_kernel(A, Arow, Wq, Ws, Wz, C, ExpertIds, num_valid, E, N, K,
                       sar, sak, swe, swn, swk, sse, ssn, ssg, scr, scn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pid_m = pid // npn
    pid_n = pid % npn
    e = tl.load(ExpertIds + pid_m)
    if e >= E:                                                  # padding block (fixed-cap over-launch) -> skip
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    a_rows = tl.load(Arow + offs_m)                              # which A row each output row reads
    rmask = a_rows < num_valid
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_kh = tl.arange(0, BK // 2)
    acc = tl.zeros((BM, BN), tl.float32)
    e64 = e.to(tl.int64)                # 64-bit weight base: e*swe overflows int32 for ~8x-DG-size experts
    wq_e = Wq + e64 * swe
    ws_e = Ws + e64 * sse
    wz_e = Wz + e64 * sse
    for kb in range(K // BK):
        k0 = kb * BK
        a = tl.load(A + a_rows[:, None] * sar + (k0 + offs_k)[None, :] * sak,
                    mask=rmask[:, None], other=0.0).to(tl.float16)
        qb = tl.load(wq_e + offs_n[:, None] * swn + (k0 // 2 + offs_kh)[None, :] * swk,
                     mask=offs_n[:, None] < N, other=0)
        lo = (qb & 0xF).to(tl.float16)
        hi = ((qb >> 4) & 0xF).to(tl.float16)
        w = tl.interleave(lo, hi)
        sc = tl.load(ws_e + offs_n * ssn + (k0 // BK) * ssg, mask=offs_n < N, other=0.0).to(tl.float32)
        ze = tl.load(wz_e + offs_n * ssn + (k0 // BK) * ssg, mask=offs_n < N, other=0.0).to(tl.float32)
        w = (w.to(tl.float32) - ze[:, None]) * sc[:, None]
        acc += tl.dot(a, tl.trans(w.to(tl.float16)))
    tl.store(C + offs_m[:, None] * scr + offs_n[None, :] * scn, acc.to(tl.float16),
             mask=rmask[:, None] & (offs_n[None, :] < N))


def _grouped_w4(A, a_row, Wq, Ws, Wz, C, expert_ids, num_valid, E, N, K, gs, BN=128):
    max_blocks = expert_ids.shape[0]
    grid = (max_blocks * triton.cdiv(N, BN),)                   # fixed (no sync); pad blocks early-exit
    _grouped_w4_kernel[grid](
        A, a_row.to(torch.int64), Wq, Ws, Wz, C, expert_ids.to(torch.int64), num_valid, E, N, K,
        A.stride(0), A.stride(1), Wq.stride(0), Wq.stride(1), Wq.stride(2),
        Ws.stride(0), Ws.stride(1), Ws.stride(2), C.stride(0), C.stride(1),
        BM=BLOCK_M, BN=BN, BK=gs, num_warps=4, num_stages=3)
    return C


@torch.no_grad()
def fused_moe_w4(hidden, exp, topk_ids, topk_w):
    """One MoE layer, fused. exp carries packed buffers gu_q/gu_s/gu_z, dn_q/dn_s/dn_z + dims."""
    T, K = hidden.shape
    tk = topk_ids.shape[1]
    E, twoI, _ = exp.gu_q.shape
    I = twoI // 2
    Nexp = T * tk
    sorted_rows, expert_ids, cap, inv = moe_align(topk_ids, BLOCK_M, E)
    # gate_up: each output row reads hidden[token]; token = row//tk; pad rows -> T (masked by num_valid=T)
    a_row_gu = torch.where(sorted_rows < Nexp, sorted_rows // tk, T)
    out1 = torch.zeros(cap, twoI, device=hidden.device, dtype=torch.float16)
    _grouped_w4(hidden.to(torch.float16), a_row_gu, exp.gu_q, exp.gu_s, exp.gu_z, out1, expert_ids,
                num_valid=T, E=E, N=twoI, K=K, gs=128)
    gate, up = out1.chunk(2, dim=-1)
    out2 = (exp.act_fn(gate) * up).contiguous()                 # [cap, I]; pad rows are 0
    # down: row p reads out2[p] (sorted layout); pad rows are 0 -> harmless
    a_row_dn = torch.arange(cap, device=hidden.device)
    out3 = torch.zeros(cap, K, device=hidden.device, dtype=torch.float16)
    _grouped_w4(out2, a_row_dn, exp.dn_q, exp.dn_s, exp.dn_z, out3, expert_ids,
                num_valid=cap, E=E, N=K, K=I, gs=64)
    # Deterministic weighted combine (NOT index_add_: duplicate-token CUDA atomics are nondeterministic
    # on this GPU and fork decode trajectories). Expanded-row order is (t0,k0)..(t0,k_{tk-1}),(t1,...) .
    out_r = out3[inv].view(T, tk, K)                            # [T, tk, K]
    return (out_r * topk_w.to(out_r.dtype).unsqueeze(-1)).sum(dim=1).to(hidden.dtype)


def _bench():
    import sys, pathlib, time
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from transformers import AutoConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextExperts
    from pack_experts import pack_experts_module, packed_experts_forward
    cfg = AutoConfig.from_pretrained("google/diffusiongemma-26B-A4B-it").get_text_config()
    cfg.num_experts = 128
    torch.manual_seed(0)
    T, tk, H = 256, 8, cfg.hidden_size
    def mk():
        e = DiffusionGemmaTextExperts(cfg).cuda().half()
        with torch.no_grad():
            e.gate_up_proj.normal_(0, 0.02); e.down_proj.normal_(0, 0.02)
        return e
    exp_bf = mk()
    hs = torch.randn(T, H, device='cuda', dtype=torch.float16) * 0.1
    idx = torch.randint(0, 128, (T, tk), device='cuda')
    wts = torch.softmax(torch.randn(T, tk, device='cuda', dtype=torch.float16), -1)
    bf_ref = exp_bf.forward(hs, idx, wts).float()
    exp = mk(); 
    with torch.no_grad(): exp.gate_up_proj.copy_(exp_bf.gate_up_proj); exp.down_proj.copy_(exp_bf.down_proj)
    pack_experts_module(exp, free_bf16=True)
    loop = exp.forward(hs, idx, wts).float()
    fused = fused_moe_w4(hs, exp, idx, wts).float()
    rmean = ((fused - loop).abs().mean() / (loop.abs().mean() + 1e-9)).item()
    rmax = ((fused - loop).abs().max() / (loop.abs().max() + 1e-9)).item()
    qmean = ((fused - bf_ref).abs().mean() / (bf_ref.abs().mean() + 1e-9)).item()
    print(f"  fused vs W4-loop: rel_mean={rmean:.2e} rel_max={rmax:.2e} "
          f"{'OK' if rmean < 5e-3 else '**FAIL**'}   (vs bf16 quant err {qmean:.2e})")
    def t(fn, it=80):
        fn(); torch.cuda.synchronize(); s = time.time()
        for _ in range(it): fn()
        torch.cuda.synchronize(); return (time.time() - s) / it * 1e3
    mb = t(lambda: exp_bf.forward(hs, idx, wts))
    ml = t(lambda: exp.forward(hs, idx, wts))
    mf = t(lambda: fused_moe_w4(hs, exp, idx, wts))
    print(f"  per MoE-layer:  bf16-loop {mb:.3f}ms | W4-loop {ml:.3f}ms ({mb/ml:.2f}x) | "
          f"W4-fused {mf:.3f}ms ({mb/mf:.2f}x vs bf16)")


if __name__ == "__main__":
    _bench()
