"""w4a16.py — REAL packed 4-bit weight / 16-bit activation GEMM for sm_121 (Triton).

The kernel reads PACKED 4-bit weights (2 nibbles/byte) and dequantizes INLINE, so it actually streams
4-bit weight traffic — this is what gives the bandwidth win on the (bandwidth-bound) expert path. NOT a
fake-quant: F.linear(X, dequant(W)) is the reference; the Triton kernel must match it.

Quant = asymmetric group-wise (group_size along the input dim K): per (row, group) scale+zero, 4-bit.
Layout: qweight uint8[N, K//2] (col j = W[:,2j] low nibble | W[:,2j+1] high), scale/zero fp16[N, K//GS].

Start dense (one Linear) for the correctness gate; grouped-MoE version reuses the same pack + kernel.
"""
import torch
import triton
import triton.language as tl

GS = 128  # group size along K (== BLOCK_K so one scale/zero per row per K-chunk)


@torch.no_grad()
def quantize_w4(W, group_size=GS):
    """W[N,K] fp -> (qweight uint8[N,K//2], scale fp16[N,K//gs], zero fp16[N,K//gs]) asymmetric 4-bit."""
    N, K = W.shape
    assert K % group_size == 0
    Wg = W.float().reshape(N, K // group_size, group_size)
    wmin = Wg.min(-1).values
    wmax = Wg.max(-1).values
    scale = (wmax - wmin).clamp(min=1e-8) / 15.0                 # [N, ng]
    zero = (-wmin / scale).round().clamp(0, 15)                  # [N, ng]
    q = (Wg / scale[..., None] + zero[..., None]).round().clamp(0, 15).to(torch.uint8)  # [N,ng,gs]
    q = q.reshape(N, K)
    qpacked = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()      # [N, K//2] uint8
    return qpacked, scale.to(torch.float16).contiguous(), zero.to(torch.float16).contiguous()


@torch.no_grad()
def dequant_w4(qpacked, scale, zero, group_size=GS):
    """Inverse (fp16 reference weight). qpacked[N,K//2] -> W[N,K]."""
    N, Kh = qpacked.shape
    K = Kh * 2
    lo = (qpacked & 0xF).to(torch.float16)
    hi = (qpacked >> 4).to(torch.float16)
    q = torch.empty(N, K, dtype=torch.float16, device=qpacked.device)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    ng = K // group_size
    q = q.reshape(N, ng, group_size)
    W = (q - zero[..., None].float()) * scale[..., None].float()
    return W.reshape(N, K).to(torch.float16)


def _cfgs():
    out = []
    for bm in (16, 32):                       # MoE M/expert is small; big BM wastes rows
        for bn in (128, 256):
            for st in (3, 4):
                for w in (4, 8):
                    out.append(triton.Config({"BM": bm, "BN": bn}, num_stages=st, num_warps=w))
    return out


# key on the WEIGHT SHAPE only (NOT M): in the MoE loop M varies per expert, so keying on M would
# re-autotune every call. One tune per (N,K,BK) shape, reused across all token counts.
@triton.autotune(configs=_cfgs(), key=["N", "K", "BK"])
@triton.jit
def _w4a16_kernel(X, Qw, Scl, Zer, Y, M, N, K,
                  sxm, sxk, sqn, sqk, ssn, ssg, sym, syn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rkh = tl.arange(0, BK // 2)
    acc = tl.zeros((BM, BN), tl.float32)
    nkg = K // BK
    for kb in range(nkg):
        k0 = kb * BK
        x = tl.load(X + rm[:, None] * sxm + (k0 + rk)[None, :] * sxk,
                    mask=(rm[:, None] < M), other=0.0).to(tl.float16)            # [BM,BK]
        # packed weight for [rn, k0:k0+BK] -> uint8[BN, BK//2]
        qb = tl.load(Qw + rn[:, None] * sqn + (k0 // 2 + rkh)[None, :] * sqk,
                     mask=(rn[:, None] < N), other=0)                            # [BN,BK//2] uint8
        lo = (qb & 0xF).to(tl.float16)
        hi = ((qb >> 4) & 0xF).to(tl.float16)
        w = tl.interleave(lo, hi)                                                # [BN,BK] nibble values
        g = k0 // BK                                                             # BK==GS -> one group per chunk
        sc = tl.load(Scl + rn * ssn + g * ssg, mask=rn < N, other=0.0).to(tl.float32)   # [BN]
        ze = tl.load(Zer + rn * ssn + g * ssg, mask=rn < N, other=0.0).to(tl.float32)   # [BN]
        w = (w.to(tl.float32) - ze[:, None]) * sc[:, None]                       # fp32 dequant (matches ref)
        acc += tl.dot(x, tl.trans(w.to(tl.float16)))                            # [BM,BK]@[BK,BN]
    y = acc.to(tl.float16)
    tl.store(Y + rm[:, None] * sym + rn[None, :] * syn, y,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def w4a16_linear(X, qpacked, scale, zero, BK=GS):
    """Y[M,N] = X[M,K] @ dequant(qpacked)[N,K].T  — Triton, reads packed 4-bit, dequant inline.
    Autotuned over tile sizes + num_stages (DRAM-latency pipelining); BK fixed = group size."""
    M, K = X.shape
    N = qpacked.shape[0]
    assert K % BK == 0, "K must be divisible by BK (== group size)"
    Y = torch.empty(M, N, device=X.device, dtype=torch.float16)
    Xc = X.contiguous().to(torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    _w4a16_kernel[grid](Xc, qpacked, scale, zero, Y, M, N, K,
                        Xc.stride(0), Xc.stride(1), qpacked.stride(0), qpacked.stride(1),
                        scale.stride(0), scale.stride(1), Y.stride(0), Y.stride(1), BK=BK)
    return Y


def _killtest():
    torch.manual_seed(0)
    print(f"{'shape (N,K) @ M':>22} {'quant rel(W)':>13} {'kernel-vs-Flin':>15} {'rel':>9}")
    SHAPES = [("dense q", 2816, 2816, 256), ("gate_up", 1408, 2816, 64),
              ("down", 2816, 704, 64), ("lm_head", 4096, 2816, 256)]
    gs_of = lambda K: 128 if K % 128 == 0 else 64                   # 64 divides 704 (down_proj)
    worst = 0.0
    for nm, N, K, M in SHAPES:
        gs = gs_of(K)
        W = (torch.randn(N, K, device="cuda") * 0.02)
        W[:, : K // 40] += torch.randn(N, K // 40, device="cuda") * 0.15
        qp, sc, ze = quantize_w4(W, gs)
        Wdq = dequant_w4(qp, sc, ze, gs)
        qrel = ((Wdq.float() - W).norm() / W.norm()).item()
        X = torch.randn(M, K, device="cuda", dtype=torch.float16)
        y_ref = torch.nn.functional.linear(X, Wdq)                  # == F.linear(X, dequant(W))
        y_ker = w4a16_linear(X, qp, sc, ze, BK=gs)
        d = (y_ker.float() - y_ref.float()).abs().max().item()
        rel = d / (y_ref.float().abs().max().item() + 1e-9)
        worst = max(worst, rel)
        print(f"{f'({N},{K})@{M}':>22} {qrel:>13.4f} {d:>15.2e} {rel:>9.1e} {'OK' if rel < 2e-2 else '**FAIL**'}")
    print(f"\nworst kernel-vs-Flinear rel = {worst:.1e} -> "
          f"{'KERNEL CORRECT (packed 4-bit == dequant ref)' if worst < 2e-2 else 'FAIL — fix before integrating'}")


if __name__ == "__main__":
    _killtest()
