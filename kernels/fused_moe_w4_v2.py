"""fused_moe_w4_v2.py — DEEPER-fused W4 MoE: fold `act` into GEMM1's epilogue.
Combine is DETERMINISTIC: store down-proj to [cap,K], gather via inv, then view[T,tk,K].sum —
never index_add_/tl.atomic_add (duplicate-token atomics forked decode trajectories;
fork_block: W4+index_add = nondet; W4 off / view+sum = identical).

- GEMM1+act: per output tile compute BOTH gate and up, write out2 = gelu_tanh(gate)*up.
- GEMM2: down-proj into out3[cap,K]; weighted sum over top-k in fixed slot order.

Reuses moe_align from fused_moe_w4.
"""
import glob, os
from pathlib import Path
import torch, triton, triton.language as tl
from fused_moe_w4 import moe_align

# --- optional CUDA align accelerator (bit-exact, deterministic; +3-7% whole-forward) ---
# Replaces the 14-torch-op moe_align with marlin's single-kernel moe_align_block_size + one
# Triton finalize kernel. Falls back to the pure-torch align when the .so is absent, so the
# kernel stays dependency-free. Disable with MOE_CUDA_ALIGN=0.
# GOTCHA: marlin's align writes -1 for padding blocks (our sentinel is E) -> remapped below.
_AOPS = None
if os.environ.get("MOE_CUDA_ALIGN", "1") == "1":
    try:
        _so = glob.glob(str(Path.home() / ".cache/torch_extensions/*/marlin_aux_gb10/marlin_aux_gb10.so"))
        if _so:
            torch.ops.load_library(_so[0])
            _AOPS = torch.ops.marlin_aux_gb10
    except Exception:
        _AOPS = None


@triton.jit
def _align_finalize(SortedIds, Inv, ARow, Eids, Nexp, T, tk, mx, nblk, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < mx
    s = tl.load(SortedIds + offs, mask=m, other=Nexp)
    valid = s < Nexp
    tl.store(Inv + s, offs, mask=m & valid)              # inv[s] = sorted position
    tl.store(ARow + offs, tl.where(valid, s // tk, T), mask=m)
    m2 = offs < nblk                                     # fold marlin's -1 pad sentinel -> our E here
    e = tl.load(Eids + offs, mask=m2, other=0)           # (was a torch.where = 1 extra launch/layer)
    tl.store(Eids + offs, tl.where(e < 0, E, e), mask=m2)


@torch.no_grad()
def moe_align_cuda(topk_ids, block_m, E):
    """CUDA-align drop-in: returns (a_row i32[cap], expert_ids i32[nblk], cap, inv i32[Nexp]).
    CONTRACT: every id must be a REAL expert (0 <= id < E). The torch fallback tolerates the
    dynamic-k drop sentinel (id == E); this path does NOT — marlin's align indexes count arrays
    of size E (OOB write) and missing pairs would leave inv uninitialized (garbage gather in the
    combine). If a caller ever emits sentinels, run with MOE_CUDA_ALIGN=0. MOE_VALIDATE=1 adds a
    synchronous range assert (debug only — it syncs)."""
    if os.environ.get("MOE_VALIDATE") == "1":
        mn, mxv = int(topk_ids.min()), int(topk_ids.max())
        assert 0 <= mn and mxv < E, f"moe_align_cuda: ids must be in [0,{E}); got [{mn},{mxv}]"
    T, tk = topk_ids.shape
    dev = topk_ids.device
    Nexp = T * tk
    mx = Nexp + E * (block_m - 1)
    if Nexp < E:
        mx = min(Nexp * block_m, mx)
    nblk = (mx + block_m - 1) // block_m
    cap = nblk * block_m
    sorted_ids = torch.full((cap,), Nexp, dtype=torch.int32, device=dev)
    expert_ids = torch.full((nblk,), E, dtype=torch.int32, device=dev)
    num_post = torch.empty(1, dtype=torch.int32, device=dev)
    _AOPS.moe_align_block_size(topk_ids.to(torch.int32), E, block_m,
                               sorted_ids[:mx], expert_ids, num_post, None)
    inv = torch.empty(Nexp, dtype=torch.int32, device=dev)
    a_row = torch.empty(cap, dtype=torch.int32, device=dev)
    _align_finalize[(triton.cdiv(cap, 1024),)](sorted_ids, inv, a_row, expert_ids, Nexp, T, tk,
                                               cap, nblk, E, BLOCK=1024, num_warps=4)
    return a_row, expert_ids, cap, inv

# Per-arch launch configs.
# GB10 (sm_121) keeps the shipped originals byte-for-byte — records unaffected.
_CC = torch.cuda.get_device_capability() if torch.cuda.is_available() else (12, 1)
if _CC[0] == 12 and _CC[1] == 0:                     # sm_120: RTX 5090 / Blackwell consumer
    # Swept + e2e-gated on a 5090 (torch 2.13/triton 3.7): BM=16 is 1.08x on the isolated layer and
    # +3.4%/+3.9% end-to-end on gsm8k/humaneval vs the BM=32 fallback, accuracy unchanged (48/50 both).
    # Smaller BM wins because M/expert is ~16 at canvas 256, so BM=32 tiles are half padding.
    BM_V2 = 16
    G1 = dict(BN=64, num_warps=4, num_stages=3)
    G2 = dict(BN=128, num_warps=8, num_stages=5)
else:                                               # GB10 sm_121 (the tuned home) + safe default
    BM_V2 = 32      # swept (bench_sweep.py): BM=64 tiles were 75% padding at M/expert≈16; 32 → 1.58x
    G1 = dict(BN=64, num_warps=4, num_stages=5)     # gate_up+act: 209 GB/s (84% GB10 peak)
    G2 = dict(BN=128, num_warps=8, num_stages=5)    # down: 158 GB/s
# NOTE: w4 s8 microbenched +20.7% @T=256 (w4_down_sweep.py) but REGRESSED e2e:
# GSM8K n=50 45/50 @ 197.2 tok/s vs 254.1 shipped (same acc, same fwd/q) — triton_s8_gsm50.log.
# Deeper pipelines cost shared memory -> fewer CTAs/SM; serving spans many more window sizes
# than the T=256/512 the sweep covered. DO NOT retune this from microbench alone: gate on e2e.


@triton.jit
def _gateup_act_kernel(A, Arow, Wq, Ws, Wz, C, ExpertIds, num_valid, E, I, H,
                       sar, sak, swe, swn, swk, sse, ssn, ssg, scr, scn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    npn = tl.cdiv(I, BN)
    pid_m = pid // npn
    pid_n = pid % npn
    e = tl.load(ExpertIds + pid_m)
    if e >= E:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    a_rows = tl.load(Arow + offs_m)
    rmask = a_rows < num_valid
    offs_n = pid_n * BN + tl.arange(0, BN)                       # out2 cols in [0,I)
    offs_k = tl.arange(0, BK)
    offs_kh = tl.arange(0, BK // 2)
    accg = tl.zeros((BM, BN), tl.float32)
    accu = tl.zeros((BM, BN), tl.float32)
    e64 = e.to(tl.int64)                # e*swe overflows int32 once E*2I*H/2 >= 2^31 (~8x DG size)
    wq_e = Wq + e64 * swe
    ws_e = Ws + e64 * sse
    wz_e = Wz + e64 * sse
    for kb in range(H // BK):
        k0 = kb * BK
        a = tl.load(A + a_rows[:, None] * sar + (k0 + offs_k)[None, :] * sak,
                    mask=rmask[:, None], other=0.0).to(tl.float16)
        # gate weight rows = offs_n
        qg = tl.load(wq_e + offs_n[:, None] * swn + (k0 // 2 + offs_kh)[None, :] * swk, mask=offs_n[:, None] < I, other=0)
        wg = tl.interleave((qg & 0xF).to(tl.float16), ((qg >> 4) & 0xF).to(tl.float16))
        scg = tl.load(ws_e + offs_n * ssn + (k0 // BK) * ssg, mask=offs_n < I, other=0.0).to(tl.float16)
        zg = tl.load(wz_e + offs_n * ssn + (k0 // BK) * ssg, mask=offs_n < I, other=0.0).to(tl.float16)
        accg += tl.dot(a, tl.trans((wg - zg[:, None]) * scg[:, None]))
        # up weight rows = I + offs_n
        offu = I + offs_n
        qu = tl.load(wq_e + offu[:, None] * swn + (k0 // 2 + offs_kh)[None, :] * swk, mask=offu[:, None] < 2 * I, other=0)
        wu = tl.interleave((qu & 0xF).to(tl.float16), ((qu >> 4) & 0xF).to(tl.float16))
        scu = tl.load(ws_e + offu * ssn + (k0 // BK) * ssg, mask=offu < 2 * I, other=0.0).to(tl.float16)
        zu = tl.load(wz_e + offu * ssn + (k0 // BK) * ssg, mask=offu < 2 * I, other=0.0).to(tl.float16)
        accu += tl.dot(a, tl.trans((wu - zu[:, None]) * scu[:, None]))
    inner = 0.7978845608028654 * (accg + 0.044715 * accg * accg * accg)
    tanh = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0             # stable tanh (tl.tanh absent)
    out = (0.5 * accg * (1.0 + tanh) * accu).to(tl.float16)
    tl.store(C + offs_m[:, None] * scr + offs_n[None, :] * scn, out, mask=rmask[:, None] & (offs_n[None, :] < I))


# Tile ordering: m-major (pid_m = pid // npn) is MEASURED optimal — consecutive CTAs share
# the same A slice (L2-hot); n-major "swizzle" for weight reuse is 12-15% WORSE (swizzle_probe.py:
# reuse factor <=2 blocks/expert never pays for the A-locality loss).
@triton.jit
def _down_kernel(A, Arow, Wq, Ws, Wz, C, ExpertIds, num_valid, E, N, K,
                 sar, sak, swe, swn, swk, sse, ssn, ssg, scr, scn,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Down-proj only — NO atomic scatter (that was the trajectory fork)."""
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pid_m = pid // npn
    pid_n = pid % npn
    e = tl.load(ExpertIds + pid_m)
    if e >= E:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    a_rows = tl.load(Arow + offs_m)
    rmask = a_rows < num_valid
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_kh = tl.arange(0, BK // 2)
    acc = tl.zeros((BM, BN), tl.float32)
    e64 = e.to(tl.int64)                # 64-bit weight base: see gate_up kernel note
    wq_e = Wq + e64 * swe
    ws_e = Ws + e64 * sse
    wz_e = Wz + e64 * sse
    for kb in range(K // BK):
        k0 = kb * BK
        a = tl.load(A + a_rows[:, None] * sar + (k0 + offs_k)[None, :] * sak, mask=rmask[:, None], other=0.0).to(tl.float16)
        qb = tl.load(wq_e + offs_n[:, None] * swn + (k0 // 2 + offs_kh)[None, :] * swk, mask=offs_n[:, None] < N, other=0)
        w = tl.interleave((qb & 0xF).to(tl.float16), ((qb >> 4) & 0xF).to(tl.float16))
        sc = tl.load(ws_e + offs_n * ssn + (k0 // BK) * ssg, mask=offs_n < N, other=0.0).to(tl.float16)
        ze = tl.load(wz_e + offs_n * ssn + (k0 // BK) * ssg, mask=offs_n < N, other=0.0).to(tl.float16)
        acc += tl.dot(a, tl.trans((w - ze[:, None]) * sc[:, None]))
    tl.store(C + offs_m[:, None] * scr + offs_n[None, :] * scn,
             acc.to(tl.float16), mask=rmask[:, None] & (offs_n[None, :] < N))


# --- MOE_NS=1: nibble-preshuffled + transposed DOWN pack (the Marlin repack trick in Triton) ---
# Probe verdict (nibble_shuffle_probe.py, GB10 2026-07-23): dn-B = 1.07-1.09x layer at T<=256
# (0.90x at T=1024 — decode is T=256, prefill is one-shot); gate_up variant HURTS (0.94x), so only
# the down GEMM converts. Pack: within each BK=64 chunk, byte j = (q[k0+j] | q[k0+32+j]<<4), stored
# TRANSPOSED [E, K/2, N] -> the k-loop needs zero tl.interleave and zero tl.trans: two half-tiles
# feed two tl.dot's. NOT bit-comparable to the shipped down (two dots reorder k-accumulation) but
# run-to-run deterministic (fork_block contract holds).
@torch.no_grad()
def preshuffle_T_dn(qpacked, BK=64):
    """Shipped dn pack [E,N,K/2] -> preshuffled TRANSPOSED [E,K/2,N] for _down_nsT_kernel."""
    E_, N, Kh = qpacked.shape
    K = Kh * 2
    codes = torch.empty(E_, N, K, dtype=torch.uint8, device=qpacked.device)
    codes[..., 0::2] = qpacked & 0xF
    codes[..., 1::2] = qpacked >> 4
    c = codes.reshape(E_, N, K // BK, BK)
    ns = (c[..., : BK // 2] | (c[..., BK // 2:] << 4)).reshape(E_, N, Kh)
    return ns.transpose(1, 2).contiguous()


def enable_ns_down(exp):
    """Swap exp.dn_q to the preshuffled-transposed layout in place (idempotent via dn_ns flag)."""
    if getattr(exp, "dn_ns", False):
        return exp
    exp.dn_q = preshuffle_T_dn(exp.dn_q)
    exp.dn_ns = True
    return exp


@triton.jit
def _down_nsT_kernel(A, Arow, Wq, Ws, Wz, C, ExpertIds, num_valid, E, N, K,
                     sar, sak, swe, swk, swn, sse, ssn, ssg, scr, scn,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Down-proj on the MOE_NS pack: lo-nibbles = first half-chunk, hi = second; weight tile loads
    [BK/2, BN] directly (transposed pack) so tl.dot consumes it with no register permutes."""
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pid_m = pid // npn
    pid_n = pid % npn
    e = tl.load(ExpertIds + pid_m)
    if e >= E:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    a_rows = tl.load(Arow + offs_m)
    rmask = a_rows < num_valid
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_kh = tl.arange(0, BK // 2)
    acc = tl.zeros((BM, BN), tl.float32)
    e64 = e.to(tl.int64)                # 64-bit weight base: see gate_up kernel note
    wq_e = Wq + e64 * swe
    ws_e = Ws + e64 * sse
    wz_e = Wz + e64 * sse
    for kb in range(K // BK):           # gs == BK -> one scale/zero per chunk (kb IS the group)
        k0 = kb * BK
        a_lo = tl.load(A + a_rows[:, None] * sar + (k0 + offs_kh)[None, :] * sak,
                       mask=rmask[:, None], other=0.0).to(tl.float16)
        a_hi = tl.load(A + a_rows[:, None] * sar + (k0 + BK // 2 + offs_kh)[None, :] * sak,
                       mask=rmask[:, None], other=0.0).to(tl.float16)
        qb = tl.load(wq_e + (k0 // 2 + offs_kh)[:, None] * swk + offs_n[None, :] * swn,
                     mask=offs_n[None, :] < N, other=0)
        sc = tl.load(ws_e + offs_n * ssn + kb * ssg, mask=offs_n < N, other=0.0).to(tl.float16)
        ze = tl.load(wz_e + offs_n * ssn + kb * ssg, mask=offs_n < N, other=0.0).to(tl.float16)
        acc += tl.dot(a_lo, ((qb & 0xF).to(tl.float16) - ze[None, :]) * sc[None, :])
        acc += tl.dot(a_hi, (((qb >> 4) & 0xF).to(tl.float16) - ze[None, :]) * sc[None, :])
    tl.store(C + offs_m[:, None] * scr + offs_n[None, :] * scn,
             acc.to(tl.float16), mask=rmask[:, None] & (offs_n[None, :] < N))


@triton.jit
def _combine_kernel(Out3, Inv, W, Y, K, s3r, syr,
                    TK: tl.constexpr, BN: tl.constexpr, OUT_BF16: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BN + tl.arange(0, BN)
    mask = offs < K
    acc = tl.zeros((BN,), tl.float32)
    for k in range(TK):                                          # static, fixed order -> deterministic
        r = tl.load(Inv + t * TK + k)
        w = tl.load(W + t * TK + k).to(tl.float32)
        acc += w * tl.load(Out3 + r * s3r + offs, mask=mask, other=0.0).to(tl.float32)
    if OUT_BF16:
        tl.store(Y + t * syr + offs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(Y + t * syr + offs, acc.to(tl.float16), mask=mask)


_IOTA = {}


def _iota(n, dev):
    """Cached arange (grow-and-slice). The down GEMM's a_row is always identity; allocating it
    fresh cost one launch/layer on a box where launches are ~15-25us."""
    buf = _IOTA.get(str(dev))
    if buf is None or buf.numel() < n:
        buf = torch.arange(max(n, 16384), device=dev)
        _IOTA[str(dev)] = buf
    return buf[:n]


def _combine(out3, inv, topk_w, T, tk, K, out_dtype):
    y = torch.empty(T, K, device=out3.device, dtype=out_dtype)
    # inv rides in its native dtype (i32 CUDA align / i64 torch align) and topk_w in its native
    # float dtype: the kernel indexes and upcasts in-register, so the old host-side .to(int64)/
    # .to(float32) copies were pure launch overhead. Bit-exact: fp16/bf16 -> fp32 scalar upcast
    # is lossless, so the fixed-order accumulation sees identical values.
    _combine_kernel[(T, triton.cdiv(K, 256))](
        out3, inv, topk_w.contiguous(), y, K,
        out3.stride(0), y.stride(0), TK=tk, BN=256, OUT_BF16=(out_dtype == torch.bfloat16), num_warps=4)
    return y


_TANH_GELU_NAMES = ("gelu_pytorch_tanh", "gelu_new", "gelu_tanh", "quick_gelu_tanh")


def _check_act(exp):
    """One-time guard: _gateup_act_kernel's epilogue IS tanh-GELU. Verify the module agrees."""
    fn = getattr(exp, "act_fn", None)
    name = getattr(fn, "__name__", None) or type(fn).__name__ if fn is not None else None
    ok = name in _TANH_GELU_NAMES or "gelu" in str(name).lower()
    if not ok and fn is not None:                        # fall back to a numeric probe
        x = torch.tensor([-1.0, 0.0, 0.5, 2.0], dtype=torch.float32, device=hidden_device(exp))
        ref = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))
        ok = torch.allclose(fn(x).float(), ref, atol=1e-3)
    assert ok, (
        f"fused_moe_w4_v2 hardcodes tanh-GELU but this module's act_fn is {name!r}. "
        "Use fused_moe_w4 (MOE_V2=0) or extend the kernel epilogue."
    )
    exp._act_checked = True
    return True


def hidden_device(exp):
    for b in ("gu_s", "gu_q"):
        t = getattr(exp, b, None)
        if t is not None:
            return t.device
    return "cpu"


@torch.no_grad()
def fused_moe_w4_v2(hidden, exp, topk_ids, topk_w):
    T, K = hidden.shape
    tk = topk_ids.shape[1]
    E, twoI, _ = exp.gu_q.shape
    I = twoI // 2
    # The k-loops are UNMASKED for speed — these divisibility contracts are load-bearing.
    # (An unmasked BK=128 with I=704 once silently dropped the last 64 channels: -14 acc.)
    assert K % 128 == 0, f"hidden dim {K} must be divisible by 128 (gate_up BK)"
    assert I % 64 == 0, f"expert intermediate {I} must be divisible by 64 (down BK / gs)"
    # _gateup_act_kernel hardcodes tanh-GELU in its epilogue and never consults exp.act_fn (v1 does).
    # Correct for DiffusionGemma, but a config with a different activation would silently compute the
    # wrong thing, so pin it here rather than discovering it in an eval.
    assert getattr(exp, "_act_checked", False) or _check_act(exp), "unsupported activation"
    Nexp = T * tk
    dev = hidden.device
    if _AOPS is not None:
        a_row_gu, expert_ids, cap, inv = moe_align_cuda(topk_ids, BM_V2, E)   # bit-exact, +3-7% fwd
    else:
        sorted_rows, expert_ids, cap, inv = moe_align(topk_ids, BM_V2, E)
        a_row_gu = torch.where(sorted_rows < Nexp, sorted_rows // tk, T).to(torch.int64)
        expert_ids = expert_ids.to(torch.int64)
    max_blocks = expert_ids.shape[0]
    # GEMM1 + act -> out2 [cap, I]. EMPTY not zeros: pad rows are never GATHERED downstream (inv maps
    # only valid rows) — garbage stays confined to pad rows (all downstream ops are row-independent).
    # cap is block-padding-inflated (~8x Nexp at small q): zeros cost ~46MB memset/layer = ~4ms/fwd.
    out2 = torch.empty(cap, I, device=dev, dtype=torch.float16)
    _gateup_act_kernel[(max_blocks * triton.cdiv(I, G1["BN"]),)](
        hidden.to(torch.float16), a_row_gu, exp.gu_q, exp.gu_s, exp.gu_z, out2,
        expert_ids, T, E, I, K,
        hidden.stride(0), hidden.stride(1), exp.gu_q.stride(0), exp.gu_q.stride(1), exp.gu_q.stride(2),
        exp.gu_s.stride(0), exp.gu_s.stride(1), exp.gu_s.stride(2), out2.stride(0), out2.stride(1),
        BM=BM_V2, BN=G1["BN"], BK=128, num_warps=G1["num_warps"], num_stages=G1["num_stages"])
    # GEMM2 -> out3 [cap, K]; deterministic weighted sum over top-k (no index_add_/atomics)
    a_row_dn = _iota(cap, dev)
    out3 = torch.empty(cap, K, device=dev, dtype=torch.float16)      # empty: pad rows never gathered
    if getattr(exp, "dn_ns", False):                                 # MOE_NS pack [E, K/2, N]
        _down_nsT_kernel[(max_blocks * triton.cdiv(K, G2["BN"]),)](
            out2, a_row_dn, exp.dn_q, exp.dn_s, exp.dn_z, out3,
            expert_ids, cap, E, K, I,
            out2.stride(0), out2.stride(1), exp.dn_q.stride(0), exp.dn_q.stride(1), exp.dn_q.stride(2),
            exp.dn_s.stride(0), exp.dn_s.stride(1), exp.dn_s.stride(2), out3.stride(0), out3.stride(1),
            BM=BM_V2, BN=G2["BN"], BK=64, num_warps=G2["num_warps"], num_stages=G2["num_stages"])
    else:
        _down_kernel[(max_blocks * triton.cdiv(K, G2["BN"]),)](
            out2, a_row_dn, exp.dn_q, exp.dn_s, exp.dn_z, out3,
            expert_ids, cap, E, K, I,
            out2.stride(0), out2.stride(1), exp.dn_q.stride(0), exp.dn_q.stride(1), exp.dn_q.stride(2),
            exp.dn_s.stride(0), exp.dn_s.stride(1), exp.dn_s.stride(2), out3.stride(0), out3.stride(1),
            BM=BM_V2, BN=G2["BN"], BK=64, num_warps=G2["num_warps"], num_stages=G2["num_stages"])
    # Deterministic combine, ONE kernel (was gather+view+mul+sum = 3-4 kernels + temps): fixed k-order
    # loop per token — same reduction order every run, no atomics.
    return _combine(out3, inv, topk_w, T, tk, K, hidden.dtype)


def _killtest():
    import sys, pathlib, time
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from transformers import AutoConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextExperts
    from pack_experts import pack_experts_module
    from fused_moe_w4 import fused_moe_w4
    cfg = AutoConfig.from_pretrained("google/diffusiongemma-26B-A4B-it").get_text_config(); cfg.num_experts = 128
    torch.manual_seed(0)
    T, tk, H = 256, 8, cfg.hidden_size
    e = DiffusionGemmaTextExperts(cfg).cuda().half()
    with torch.no_grad():
        e.gate_up_proj.normal_(0, 0.02); e.down_proj.normal_(0, 0.02)
    hs = torch.randn(T, H, device='cuda', dtype=torch.float16) * 0.1
    idx = torch.randint(0, 128, (T, tk), device='cuda')
    wts = torch.softmax(torch.randn(T, tk, device='cuda', dtype=torch.float16), -1)
    pack_experts_module(e)
    v1 = fused_moe_w4(hs, e, idx, wts).float()
    v2 = fused_moe_w4_v2(hs, e, idx, wts).float()
    rmean = ((v2 - v1).abs().mean() / (v1.abs().mean() + 1e-9)).item()
    rmax = ((v2 - v1).abs().max() / (v1.abs().max() + 1e-9)).item()
    print(f"  v2 (deep-fused) vs v1: rel_mean={rmean:.2e} rel_max={rmax:.2e} "
          f"{'OK' if rmean < 5e-3 else '**FAIL**'}")
    # self-stability: same inputs twice must be bit-identical (the fork_block contract)
    a = fused_moe_w4_v2(hs, e, idx, wts)
    b = fused_moe_w4_v2(hs, e, idx, wts)
    mism = (a.view(torch.int16) != b.view(torch.int16)).sum().item()
    print(f"  v2 self bit-mismatch: {mism}/{a.numel()} {'OK' if mism == 0 else '**FAIL**'}")
    def t(fn, it=100):
        fn(); torch.cuda.synchronize(); s = time.time()
        for _ in range(it): fn()
        torch.cuda.synchronize(); return (time.time() - s) / it * 1e3
    m1 = t(lambda: fused_moe_w4(hs, e, idx, wts))
    m2 = t(lambda: fused_moe_w4_v2(hs, e, idx, wts))
    print(f"  per MoE-layer:  v1 {m1:.3f}ms | v2 {m2:.3f}ms ({m1/m2:.2f}x)")
    # MOE_NS branch: repack down in place, gate numerics + self-determinism + timing
    enable_ns_down(e)
    n1 = fused_moe_w4_v2(hs, e, idx, wts)
    rm_ns = ((n1.float() - v2).abs().mean() / (v2.abs().mean() + 1e-9)).item()
    n2 = fused_moe_w4_v2(hs, e, idx, wts)
    mism_ns = (n1.view(torch.int16) != n2.view(torch.int16)).sum().item()
    m3 = t(lambda: fused_moe_w4_v2(hs, e, idx, wts))
    print(f"  v2+NS (preshuffled-T down): rel_mean={rm_ns:.2e} {'OK' if rm_ns < 5e-3 else '**FAIL**'} | "
          f"self bit-mismatch {mism_ns}/{n1.numel()} {'OK' if mism_ns == 0 else '**FAIL**'} | "
          f"{m3:.3f}ms ({m2/m3:.2f}x vs v2)")


if __name__ == "__main__":
    _killtest()
