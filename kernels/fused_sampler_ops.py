"""fused_sampler_ops.py — ACCURACY-LOSSLESS (parity-gated) Triton kernels for HF generate's
logits pipeline on [canvas=256, V=262144] fp32. NOT bitwise vs stock (reduction order / RNG
stream differ) — gate = accuracy parity, same standard as fused_rmsnorm.

1. fused_entropy: Categorical(logits).entropy() = logsumexp+sub+exp+mul+sum (5 kernels, ~16ms)
   -> ONE online-softmax pass (~1.1ms bandwidth floor): track running (max m, sum-exp s,
   sum x*exp t); H = lse - t/s.
2. gumbel_argmax_sample: softmax(5ms) + multinomial(9.9ms) + argmax(1.25ms) -> ONE pass.
   argmax(x + G), G Gumbel(0,1) iid, samples Categorical(softmax(x)) EXACTLY (Gumbel-max trick);
   noise from tl.rand (Philox) inline — no 268MB noise tensor. Same kernel also reduces plain
   argmax (second running max) so the stock new_argmax_canvas comes free.
   Determinism: seed = f(question-seed, step) -> reproducible runs; RNG stream differs from
   torch.multinomial by construction (parity tier).
"""
import torch
import triton
import triton.language as tl

BLOCK = 8192


@triton.jit
def _entropy_kernel(X, H, V, sxr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    xb = X + row * sxr
    m = -float("inf")
    s = 0.0
    t = 0.0
    for v0 in range(0, V, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        x = tl.load(xb + offs, mask=offs < V, other=-float("inf")).to(tl.float32)
        cm = tl.max(x, axis=0)
        m_new = tl.maximum(m, cm)
        alpha = tl.exp(m - m_new)
        e = tl.exp(x - m_new)                      # masked lanes: exp(-inf)=0 -> no contribution
        s = s * alpha + tl.sum(e, axis=0)
        t = t * alpha + tl.sum(tl.where(offs < V, x * e, 0.0), axis=0)
        m = m_new
    lse = m + tl.log(s)
    tl.store(H + row, lse - t / s)


def fused_entropy(logits):
    """logits [..., V] fp32/bf16 -> entropy [...] fp32 (nats), one pass."""
    x = logits.reshape(-1, logits.shape[-1])
    if not x.is_contiguous():
        x = x.contiguous()
    rows, V = x.shape
    h = torch.empty(rows, dtype=torch.float32, device=x.device)
    _entropy_kernel[(rows,)](x, h, V, x.stride(0), BLOCK=BLOCK, num_warps=8)
    return h.view(logits.shape[:-1])


@triton.jit
def _gumbel_argmax_kernel(X, Samp, Amax, V, sxr, seed, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    xb = X + row * sxr
    bg = -float("inf")
    bgi = 0
    ba = -float("inf")
    bai = 0
    for v0 in range(0, V, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(xb + offs, mask=mask, other=-float("inf")).to(tl.float32)
        u = tl.rand(seed + row, offs)                            # Philox: per (row, col) iid
        u = tl.minimum(tl.maximum(u, 1e-10), 1.0 - 1e-7)         # guard log(0)
        g = -tl.log(-tl.log(u))
        xg = tl.where(mask, x + g, -float("inf"))
        cbg = tl.max(xg, axis=0)
        cbgi = tl.argmax(xg, axis=0) + v0
        if cbg > bg:
            bg = cbg
            bgi = cbgi
        cba = tl.max(x, axis=0)
        cbai = tl.argmax(x, axis=0) + v0
        if cba > ba:
            ba = cba
            bai = cbai
    tl.store(Samp + row, bgi)
    tl.store(Amax + row, bai)


def gumbel_argmax_sample(logits, seed):
    """logits [..., V] -> (sampled idx, argmax idx) int64 [...]. Gumbel-max == Categorical sample."""
    x = logits.reshape(-1, logits.shape[-1])
    if not x.is_contiguous():
        x = x.contiguous()
    rows, V = x.shape
    samp = torch.empty(rows, dtype=torch.int64, device=x.device)
    amax = torch.empty(rows, dtype=torch.int64, device=x.device)
    _gumbel_argmax_kernel[(rows,)](x, samp, amax, V, x.stride(0), seed, BLOCK=BLOCK, num_warps=8)
    return samp.view(logits.shape[:-1]), amax.view(logits.shape[:-1])


def _killtest():
    torch.manual_seed(0)
    dev = "cuda"
    for rows, V in ((256, 262144), (7, 4097), (1, 262144)):
        x = torch.randn(rows, V, device=dev, dtype=torch.float32) * 3
        # entropy vs stock chain
        ref = torch.distributions.Categorical(logits=x).entropy()
        h = fused_entropy(x)
        rel = ((h - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        print(f"  entropy rows={rows} V={V}: rel_max={rel:.2e} {'OK' if rel < 1e-5 else '**FAIL**'}")
        # argmax exact; sample distribution sanity (chi2-lite on a peaked row)
        samp, amax = gumbel_argmax_sample(x, seed=1234)
        ok_amax = (amax == x.argmax(-1)).all().item()
        print(f"  argmax exact: {'OK' if ok_amax else '**FAIL**'}")
    xp = torch.full((1, 1000), -8.0, device=dev)                  # suppress the tail this time
    xp[0, :4] = torch.tensor([3.0, 2.0, 1.0, 0.0], device=dev)
    p_true = torch.softmax(xp.float(), -1)[0, :4]                 # ~[.643,.237,.087,.032]
    counts = torch.zeros(1000, device=dev)
    n_draw = 4000
    for i in range(n_draw):
        s, _ = gumbel_argmax_sample(xp, seed=i * 7919)
        counts[s[0]] += 1
    p = counts[:4] / n_draw
    tol = 3 * (p_true * (1 - p_true) / n_draw).sqrt() + 1e-3      # 3-sigma binomial band
    ok = ((p - p_true).abs() <= tol).all().item()
    print(f"  gumbel dist: {[round(v, 3) for v in p.tolist()]} vs true "
          f"{[round(v, 3) for v in p_true.tolist()]} {'OK' if ok else '**FAIL**'}")
    # determinism: same seed -> same draw
    s1, _ = gumbel_argmax_sample(xp, seed=42)
    s2, _ = gumbel_argmax_sample(xp, seed=42)
    print(f"  seed determinism: {'OK' if (s1 == s2).all().item() else '**FAIL**'}")


if __name__ == "__main__":
    _killtest()
