"""fused_rmsnorm.py — single-kernel RMSNorm for DiffusionGemma. The stock forward is
`x.float()` -> pow -> mean -> pow(-0.5) -> mul -> `*weight.float()` -> type_as = ~6 fp32 elementwise
kernel launches; ×180 norms/forward = ~1000 tiny underutilized launches = ~20ms/forward. This fuses it
to ONE Triton kernel (one pass, fp32 accumulate, bf16 out). Matches semantics: x/sqrt(mean(x^2)+eps)
[*weight if with_scale]. Kill-tested vs stock.
"""
import torch, triton, triton.language as tl


@triton.jit
def _rmsnorm_kernel(X, W, Y, eps, n_cols, HAS_W: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / n_cols + eps
    y = x * (1.0 / tl.sqrt(ms))                            # == pow(ms,-0.5) to ~1 ulp
    if HAS_W:
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    tl.store(Y + row * n_cols + cols, y.to(tl.bfloat16), mask=mask)


def fused_rmsnorm(x, weight, eps):
    orig = x.shape
    x2 = x.reshape(-1, orig[-1]).contiguous()
    rows, n_cols = x2.shape
    y = torch.empty_like(x2, dtype=torch.bfloat16)
    BLOCK = triton.next_power_of_2(n_cols)
    _rmsnorm_kernel[(rows,)](x2, weight if weight is not None else x2, y, eps, n_cols,
                             HAS_W=weight is not None, BLOCK=BLOCK, num_warps=8 if BLOCK >= 2048 else 4)
    return y.reshape(orig).to(x.dtype)


def _fused_forward(self, x):
    return fused_rmsnorm(x, self.weight if self.with_scale else None, self.eps)


@torch.no_grad()
def patch_rmsnorms(model, verbose=True):
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaRMSNorm
    n = 0
    for mod in model.modules():
        if isinstance(mod, DiffusionGemmaRMSNorm):
            mod.forward = _fused_forward.__get__(mod, mod.__class__); n += 1
    if verbose:
        print(f"[fused-rmsnorm] patched {n} RMSNorm modules -> single-kernel", flush=True)
    return n


def _killtest():
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaRMSNorm
    torch.manual_seed(0)
    print(f"{'shape':>16} {'with_scale':>10} {'fused-vs-stock rel':>18}")
    worst = 0.0
    for dim, shp, ws in [(2816, (1, 256, 2816), True), (2816, (1, 768, 2816), True),
                         (256, (1, 256, 16, 256), True), (256, (1, 256, 16, 256), False)]:
        rn = DiffusionGemmaRMSNorm(dim, with_scale=ws).cuda().bfloat16()
        if ws:
            rn.weight.data.normal_(0, 0.5)
        x = torch.randn(*shp, device="cuda", dtype=torch.bfloat16) * 0.3
        ref = rn.forward(x).float()
        fused = fused_rmsnorm(x, rn.weight if ws else None, rn.eps).float()
        rel = ((fused - ref).abs().mean() / (ref.abs().mean() + 1e-9)).item()
        worst = max(worst, rel)
        print(f"{str(shp):>16} {str(ws):>10} {rel:>18.2e} {'OK' if rel < 5e-3 else '**FAIL**'}")
    # speed: 4 norms/layer x 30 = 120 layer-norms, stock vs fused
    import time
    rn = DiffusionGemmaRMSNorm(2816).cuda().bfloat16()
    x = torch.randn(1, 256, 2816, device="cuda", dtype=torch.bfloat16)
    def t(fn, it=500):
        for _ in range(20): fn()
        torch.cuda.synchronize(); s = time.time()
        for _ in range(it): fn()
        torch.cuda.synchronize(); return (time.time() - s) / it * 1e6
    ms_stock = t(lambda: rn.forward(x)); ms_fused = t(lambda: fused_rmsnorm(x, rn.weight, rn.eps))
    print(f"\nper-norm: stock {ms_stock:.1f}us | fused {ms_fused:.1f}us ({ms_stock/ms_fused:.1f}x)  "
          f"-> ~120 layer-norms/fwd: {120*ms_stock/1000:.1f}ms -> {120*ms_fused/1000:.1f}ms")
    print(f"worst rel = {worst:.2e} -> {'CORRECT' if worst < 5e-3 else 'FAIL'}")


if __name__ == "__main__":
    _killtest()
