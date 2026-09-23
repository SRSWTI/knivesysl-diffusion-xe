"""fast_sampler.py — BITWISE-safe dedup of HF generate's duplicated entropy chain.

profile_sampler_ops.log: per denoising step the [1,256,262144] fp32 logits get a FULL
Categorical-entropy chain (logsumexp 7.4ms + sub 2.4 + exp 2.4 + mul ~2.4 + sum ~1.2) TWICE —
once in EntropyBoundSampler.accept_canvas, once in StableAndConfidentStoppingCriteria.__call__ —
on the SAME tensor. Identical ops on identical inputs -> caching the first result is bitwise
identical, no gate risk. ~16ms/fwd GPU + 5 fewer 268MB-kernel launches (Command Buffer Full
relief: 41ms/fwd CPU stall in the profile).

Safety guard: the cache is keyed by (data_ptr, _version); _denoising_step REPLACES the logits
tensor when finished_denoising.any() (torch.where -> new allocation), which misses the cache and
falls back to the stock computation. Everything RNG-facing (softmax->multinomial) is untouched.

Usage: fast_sampler.install() / .uninstall(). Gate: texts+forwards MUST equal native exactly.
"""
import torch
from transformers.models.diffusion_gemma import generation_diffusion_gemma as G

_STATE = {}
_ORIG = {}
_PARITY = {"on": False}


def _entropy(logits):
    """Tier dispatch: bitwise tier = stock Categorical chain; parity tier = fused one-pass."""
    if _PARITY["on"]:
        from fused_sampler_ops import fused_entropy
        return fused_entropy(logits)
    return torch.distributions.Categorical(logits=logits).entropy()


def _accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
    token_entropy = _entropy(logits)                     # (batch, canvas)
    _STATE["key"] = (logits.data_ptr(), logits._version)
    _STATE["entropy"] = token_entropy
    sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
    cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)
    sorted_selection_mask = cumulative_entropy - sorted_token_entropy <= self.entropy_bound
    self.accepted_token_mask = torch.scatter(
        input=torch.zeros_like(sorted_selection_mask), dim=-1, index=sorted_indices, src=sorted_selection_mask
    )
    return torch.where(self.accepted_token_mask, denoiser_canvas, current_canvas)


def _stop_call(self, argmax_canvas, logits, **kwargs):
    # stability part: verbatim stock logic
    if self.stability_threshold == 0:
        stable = torch.ones((logits.shape[0]), device=logits.device, dtype=torch.bool)
    else:
        if self.argmax_canvas_history is None:
            self.argmax_canvas_history = torch.full(
                (self.stability_threshold, argmax_canvas.shape[0], argmax_canvas.shape[1]),
                -1, dtype=argmax_canvas.dtype, device=argmax_canvas.device)
        stable = (self.argmax_canvas_history == argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
        self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, shifts=-1, dims=0)
        self.argmax_canvas_history[-1] = argmax_canvas
    # confidence part: reuse the sampler's entropy when the tensor is byte-identical
    if _STATE.get("key") == (logits.data_ptr(), logits._version):
        token_entropy = _STATE["entropy"]
        _STATE["hits"] = _STATE.get("hits", 0) + 1
    else:
        token_entropy = _entropy(logits)
        _STATE["misses"] = _STATE.get("misses", 0) + 1
    confident = torch.mean(token_entropy, dim=-1) < self.confidence_threshold
    return stable & confident


def _denoising_step_parity(self, decoder_forward, current_canvas, argmax_canvas, input_ids,
                           decoder_position_ids, self_conditioning_logits, mask_mapping,
                           past_key_values, finished_denoising, cur_step, sampler,
                           logits_processor, diffusion_stopping_criteria, **model_kwargs):
    """Stock _denoising_step VERBATIM except the sampling block: softmax(5ms) + multinomial(10ms)
    + argmax(1.3ms) -> one fused Gumbel-max kernel (~1.2ms). Gumbel-max samples the identical
    Categorical distribution; RNG stream differs from torch.multinomial -> PARITY tier only.
    Seed: CPU torch.randint (consumes the per-question-seeded CPU generator -> runs reproduce,
    mirroring native's reseeded-multinomial structure) mixed with the step index."""
    from fused_sampler_ops import gumbel_argmax_sample
    step_int = int(cur_step)
    cur_step = torch.tensor(cur_step, device=current_canvas.device, dtype=torch.int32)
    torch.compiler.cudagraph_mark_step_begin()
    decoder_outputs = decoder_forward(
        decoder_input_ids=current_canvas,
        self_conditioning_logits=self_conditioning_logits,
        decoder_attention_mask=mask_mapping,
        past_key_values=past_key_values,
        decoder_position_ids=decoder_position_ids,
        **model_kwargs,
    )
    raw_logits = decoder_outputs.logits
    processed_logits = logits_processor(input_ids, raw_logits, cur_step=cur_step)
    batch_size, canvas_length = current_canvas.shape
    seed = (int(torch.randint(0, 2**31 - 1, (1,)).item()) ^ (step_int * 0x9E3779B1)) & 0x7FFFFFFF
    samp, amax = gumbel_argmax_sample(processed_logits, seed)
    denoiser_canvas = samp.view(batch_size, canvas_length)
    new_argmax_canvas = amax.view(batch_size, canvas_length)
    accepted_canvas = sampler.accept_canvas(current_canvas, denoiser_canvas, processed_logits, cur_step)
    accepted_canvas = accepted_canvas.clone()
    new_current_canvas = sampler.renoise_canvas(accepted_canvas, cur_step)
    new_current_canvas = new_current_canvas.clone()
    if diffusion_stopping_criteria is not None:
        if finished_denoising.any():
            new_argmax_canvas = torch.where(finished_denoising[:, None], argmax_canvas, new_argmax_canvas)
            new_current_canvas = torch.where(finished_denoising[:, None], current_canvas, new_current_canvas)
            processed_logits = torch.where(
                finished_denoising[:, None, None], self_conditioning_logits, processed_logits
            )
        finished_denoising |= diffusion_stopping_criteria(new_argmax_canvas, processed_logits)
    embeddings_dtype = self.model.decoder.embed_tokens.weight.dtype
    self_conditioning_logits = processed_logits.to(embeddings_dtype)
    return (new_current_canvas, new_argmax_canvas, self_conditioning_logits, finished_denoising)


def install(parity=False):
    if _ORIG:
        if bool(parity) != _PARITY["on"]:                # never silently downgrade the tier
            raise RuntimeError(
                f"fast_sampler already installed with parity={_PARITY['on']}; "
                f"call uninstall() before install(parity={bool(parity)})"
            )
        return
    _PARITY["on"] = bool(parity)
    _ORIG["accept"] = G.EntropyBoundSampler.accept_canvas
    _ORIG["stop"] = G.StableAndConfidentStoppingCriteria.__call__
    G.EntropyBoundSampler.accept_canvas = _accept_canvas
    G.StableAndConfidentStoppingCriteria.__call__ = _stop_call
    if parity:
        _ORIG["step"] = G.DiffusionGemmaGenerationMixin._denoising_step
        G.DiffusionGemmaGenerationMixin._denoising_step = _denoising_step_parity
    _STATE.clear()


def uninstall():
    if not _ORIG:
        return
    G.EntropyBoundSampler.accept_canvas = _ORIG.pop("accept")
    G.StableAndConfidentStoppingCriteria.__call__ = _ORIG.pop("stop")
    if "step" in _ORIG:
        G.DiffusionGemmaGenerationMixin._denoising_step = _ORIG.pop("step")
    _PARITY["on"] = False


def stats():
    return {k: _STATE.get(k, 0) for k in ("hits", "misses")}
