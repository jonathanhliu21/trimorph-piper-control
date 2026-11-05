"""Real-time chunking (RTC) for the diffusion policy: each chunk is inpainted onto the previous one.

Black, Galliker, Levine 2025, "Real-Time Execution of Action Chunking Flow
Policies". The executor keeps executing the current chunk while the next one is
sampled, and sends the rest of the current chunk from the observation step on.
The policy's actions are window-relative to the observed state (index 0 is the
observation), so new index i continues old step k0 + i, where k0 is the step
whose raw action the executor sent as the observed arm pose. The new chunk is
sampled with pseudoinverse guidance (PiGDM) toward that remainder: weight 1 on
the observation and the `delay` steps the executor runs before the new chunk
arrives, exponentially decaying weights over the rest of the overlap, and 0
after it. The new chunk therefore takes over where the old one is, moving the
same way, and follows its own observation beyond the overlap.

DDIM/DDPM form of the paper's flow-matching guidance: at every denoising step
the denoised chunk x0 is moved by gain * J^T W (target - x0), J = dx0/dx_t. The
PiGDM gain is 1/sqrt(alpha_bar_t), clipped to max_guidance * sigma_t
(sigma_t = sqrt(1 - alpha_bar_t)): the paper's guidance-weight clip beta,
mapped from flow matching to this noise schedule.
"""
from __future__ import annotations

import functools

import numpy as np


def guidance_weights(horizon, overlap, frozen):
    """RTC soft mask over the new chunk (paper Eq. 5).

    1 on the first `frozen` steps, c (e^c - 1) / (e - 1) with c falling linearly
    toward 0 over the rest of the `overlap` steps that have a previous action,
    and 0 after them.
    """
    if not 0 <= frozen <= overlap <= horizon:
        raise ValueError(f"Require 0 <= frozen ({frozen}) <= overlap ({overlap}) <= horizon ({horizon})")
    weights = np.zeros(horizon)
    weights[:frozen] = 1.0
    c = (overlap - np.arange(frozen, overlap)) / (overlap - frozen + 1)
    weights[frozen:overlap] = c * np.expm1(c) / np.expm1(1.0)
    return weights


def guided_sample(policy, target, weights, max_guidance, cond_data_BTF, cond_mask_BTF, cond_BTL=None,
                  cond_BG=None, generator=None, **kwargs):
    """policy.conditional_sample, with every denoised chunk guided toward `target` (normalized [1, T, F])."""
    import torch

    scheduler = policy.noise_scheduler
    prediction = scheduler.config.prediction_type
    if prediction not in ("epsilon", "sample"):
        raise ValueError(f"RTC guidance supports epsilon or sample prediction, not {prediction}")
    trajectory = torch.randn(size=cond_data_BTF.shape, dtype=cond_data_BTF.dtype, device=cond_data_BTF.device,
                             generator=generator)
    scheduler.set_timesteps(policy.num_inference_steps)
    for t in scheduler.timesteps:
        trajectory[cond_mask_BTF] = cond_data_BTF[cond_mask_BTF]
        alpha_bar = scheduler.alphas_cumprod[t].to(trajectory)
        signal, sigma = alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
        with torch.enable_grad():
            noisy = trajectory.detach().requires_grad_(True)
            output = policy.model(sample_BTF=noisy, timesteps_B=t, cond_BTL=cond_BTL, cond_BG=cond_BG)
            denoised = output if prediction == "sample" else (noisy - sigma * output) / signal
            # PiGDM: the weighted error, pulled back through the denoiser's Jacobian.
            (correction,) = torch.autograd.grad(denoised, noisy, grad_outputs=weights * (target - denoised))
        gain = torch.minimum(max_guidance * sigma, 1 / signal)
        denoised = denoised.detach() + gain * correction
        output = denoised if prediction == "sample" else (trajectory - signal * denoised) / sigma
        trajectory = scheduler.step(output, t, trajectory, generator=generator, **kwargs).prev_sample
    trajectory[cond_mask_BTF] = cond_data_BTF[cond_mask_BTF]
    return trajectory


class RealTimeChunker:
    """Serves RTC requests with a MidasPolicyInference wrapper: full-horizon absolute chunks."""

    def __init__(self, wrapper, to_relative, to_absolute):
        """to_relative(absolute [L, D], state [D]) and to_absolute(relative [T, D], state) must be the
        policy's window-relative action convention (shared.env.midas.se3_utils)."""
        if not wrapper.policy.global_obs_cond:
            raise NotImplementedError("RTC guidance needs a policy with global observation conditioning")
        self.wrapper = wrapper
        self.policy = wrapper.policy
        self.to_relative, self.to_absolute = to_relative, to_absolute
        self.policy.requires_grad_(False)  # guidance differentiates only with respect to the noisy chunk

    def predict(self, image, state, prev_chunk, delay, max_guidance):
        """Absolute [horizon, D] chunk; index 0 is `state`, index i continues prev_chunk[i].

        prev_chunk: the previous chunk's absolute actions from the observation step on ([L, D], L may be 0).
        delay: previous actions the executor runs before this chunk takes over.
        """
        import torch

        policy = self.policy
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        prev_chunk = np.asarray(prev_chunk, dtype=np.float32).reshape(-1, state.size)
        horizon = int(policy.horizon)
        overlap = min(len(prev_chunk), horizon)
        obs = self.wrapper._prepare_obs_torch(image=image, robot_state=state)
        with torch.no_grad():
            if overlap and max_guidance > 0:
                target = np.zeros((horizon, state.size), dtype=np.float32)
                target[:overlap] = self.to_relative(prev_chunk[:overlap], state)
                target = policy.normalizer["action"].normalize(
                    torch.from_numpy(target).to(policy.device, policy.dtype))
                weights = torch.as_tensor(guidance_weights(horizon, overlap, min(int(delay) + 1, overlap)),
                                          device=policy.device, dtype=policy.dtype)
                # predict_action keeps its observation encoding and unnormalization; only sampling changes.
                policy.conditional_sample = functools.partial(
                    guided_sample, policy, target[None], weights[None, :, None], float(max_guidance))
                try:
                    result = policy.predict_action(obs)
                finally:
                    del policy.conditional_sample
            else:
                result = policy.predict_action(obs)
        relative = result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
        return self.to_absolute(relative, state)
