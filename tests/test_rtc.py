"""Real-time chunking guidance (inference/predictor/rtc.py). The sampling tests need torch and diffusers
(the GPU predictor environment) and are skipped without them."""
import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

from inference.predictor.rtc import guidance_weights

HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "diffusers"))
T = 16


class GuidanceWeightTests(unittest.TestCase):
    def test_weights_follow_the_paper_soft_mask(self):
        weights = guidance_weights(T, overlap=8, frozen=3)
        np.testing.assert_array_equal(weights[:3], 1.0)
        np.testing.assert_array_equal(weights[8:], 0.0)
        c = (8 - 3) / (8 - 3 + 1)
        self.assertAlmostEqual(weights[3], c * (np.e ** c - 1) / (np.e - 1))
        self.assertTrue(np.all(np.diff(weights[2:9]) < 0))
        np.testing.assert_array_equal(guidance_weights(T, 0, 0), 0.0)
        np.testing.assert_array_equal(guidance_weights(T, 4, 4)[:4], 1.0)
        with self.assertRaises(ValueError):
            guidance_weights(T, overlap=4, frozen=5)


def gp_policy():
    """A 'policy' whose denoiser is exact for a smooth Gaussian-process prior over 16-step chunks,
    with the deployed checkpoint's DDIM settings."""
    import torch
    from diffusers import DDIMScheduler

    steps = np.arange(T)
    prior = 0.3 ** 2 * np.exp(-0.5 * (steps[:, None] - steps[None, :]) ** 2 / 4.0 ** 2) + 1e-6 * np.eye(T)
    covariance = torch.tensor(prior)
    scheduler = DDIMScheduler(num_train_timesteps=50, beta_start=1e-4, beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")

    class Policy(torch.nn.Module):
        horizon, num_inference_steps, global_obs_cond, kwargs = T, 16, True, {}
        noise_scheduler, device, dtype = scheduler, torch.device("cpu"), torch.float64
        normalizer = {"action": SimpleNamespace(normalize=lambda x: x)}

        def model(self, sample_BTF, timesteps_B, cond_BTL=None, cond_BG=None):
            alpha_bar = scheduler.alphas_cumprod[timesteps_B].to(sample_BTF)
            a, s = alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
            gain = a * covariance @ torch.linalg.inv(a ** 2 * covariance + s ** 2 * torch.eye(T, dtype=torch.float64))
            return (sample_BTF - a * torch.einsum("ij,bjf->bif", gain, sample_BTF)) / s

        def conditional_sample(self, cond_data_BTF, cond_mask_BTF, cond_BTL=None, cond_BG=None, **kwargs):
            trajectory = torch.randn(cond_data_BTF.shape, dtype=torch.float64)
            scheduler.set_timesteps(self.num_inference_steps)
            for t in scheduler.timesteps:
                trajectory = scheduler.step(self.model(trajectory, t), t, trajectory).prev_sample
            return trajectory

        def predict_action(self, obs):
            cond = torch.zeros(obs["batch"], T, obs["dim"], dtype=torch.float64)
            sample = self.conditional_sample(cond, torch.zeros_like(cond, dtype=torch.bool))
            return {"action": sample[:, :8], "action_pred": sample}

    return Policy(), np.linalg.cholesky(prior)


@unittest.skipUnless(HAS_TORCH, "needs torch and diffusers")
class GuidedSamplingTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(0)

    def test_guided_chunk_continues_the_previous_chunk_at_the_takeover(self):
        import torch
        from inference.predictor.rtc import guided_sample

        policy, cholesky = gp_policy()
        rng = np.random.default_rng(1)
        delay, gaps = 2, {}
        for beta in (0.0, 20.0):
            torch.manual_seed(0)
            gap = []
            for _ in range(5):
                prev = cholesky @ rng.normal(size=T)  # previous chunk from new index 0 on
                weights = torch.tensor(guidance_weights(T, 8, delay + 1))[None, :, None]
                cond = torch.zeros(100, T, 1, dtype=torch.float64)
                x = guided_sample(policy, torch.tensor(prev)[None, :, None], weights, beta, cond,
                                  torch.zeros_like(cond, dtype=torch.bool))[..., 0].numpy()
                gap.append(np.abs(x[:, delay + 1] - prev[delay + 1]).mean())  # first new action executed
            gaps[beta] = np.mean(gap)
        typical_step = 0.3 * np.sqrt(2 * (1 - np.exp(-1 / 32)))
        self.assertLess(gaps[20.0], 1.5 * typical_step)
        self.assertLess(gaps[20.0], 0.4 * gaps[0.0])

    def test_chunker_returns_full_horizon_and_restores_the_policy_sampler(self):
        from inference.predictor.rtc import RealTimeChunker

        policy, _ = gp_policy()
        wrapper = SimpleNamespace(policy=policy,
                                  _prepare_obs_torch=lambda image, robot_state: {"batch": 1, "dim": 19})
        chunker = RealTimeChunker(wrapper, to_relative=lambda actions, state: actions - state,
                                  to_absolute=lambda relative, state: relative + state)
        state = np.full(19, 0.5, dtype=np.float32)
        prev = state + 0.05 * np.arange(8, dtype=np.float32)[:, None]  # moving 0.05 per step
        chunk = chunker.predict(None, state, prev, delay=2, max_guidance=20.0)
        self.assertEqual(chunk.shape, (T, 19))
        self.assertNotIn("conditional_sample", vars(policy))
        # Continues the previous chunk where it takes over (index 3) ...
        self.assertLess(np.abs(chunk[1:4] - prev[1:4]).mean(), 0.05)
        # ... and is still a policy sample beyond the overlap, not a collapsed one.
        self.assertGreater(np.ptp(chunk[8:], axis=0).mean(), 0.05)
        unguided = chunker.predict(None, state, prev[:0], delay=0, max_guidance=20.0)
        self.assertEqual(unguided.shape, (T, 19))
        self.assertGreater(np.abs(unguided[1:4] - prev[1:4]).mean(), 0.05)


if __name__ == "__main__":
    unittest.main()
