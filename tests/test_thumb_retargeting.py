import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from collection.thumb_retargeting import (
    MODEL_FORMAT,
    ThumbRetargetModel,
    apply_thumb_retargeting,
    build_follower_target,
    load_thumb_retarget_model,
)
from fit_model_to_endpoints import fit_linear_model
from retarget_thumb import generate_midas_waypoints


class ThumbRetargetingTests(unittest.TestCase):
    def test_linear_fit_predicts_midas_from_trimorph(self):
        rng = np.random.default_rng(12)
        trimorph = rng.normal(size=(40, 2))
        expected_a = np.asarray(
            [
                [1.1, 0.4],
                [-0.2, 0.9],
            ]
        )
        expected_b = np.asarray([0.3, 0.4])
        midas = trimorph @ expected_a.T + expected_b

        matrix, bias, metrics = fit_linear_model(trimorph, midas)

        np.testing.assert_allclose(matrix, expected_a, atol=1e-12)
        np.testing.assert_allclose(bias, expected_b, atol=1e-12)
        self.assertLess(metrics["rmse_total_rad"], 1e-12)

    def test_follower_target_overwrites_only_thumb_after_scaling(self):
        model = ThumbRetargetModel(np.eye(2) * 2.0, np.ones(2))
        target = build_follower_target(
            [1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4],
            command_scale=0.5,
            joint_command_scales=[0.1, 0.2, 0.3, 0.4, 0.5],
            thumb_model=model,
        )

        np.testing.assert_allclose(target, [0.05, 0.2, 7, 9, 1.25])

    def test_follower_target_adds_midas_zero_offsets_after_scaling(self):
        target = build_follower_target(
            [0.0, 2.0, -3.0],
            [4, 5, 6],
            command_scale=0.5,
            joint_command_scales=[0.2, 0.5, 2.0],
            joint_command_offsets=[0.25, -0.1, 0.4],
            thumb_model=None,
        )

        np.testing.assert_allclose(target, [0.25, 0.4, -2.6])

    def test_thumb_model_owns_bias_instead_of_generic_offsets(self):
        model = ThumbRetargetModel(np.eye(2), np.asarray([0.3, 0.4]))
        target = build_follower_target(
            [0.0, 0.0, 1.0, 2.0],
            [0, 1, 2, 3],
            command_scale=1.0,
            joint_command_scales=[1.0] * 4,
            joint_command_offsets=[0.1, 0.2, 10.0, 20.0],
            thumb_model=model,
        )

        np.testing.assert_allclose(target, [0.1, 0.2, 1.3, 2.4])

    def test_partial_thumb_selection_is_rejected(self):
        model = ThumbRetargetModel(np.eye(2), np.zeros(2))
        with self.assertRaisesRegex(ValueError, "missing active motor IDs"):
            apply_thumb_retargeting([0, 0, 0], [0, 1, 2], model)

    def test_model_json_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "format": MODEL_FORMAT,
                        "trimorph_motor_ids": [2, 3],
                        "midas_motor_ids": [2, 3],
                        "A": np.eye(2).tolist(),
                        "b": [3, 4],
                    }
                ),
                encoding="utf-8",
            )

            model = load_thumb_retarget_model(path)

            np.testing.assert_allclose(model.predict([1, 1]), [4, 5])

    def test_old_four_joint_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old_model.json"
            path.write_text(
                json.dumps(
                    {
                        "format": "trimorph-thumb-linear-v1",
                        "trimorph_motor_ids": [0, 1, 2, 3],
                        "midas_motor_ids": [0, 1, 2, 3],
                        "A": np.eye(4).tolist(),
                        "b": [0, 0, 0, 0],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "re-run"):
                load_thumb_retarget_model(path)

    def test_waypoints_respect_joint_limit_margin_and_cover_strata(self):
        lower = np.asarray([-1.5, -1.4, -0.7, 0.0])
        upper = np.asarray([1.4, 1.61, 0.8, 2.08])
        points = generate_midas_waypoints(
            lower, upper, count=40, seed=7, margin=0.05
        )
        safe_lower = lower[2:] + 0.05 * (upper[2:] - lower[2:])
        safe_upper = upper[2:] - 0.05 * (upper[2:] - lower[2:])

        self.assertEqual(points.shape, (40, 4))
        np.testing.assert_array_equal(points[:, :2], np.zeros((40, 2)))
        self.assertTrue(np.all(points[:, 2:] > safe_lower))
        self.assertTrue(np.all(points[:, 2:] < safe_upper))
        for joint in range(2):
            normalized = (points[:, joint + 2] - safe_lower[joint]) / (
                safe_upper[joint] - safe_lower[joint]
            )
            strata = np.floor(normalized * 40).astype(int)
            np.testing.assert_array_equal(np.sort(strata), np.arange(40))


if __name__ == "__main__":
    unittest.main()
