import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.calib.fit_quest_calibration import fit_vicon_to_flange, output_name, vicon_flange


def synthetic_poses(count, rotate=True, seed=0):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-25, 25, (count, 3)) if rotate else np.zeros((count, 3))
    flange_r = Rotation.from_euler("xyz", angles, degrees=True) * Rotation.from_euler("x", 180, degrees=True)
    flange_p = np.array([0.4, 0.0, -0.13]) + rng.uniform(-0.03, 0.03, (count, 3))
    return flange_p, flange_r


class ViconReferenceTests(unittest.TestCase):
    def test_recovers_world_and_body_transforms(self):
        flange_p, flange_r = synthetic_poses(400)
        world = Rotation.from_euler("zyx", [120, 3, -2], degrees=True)  # Vicon -> Piper
        t_world = np.array([0.5, -1.2, 0.8])
        body = Rotation.from_euler("zyx", [10, -70, 25], degrees=True)  # flange in Vicon body frame
        t_body = np.array([0.02, -0.06, 0.09])
        # T_vicon = W^-1 T_flange X^-1
        vicon_r = world.inv() * flange_r * body.inv()
        vicon_p = world.inv().apply(flange_p - t_world) - vicon_r.apply(t_body)
        transform = fit_vicon_to_flange(1.0 * vicon_p, vicon_r.as_matrix(), flange_p, flange_r.as_matrix())
        np.testing.assert_allclose(transform[0], world.as_matrix(), atol=1e-9)
        np.testing.assert_allclose(transform[1], t_world, atol=1e-9)
        np.testing.assert_allclose(transform[2], body.as_matrix(), atol=1e-9)
        np.testing.assert_allclose(transform[3], t_body, atol=1e-9)
        position, rotation = vicon_flange(transform, vicon_p, vicon_r.as_matrix())
        np.testing.assert_allclose(position, flange_p, atol=1e-9)
        np.testing.assert_allclose(rotation, flange_r.as_matrix(), atol=1e-9)

    def test_rejects_translation_only_recording(self):
        flange_p, flange_r = synthetic_poses(400, rotate=False)
        with self.assertRaisesRegex(ValueError, "wrist rotation"):
            fit_vicon_to_flange(flange_p, flange_r.as_matrix(), flange_p, flange_r.as_matrix())

    def test_output_directory_names(self):
        self.assertEqual(output_name("quest", "fk"), "calibration_quest")
        self.assertEqual(output_name("quest", "mocap"), "calibration_quest_vicon_ref")


if __name__ == "__main__":
    unittest.main()
