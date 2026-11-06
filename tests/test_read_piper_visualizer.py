import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from scripts.piper_api import read_piper_visualizer as viewer


class ReadPiperVisualizerTests(unittest.TestCase):
    def test_stale_missing_invalid_and_future_feedback_are_hidden(self):
        self.assertIsNone(viewer.fresh_values(None, 'pose', 6, 100., .5))
        for stamp, values in ((99., [0.] * 6), (101., [0.] * 6),
                              (100., [np.nan] * 6), (100., [0.] * 3)):
            state = SimpleNamespace(pose=values, timestamp=stamp)
            self.assertIsNone(viewer.fresh_values(state, 'pose', 6, 100., .5))
        state = SimpleNamespace(pose=[.1] * 6, timestamp=99.9)
        np.testing.assert_allclose(viewer.fresh_values(state, 'pose', 6, 100., .5), [.1] * 6)

    def test_live_loop_only_connects_reads_and_disconnects(self):
        calls = []
        class ReadOnlyAPI:
            # Any motion, enable, disable or configuration call is absent and
            # would fail. This also checks the actual main-loop cleanup path.
            def __init__(self, **kwargs):
                calls.append(('init', kwargs))
            def connect(self):
                calls.append('connect')
                return True
            def get_eef_state(self):
                return SimpleNamespace(pose=[.22, -.25, .13, 2.8, 0., 1.6], timestamp=time.time())
            def get_joint_state(self):
                return SimpleNamespace(joints=[0.] * 6, timestamp=time.time())
            def get_state(self):
                return {}
            def disconnect(self):
                calls.append('disconnect')
        frames = []
        with patch.object(viewer, 'PiperAPI', ReadOnlyAPI), \
             patch.multiple(cv2, namedWindow=lambda *a: None, resizeWindow=lambda *a: None,
                            setWindowProperty=lambda *a: None,
                            imshow=lambda name, image: frames.append(image),
                            waitKey=lambda *a: ord('q'), destroyWindow=lambda *a: None):
            self.assertEqual(viewer.run(viewer.parse_args(['--fullscreen'])), 0)
        self.assertIsNone(calls[0][1]['installation_pos'])
        self.assertIsNone(calls[0][1]['payload'])
        self.assertEqual(calls[1:], ['connect', 'disconnect'])
        self.assertEqual(frames[0].shape, (900, 1280, 3))


if __name__ == '__main__':
    unittest.main()
