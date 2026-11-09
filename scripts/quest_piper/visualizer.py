"""Reuse the Quest/Piper accuracy viewer for passive collection and replay."""
from concurrent.futures import ThreadPoolExecutor
import json
import time

import numpy as np

from data_collection.collection.tracking_monitor import project
from scripts.quest.visualize_quest_accuracy_piper import (
    Comparison, PiperView, LatestIK, fit_capture, parse_args,
)

ORIGINAL_COLOR = (220, 80, 220)


class PipelineView(PiperView):
    """The same error/IK view, with an optional original-arm trajectory overlay."""
    def draw(self, history, now, ik):
        image = super().draw(history, now, ik)
        if getattr(self, 'replay', False):
            self.cv2.rectangle(image, (0, 842), (self.width, 874), (28, 28, 28), -1)
            self.cv2.putText(image, 'REPLAY | drag: rotate | scroll/right-drag: zoom | Q/Esc: stop replay',
                            (16, 860), self.cv2.FONT_HERSHEY_SIMPLEX, .53, (195, 195, 195), 1, self.cv2.LINE_AA)
        banner = getattr(self, 'banner', None)
        if banner:
            self.cv2.putText(image, banner, (16, 440), self.cv2.FONT_HERSHEY_SIMPLEX,
                            .6, (80, 220, 80), 1, self.cv2.LINE_AA)
        original = getattr(self, 'original', None)
        if original is not None and self.comparison.anchor is not None:
            points = project(np.asarray(original)-self.comparison.anchor[2],
                             self.yaw, self.pitch, self.scale, (640, 570))
            # Limit only display work; comparisons retain every source sample.
            step = max(1, len(points)//1000)
            self.cv2.polylines(image, [points[::step]], False, ORIGINAL_COLOR, 1)
            self.cv2.putText(image, 'Magenta: original recorded Piper FK', (16, 415),
                            self.cv2.FONT_HERSHEY_SIMPLEX, .53, ORIGINAL_COLOR, 1, self.cv2.LINE_AA)
        return image


class CollectionView:
    def __init__(self, shared, calibration_path):
        args = parse_args([])
        calibration = None if calibration_path is None else json.loads(calibration_path.read_text())
        self.shared = shared
        self.comparison = Comparison(args, calibration)
        self.view = PipelineView(shared, args, self.comparison)
        self.ik = LatestIK(shared)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.toggle_recording = False

    def update(self):
        self.toggle_recording = False
        now = time.monotonic_ns()
        history = self.shared.snapshot()
        c, view = self.comparison, self.view
        targets = c.ingest(history, now)
        joints, _ = self.shared.feedback()
        if targets and joints is not None:
            self.ik.submit(targets[-1], joints, c.generation)
        if c.capture_end is not None and now >= c.capture_end:
            c.capture_end = None
            self.pending = self.executor.submit(fit_capture, c.capture, c.latency)
            c.message = 'Fitting viewer calibration (raw recordings unchanged)...'
        if self.pending is not None and self.pending.done():
            try:
                c.calibration = self.pending.result()
                c.reanchor()
                c.message = 'Viewer fit installed; raw recordings unchanged. Validate on new poses.'
            except (ValueError, np.linalg.LinAlgError) as exc:
                c.message = f'Viewer calibration failed: {exc}'
            self.pending = None
        view.cv2.imshow(view.window_name, view.draw(history, now, self.ik.snapshot()))
        key = view.cv2.waitKeyEx(1)
        if key in (27, ord('q'), ord('Q')) or view.cv2.getWindowProperty(view.window_name, view.cv2.WND_PROP_VISIBLE) < 1:
            return False
        if key in (ord('c'), ord('C')) and self.pending is None:
            c.start_capture(now)
        elif key in (ord('a'), ord('A')):
            c.reanchor()
        elif key in (ord('r'), ord('R')):
            c.clear()
        elif key == 32:
            self.toggle_recording = True
        return True

    def close(self):
        self.ik.thread.join(timeout=2.)
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.view.close()
