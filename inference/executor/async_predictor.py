"""One in-flight prediction, with all socket I/O off the control thread."""
from concurrent.futures import ThreadPoolExecutor
import socket
import time

from inference import protocol


class AsyncPredictor:
    def __init__(self, sock, jpeg_quality):
        self.sock = sock
        self.jpeg_quality = jpeg_quality
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-inference")
        self.future = None

    def submit(self, seq, image, state, capture_time):
        if self.future is not None:
            raise RuntimeError("A prediction is already pending")
        # The worker owns snapshots; neither camera buffers nor state can change
        # underneath JPEG encoding / serialization.
        self.future = self.pool.submit(self._request, seq, image.copy(), state.copy(), capture_time)

    def _request(self, seq, image, state, capture_time):
        start = time.perf_counter()
        protocol.send_obs(self.sock, seq, image, state, t_capture=capture_time,
                          jpeg_quality=self.jpeg_quality)
        reply = protocol.recv_action(self.sock)
        if reply is None:
            raise ConnectionError("Predictor disconnected")
        if reply[0] != seq:
            raise ValueError(f"Expected reply sequence {seq}, got {reply[0]}")
        return reply, (time.perf_counter() - start) * 1000

    def ready(self):
        return self.future is not None and self.future.done()

    def check_error(self):
        if self.ready():
            error = self.future.exception()
            if error is not None:
                raise error

    def take(self):
        result = self.future.result()
        self.future = None
        return result

    def close(self):
        # Interrupt blocked send/recv before joining. No worker may outlive run()
        # or race with hardware/log teardown. The caller owns closing the socket.
        if self.future is not None and not self.future.done():
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.pool.shutdown(wait=True, cancel_futures=True)
