"""Video source handling module."""

import queue
import threading

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# Distinct sentinel pushed onto the frame queue to signal end-of-stream (frames
# themselves are numpy arrays, so a plain None would be ambiguous).
_FRAME_SENTINEL = object()

# Internal queue sizes per drop policy.
#   block: a few frames of slack so decode can run ahead of inference (backpressure).
#   drop : keep only the latest frame for minimal latency on a live stream.
_BLOCK_QUEUE_SIZE = 8
_DROP_QUEUE_SIZE = 1


class VideoSource:
    """Handle video input from webcam or file.

    Supports an optional asynchronous (threaded) read mode: a background thread
    continuously decodes + resizes frames into a queue while the main loop runs
    inference, so decode no longer sits on the critical path.

    Drop policy (only relevant in threaded mode):
        - "block": never drop a frame; the reader blocks when the queue is full
          (backpressure). Preserves every frame and keeps results deterministic —
          the right choice for video files.
        - "drop": keep only the freshest frame, discarding the oldest when the
          queue is full. Minimises latency at the cost of skipped frames — the
          right choice for a live camera / drone stream that must stay current.
        - "auto": "block" for video files, "drop" for live sources.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        threaded: bool = False,
        drop_policy: str = "auto",
    ):
        """Initialize video source.

        Args:
            source: Camera index (0 for default) or path to video file.
            width: Frame width.
            height: Frame height.
            threaded: Run decoding in a background thread (async pipeline).
            drop_policy: "auto" | "block" | "drop" — see class docstring.
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.fps = 30
        self.frame_count = 0
        self.total_frames = 0
        self.is_video_file = False

        self.threaded = threaded
        self._drop = False
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None

        self._open()

        if self.threaded:
            self._drop = self._resolve_drop(drop_policy)
            maxsize = _DROP_QUEUE_SIZE if self._drop else _BLOCK_QUEUE_SIZE
            self._queue = queue.Queue(maxsize=maxsize)
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def _open(self) -> None:
        """Open video source."""
        if isinstance(self.source, str) and Path(self.source).exists():
            self.is_video_file = True
            self.cap = cv2.VideoCapture(str(self.source))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _resolve_drop(self, drop_policy: str) -> bool:
        """Resolve the effective drop behavior. True = drop-oldest."""
        if drop_policy == "block":
            return False
        if drop_policy == "drop":
            return True
        if drop_policy != "auto":
            print(
                f"Warning: unknown video.drop_policy '{drop_policy}'; "
                "falling back to 'auto'."
            )
        # auto: live sources drop to stay current, files keep every frame.
        return not self.is_video_file

    # ── threaded reader ──────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Background thread: decode + resize frames into the queue."""
        try:
            while not self._stop.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (self.width, self.height))

                if self._drop:
                    self._put_drop_oldest(frame)
                else:
                    if not self._put_blocking(frame):
                        return  # stop requested while blocked
        except BaseException as exc:  # surface decode errors to the consumer
            self._error = exc
        finally:
            # Always signal end so a waiting consumer is released.
            self._put_drop_oldest(_FRAME_SENTINEL)

    def _put_drop_oldest(self, item) -> None:
        """Put item, discarding the oldest queued frame if full (never blocks)."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    def _put_blocking(self, item) -> bool:
        """Put item with backpressure. Returns False if a stop was requested."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from source.

        Returns (True, frame) on success, (False, None) at end of stream. In
        threaded mode this pops from the frame queue (blocking until a frame is
        available); a decode error raised in the reader thread is re-raised here.
        """
        if not self.threaded:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                frame = cv2.resize(frame, (self.width, self.height))
            return ret, frame

        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if not self._thread.is_alive():
                    return False, None
                continue

            if item is _FRAME_SENTINEL:
                if self._error is not None:
                    raise self._error
                return False, None

            self.frame_count += 1
            return True, item

    def release(self) -> None:
        """Stop the reader thread (if any) and release the capture device."""
        self._stop.set()
        if self._thread is not None:
            # Drain one slot so a blocked put can unblock, then join.
            if self._queue is not None:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

    @property
    def drops_frames(self) -> bool:
        """True if the threaded reader uses the drop-oldest policy."""
        return self.threaded and self._drop

    def get_fps(self) -> float:
        """Get frames per second."""
        return self.fps

    def get_frame_count(self) -> int:
        """Get current frame number."""
        return self.frame_count

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
