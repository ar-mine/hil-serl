import time
from multiprocessing import resource_tracker
from multiprocessing import shared_memory
from pathlib import Path
import struct

import numpy as np


MAGIC = b"HILSHM1\0"
VERSION = 1
HEADER_FORMAT = "<8sIIIIIQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
DTYPE_CODES = {
    np.dtype("uint8"): 1,
}
CODE_DTYPES = {v: k for k, v in DTYPE_CODES.items()}


def _unregister_from_resource_tracker(shm):
    # This buffer is intentionally shared across unrelated processes. Python's
    # resource_tracker assumes ownership and may unlink it when a reader exits.
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


def default_shm_name(name: str) -> str:
    return f"hilserl_{name}"


class SharedMemoryFrameWriter:
    """Single-writer shared-memory frame buffer for camera images."""

    def __init__(self, name, shape=(128, 128, 3), dtype=np.uint8, shm_name=None):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        if self.dtype not in DTYPE_CODES:
            raise ValueError(f"Unsupported dtype for shared memory frames: {self.dtype}")

        self.shm_name = shm_name or default_shm_name(name)
        self.frame_nbytes = int(np.prod(self.shape) * self.dtype.itemsize)
        size = HEADER_SIZE + self.frame_nbytes

        try:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, create=True, size=size
            )
            _unregister_from_resource_tracker(self.shm)
            self._owns_shm = True
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            _unregister_from_resource_tracker(self.shm)
            if self.shm.size < size:
                self.shm.close()
                raise ValueError(
                    f"Existing shared memory {self.shm_name} is too small: "
                    f"{self.shm.size} < {size}"
                )
            self._owns_shm = False

        self._frame = np.ndarray(
            self.shape, dtype=self.dtype, buffer=self.shm.buf, offset=HEADER_SIZE
        )
        self._sequence = 0
        self._write_header(sequence=0, timestamp_ns=0)
        self._write_manifest()

    def _write_manifest(self):
        manifest = Path("/tmp") / f"{self.shm_name}.json"
        manifest.write_text(
            (
                "{\n"
                f'  "name": "{self.name}",\n'
                f'  "shm_name": "{self.shm_name}",\n'
                f'  "shape": {list(self.shape)},\n'
                f'  "dtype": "{self.dtype.name}"\n'
                "}\n"
            ),
            encoding="utf-8",
        )

    def _write_header(self, sequence, timestamp_ns):
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            self.shape[0],
            self.shape[1],
            self.shape[2],
            DTYPE_CODES[self.dtype],
            int(sequence),
            int(timestamp_ns),
            self.frame_nbytes,
        )
        self.shm.buf[:HEADER_SIZE] = header

    def write(self, frame, timestamp_ns=None):
        frame = np.asarray(frame)
        if frame.shape != self.shape:
            raise ValueError(f"Expected frame shape {self.shape}, got {frame.shape}")
        if frame.dtype != self.dtype:
            frame = frame.astype(self.dtype, copy=False)

        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        self._sequence += 1
        self._write_header(sequence=self._sequence * 2 - 1, timestamp_ns=timestamp_ns)
        np.copyto(self._frame, frame)
        self._write_header(sequence=self._sequence * 2, timestamp_ns=timestamp_ns)

    def close(self, unlink=False):
        self.shm.close()
        if unlink and self._owns_shm:
            self.shm.unlink()


class SharedMemoryCapture:
    """Camera capture compatible with VideoCapture, backed by shared memory."""

    def __init__(
        self,
        name,
        shape=(128, 128, 3),
        dtype=np.uint8,
        shm_name=None,
        timeout=5.0,
        stale_after=1.0,
    ):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.shm_name = shm_name or default_shm_name(name)
        self.timeout = float(timeout)
        self.stale_after = float(stale_after)
        self.shm = None
        self._last_sequence = 0

    def _connect(self):
        if self.shm is None:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            _unregister_from_resource_tracker(self.shm)

    def _read_header(self):
        header = bytes(self.shm.buf[:HEADER_SIZE])
        magic, version, height, width, channels, dtype_code, sequence, timestamp_ns, nbytes = (
            struct.unpack(HEADER_FORMAT, header)
        )
        if magic != MAGIC or version != VERSION:
            raise RuntimeError(f"Invalid shared memory camera header for {self.name}")
        dtype = CODE_DTYPES.get(dtype_code)
        if dtype != self.dtype:
            raise RuntimeError(
                f"Shared memory dtype mismatch for {self.name}: {dtype} != {self.dtype}"
            )
        if (height, width, channels) != self.shape:
            raise RuntimeError(
                f"Shared memory shape mismatch for {self.name}: "
                f"{(height, width, channels)} != {self.shape}"
            )
        expected_nbytes = int(np.prod(self.shape) * self.dtype.itemsize)
        if nbytes != expected_nbytes:
            raise RuntimeError(
                f"Shared memory byte size mismatch for {self.name}: "
                f"{nbytes} != {expected_nbytes}"
            )
        return sequence, timestamp_ns

    def read(self):
        start = time.time()
        while True:
            try:
                self._connect()
                seq_before, timestamp_ns = self._read_header()
                if seq_before and seq_before % 2 == 0:
                    frame_view = np.ndarray(
                        self.shape,
                        dtype=self.dtype,
                        buffer=self.shm.buf,
                        offset=HEADER_SIZE,
                    )
                    frame = frame_view.copy()
                    seq_after, timestamp_after = self._read_header()
                    if seq_before == seq_after and timestamp_ns == timestamp_after:
                        age = time.time() - (timestamp_ns / 1e9)
                        if age > self.stale_after:
                            raise TimeoutError(
                                f"Shared memory frame for {self.name} is stale: {age:.3f}s"
                            )
                        self._last_sequence = seq_after
                        return True, frame
            except FileNotFoundError:
                pass

            if time.time() - start > self.timeout:
                raise TimeoutError(
                    f"Timed out waiting for shared memory camera {self.name} "
                    f"({self.shm_name})"
                )
            time.sleep(0.01)

    def close(self):
        if self.shm is not None:
            self.shm.close()
            self.shm = None
