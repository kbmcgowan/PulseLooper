from __future__ import annotations

import os
import queue
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf


class WavOutputWriter:
    def __init__(self, path: str | Path, sample_rate: int) -> None:
        self.path = Path(path)
        self._blocks: queue.SimpleQueue[np.ndarray | None] = queue.SimpleQueue()
        self._error: Exception | None = None
        self._finished = False
        self._committed = False

        with tempfile.NamedTemporaryFile(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            self._temporary_path = Path(temporary_file.name)

        try:
            self._file = sf.SoundFile(
                self._temporary_path,
                mode="w",
                samplerate=sample_rate,
                channels=1,
                format="WAV",
                subtype="PCM_16",
            )
        except Exception:
            self._temporary_path.unlink(missing_ok=True)
            raise

        self._thread = threading.Thread(
            target=self._write_blocks,
            name="WavOutputWriter",
            daemon=True,
        )
        self._thread.start()

    def append(self, samples: np.ndarray) -> None:
        if self._finished:
            raise RuntimeError("The WAV output writer has already finished.")
        self._blocks.put(samples.copy())

    def finish(self) -> None:
        if self._committed:
            return
        if not self._finished:
            self._finished = True
            self._blocks.put(None)
            self._thread.join()

        if self._error is not None:
            self._remove_temporary_file()
            raise RuntimeError(f"Could not write WAV output: {self._error}") from self._error

        try:
            os.replace(self._temporary_path, self.path)
        except Exception:
            self._remove_temporary_file()
            raise
        self._committed = True

    def abort(self) -> None:
        if not self._finished:
            self._finished = True
            self._blocks.put(None)
            self._thread.join()
        self._remove_temporary_file()
        if self._error is not None:
            raise RuntimeError(f"Could not close WAV output: {self._error}") from self._error

    def _write_blocks(self) -> None:
        while True:
            samples = self._blocks.get()
            if samples is None:
                break
            if self._error is None:
                try:
                    self._file.write(samples)
                except Exception as error:
                    self._error = error

        try:
            self._file.close()
        except Exception as error:
            if self._error is None:
                self._error = error

    def _remove_temporary_file(self) -> None:
        self._temporary_path.unlink(missing_ok=True)


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("The WAV file does not contain any audio samples.")

    mono_samples = samples.mean(axis=1, dtype=np.float64).astype(np.float32)
    return mono_samples, int(sample_rate)
