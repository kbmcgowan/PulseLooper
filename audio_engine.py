from __future__ import annotations

import queue
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import sounddevice as sd

from audio_files import WavOutputWriter


def delete_sample_range(
    sample: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    if not 0 <= start < end <= len(sample):
        raise ValueError("The selection must be a non-empty range within the sample.")
    return np.concatenate((sample[:start], sample[end:]))


def mix_overlapping_grains(
    sample: np.ndarray,
    start_positions: Sequence[int],
    block_start: int,
    frame_count: int,
) -> np.ndarray:
    """Sum every grain that intersects the requested output block."""
    output = np.zeros(frame_count, dtype=np.float32)
    sample_length = len(sample)
    block_end = block_start + frame_count

    for grain_start in start_positions:
        grain_end = grain_start + sample_length
        overlap_start = max(block_start, grain_start)
        overlap_end = min(block_end, grain_end)
        if overlap_start >= overlap_end:
            continue

        output_start = overlap_start - block_start
        sample_start = overlap_start - grain_start
        overlap_length = overlap_end - overlap_start
        output[output_start : output_start + overlap_length] += sample[
            sample_start : sample_start + overlap_length
        ]

    return output


def schedule_grain_starts(
    frame_count: int,
    sample_rate: int,
    repeat_rate: float,
    next_start: float,
) -> tuple[list[int], float]:
    interval = sample_rate / repeat_rate
    starts: list[int] = []
    while next_start < frame_count:
        starts.append(int(next_start))
        next_start += interval
    return starts, next_start - frame_count


class OverlapPlayer:
    def __init__(self, sample_rate: int = 44_100, block_size: int = 1024) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._sample = np.empty(0, dtype=np.float32)
        self._playing = False
        self._loop = False
        self._repeat_rate = 1.0
        self._next_start = 0.0
        self._frame_position = 0
        self._active_starts: list[int] = []
        self._stream: sd.OutputStream | None = None
        self._output_writer: WavOutputWriter | None = None
        self._audio_messages: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._repeat_rate_updates: queue.SimpleQueue[float] = queue.SimpleQueue()

    @property
    def is_playing(self) -> bool:
        return self._playing

    def start(
        self,
        sample: np.ndarray,
        repeat_rate: float,
        loop: bool,
        output_path: str | Path | None = None,
    ) -> None:
        if sample.size == 0:
            raise ValueError("There is no recorded sample to play.")
        self._validate_repeat_rate(repeat_rate)

        self.stop()
        self._discard_repeat_rate_updates()
        self._sample = np.ascontiguousarray(sample, dtype=np.float32)
        self._loop = loop
        self._repeat_rate = repeat_rate
        self._next_start = 0.0
        self._frame_position = 0
        self._active_starts = []
        self._output_writer = (
            WavOutputWriter(output_path, self.sample_rate)
            if output_path is not None
            else None
        )
        self._playing = True

        stream: sd.OutputStream | None = None
        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.block_size,
                callback=self._audio_callback,
            )
            self._stream = stream
            stream.start()
        except Exception as error:
            self._playing = False
            self._stream = None
            cleanup_errors: list[Exception] = []
            if stream is not None:
                try:
                    stream.close()
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            writer = self._output_writer
            self._output_writer = None
            if writer is not None:
                try:
                    writer.abort()
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                details = "; ".join(str(cleanup_error) for cleanup_error in cleanup_errors)
                raise RuntimeError(f"{error} (cleanup failed: {details})") from error
            raise

    @property
    def is_looping(self) -> bool:
        return self._playing and self._loop

    def set_repeat_rate(self, repeat_rate: float) -> None:
        self._validate_repeat_rate(repeat_rate)
        self._repeat_rate_updates.put(repeat_rate)

    @staticmethod
    def _validate_repeat_rate(repeat_rate: float) -> None:
        if not 1 <= repeat_rate <= 10_000:
            raise ValueError("The repeat rate must be between 1 and 10,000 Hz.")

    def _discard_repeat_rate_updates(self) -> None:
        while True:
            try:
                self._repeat_rate_updates.get_nowait()
            except queue.Empty:
                return

    def _apply_repeat_rate_updates(self) -> None:
        updated_rate: float | None = None
        while True:
            try:
                updated_rate = self._repeat_rate_updates.get_nowait()
            except queue.Empty:
                break

        if updated_rate is not None and updated_rate != self._repeat_rate:
            self._repeat_rate = updated_rate
            self._next_start = 0.0

    def stop(self) -> None:
        self._playing = False
        stream = self._stream
        self._stream = None
        stop_errors: list[Exception] = []
        if stream is not None:
            try:
                stream.stop()
            except Exception as error:
                stop_errors.append(error)
            try:
                stream.close()
            except Exception as error:
                stop_errors.append(error)

        writer = self._output_writer
        self._output_writer = None
        if writer is not None:
            try:
                writer.finish()
            except Exception as error:
                stop_errors.append(error)

        if len(stop_errors) == 1:
            raise stop_errors[0]
        if stop_errors:
            details = "; ".join(str(error) for error in stop_errors)
            raise RuntimeError(f"Could not stop playback cleanly: {details}") from stop_errors[0]

    def pop_audio_message(self) -> str | None:
        try:
            return self._audio_messages.get_nowait()
        except queue.Empty:
            return None

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del time_info
        if status:
            self._audio_messages.put(f"Audio output status: {status}")

        outdata.fill(0)
        if not self._playing:
            return

        sample = self._sample
        block_start = self._frame_position
        block_end = block_start + frames

        if self._loop:
            self._apply_repeat_rate_updates()
            starts, self._next_start = schedule_grain_starts(
                frames,
                self.sample_rate,
                self._repeat_rate,
                self._next_start,
            )
            self._active_starts.extend(block_start + start for start in starts)
        elif not self._active_starts:
            self._active_starts.append(block_start)
            self._next_start = float("inf")

        outdata[:, 0] = mix_overlapping_grains(
            sample, self._active_starts, block_start, frames
        )
        if self._output_writer is not None:
            self._output_writer.append(outdata)

        self._active_starts = [
            start
            for start in self._active_starts
            if start + len(sample) > block_end
        ]
        self._frame_position = block_end

        if not self._loop and not self._active_starts:
            self._playing = False
