from __future__ import annotations

import math
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QLineF, QSignalBlocker, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from audio_engine import OverlapPlayer, delete_sample_range
from audio_files import load_wav


SAMPLE_RATE = 44_100
MIN_REPEAT_RATE = 1
MAX_REPEAT_RATE = 10_000
RATE_SLIDER_STEPS = 10_000


def repeat_rate_for_slider(position: int) -> int:
    position = min(max(position, 0), RATE_SLIDER_STEPS)
    ratio = position / RATE_SLIDER_STEPS
    return round(
        MIN_REPEAT_RATE
        * (MAX_REPEAT_RATE / MIN_REPEAT_RATE) ** ratio
    )


def slider_position_for_repeat_rate(repeat_rate: float) -> int:
    if not MIN_REPEAT_RATE <= repeat_rate <= MAX_REPEAT_RATE:
        raise ValueError("The repeat rate must be between 1 and 10,000 Hz.")
    ratio = math.log(repeat_rate / MIN_REPEAT_RATE) / math.log(
        MAX_REPEAT_RATE / MIN_REPEAT_RATE
    )
    return round(ratio * RATE_SLIDER_STEPS)


def format_time_ms(milliseconds: float) -> str:
    value = f"{milliseconds:.3f}".rstrip("0").rstrip(".")
    return f"{value} ms"


class FrequencyScale(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(18)
        ticks = (
            ("1", 0),
            ("100", slider_position_for_repeat_rate(100)),
            ("1k", slider_position_for_repeat_rate(1_000)),
            ("5k", slider_position_for_repeat_rate(5_000)),
            ("10k", RATE_SLIDER_STEPS),
        )
        self._ticks: list[tuple[QLabel, int]] = []
        for text, position in ticks:
            label = QLabel(text, self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.adjustSize()
            self._ticks.append((label, position))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        for index, (label, position) in enumerate(self._ticks):
            x = round(position / RATE_SLIDER_STEPS * (self.width() - 1))
            if index == 0:
                x = 0
            elif index == len(self._ticks) - 1:
                x = self.width() - label.width()
            else:
                x -= label.width() // 2
            label.move(x, 0)


class WaveformWidget(QWidget):
    selectionChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._samples = np.empty(0, dtype=np.float32)
        self._sample_rate = SAMPLE_RATE
        self._selection: tuple[int, int] | None = None
        self._drag_start: int | None = None
        self._zero_crossings = np.empty(0, dtype=np.float64)
        self.setMinimumHeight(190)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_samples(self, samples: np.ndarray, sample_rate: int) -> None:
        self._samples = samples
        self._sample_rate = sample_rate
        self._zero_crossings = self._find_zero_crossings(samples)
        self._selection = None
        self._drag_start = None
        self.selectionChanged.emit()
        self.update()

    @property
    def selection(self) -> tuple[int, int] | None:
        return self._selection

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._samples.size > 0
        ):
            self._drag_start = self._sample_index_at(event.position().x())
            self._set_selection(self._drag_start, self._drag_start)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_start is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._set_selection(
                self._drag_start,
                self._sample_index_at(event.position().x()),
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._drag_start is not None
        ):
            self._set_selection(
                self._drag_start,
                self._sample_index_at(event.position().x()),
            )
            self._drag_start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _sample_index_at(self, x: float) -> int:
        if self.width() <= 1:
            return 0
        fraction = min(max(x / (self.width() - 1), 0.0), 1.0)
        return round(fraction * len(self._samples))

    @staticmethod
    def _find_zero_crossings(samples: np.ndarray) -> np.ndarray:
        zero_samples = np.flatnonzero(samples == 0).astype(np.float64)
        left = samples[:-1]
        right = samples[1:]
        sign_changes = ((left < 0) & (right > 0)) | (
            (left > 0) & (right < 0)
        )
        crossing_indices = np.flatnonzero(sign_changes)
        crossing_positions = crossing_indices + (
            np.abs(left[sign_changes])
            / (np.abs(left[sign_changes]) + np.abs(right[sign_changes]))
        )
        return np.sort(np.concatenate((zero_samples, crossing_positions)))

    def _snap_sample_index(self, sample_index: int) -> int:
        if self._zero_crossings.size == 0:
            return sample_index

        right_index = int(np.searchsorted(self._zero_crossings, sample_index))
        if right_index == 0:
            crossing = self._zero_crossings[0]
        elif right_index == self._zero_crossings.size:
            crossing = self._zero_crossings[-1]
        else:
            left_crossing = self._zero_crossings[right_index - 1]
            right_crossing = self._zero_crossings[right_index]
            crossing = (
                left_crossing
                if sample_index - left_crossing <= right_crossing - sample_index
                else right_crossing
            )
        return round(float(crossing))

    def _set_selection(self, first: int, second: int) -> None:
        first = self._snap_sample_index(first)
        second = self._snap_sample_index(second)
        start, end = sorted((first, second))
        self._selection = (start, end) if start < end else None
        self.selectionChanged.emit()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171a21"))
        plot_height = max(1, self.height() - 26)
        if self._selection is not None and self._samples.size > 0:
            start, end = self._selection
            x_start = start / len(self._samples) * self.width()
            x_end = end / len(self._samples) * self.width()
            painter.fillRect(
                QRectF(x_start, 0, x_end - x_start, plot_height),
                QColor(88, 196, 163, 55),
            )
        center = plot_height / 2
        painter.setPen(QPen(QColor("#3a404b"), 1))
        painter.drawLine(0, round(center), self.width(), round(center))

        if self._samples.size == 0:
            painter.setPen(QColor("#9aa4b2"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Record a sample or open a WAV file",
            )
            return

        samples = self._samples
        columns = min(self.width(), len(samples))
        boundaries = np.linspace(0, len(samples), columns + 1, dtype=np.int64)
        peak = max(float(np.max(np.abs(samples))), 1e-6)
        scale = (plot_height * 0.44) / peak
        painter.setPen(QPen(QColor("#58c4a3"), 1))
        lines: list[QLineF] = []

        for x in range(columns):
            start, end = int(boundaries[x]), int(boundaries[x + 1])
            if end <= start:
                continue
            segment = samples[start:end]
            top = center - float(np.max(segment)) * scale
            bottom = center - float(np.min(segment)) * scale
            lines.append(QLineF(float(x), top, float(x), bottom))

        painter.drawLines(lines)
        self._draw_time_axis(painter, plot_height)

    def _draw_time_axis(self, painter: QPainter, plot_height: int) -> None:
        axis_y = plot_height + 1
        painter.setPen(QPen(QColor("#3a404b"), 1))
        painter.drawLine(0, axis_y, self.width(), axis_y)

        font_metrics = painter.fontMetrics()
        label_y = axis_y + 3
        label_height = self.height() - label_y
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            x = round(fraction * (self.width() - 1))
            painter.drawLine(x, axis_y, x, axis_y + 5)
            milliseconds = (
                fraction * len(self._samples) * 1000 / self._sample_rate
            )
            label = format_time_ms(milliseconds)
            label_width = font_metrics.horizontalAdvance(label) + 8
            if fraction == 0:
                label_x = 0
                alignment = Qt.AlignmentFlag.AlignLeft
            elif fraction == 1:
                label_x = self.width() - label_width
                alignment = Qt.AlignmentFlag.AlignRight
            else:
                label_x = x - label_width // 2
                alignment = Qt.AlignmentFlag.AlignHCenter
            painter.drawText(
                QRectF(label_x, label_y, label_width, label_height),
                alignment | Qt.AlignmentFlag.AlignVCenter,
                label,
            )


class AudioSampleWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pulse Looper")
        self.resize(760, 420)

        self.samples = np.empty(0, dtype=np.float32)
        self.sample_rate = SAMPLE_RATE
        self.player = OverlapPlayer(sample_rate=SAMPLE_RATE)
        self._record_stream: sd.InputStream | None = None
        self._recorded_chunks: list[np.ndarray] = []
        self._record_started_at = 0.0
        self._active_output_path: Path | None = None
        self._audio_messages: queue.SimpleQueue[str] = queue.SimpleQueue()

        self.waveform = WaveformWidget()
        self.status_label = QLabel("Record or Import a pulse to get started.")

        self.record_button = QPushButton("Record Pulse")
        self.open_button = QPushButton("Import Pulse")
        self.once_button = QPushButton("Play once")
        self.loop_button = QPushButton("Play loop")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.delete_button = QPushButton("Delete selection")
        self.delete_button.setEnabled(False)

        self.rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rate_slider.setRange(0, RATE_SLIDER_STEPS)
        self.rate_slider.setPageStep(100)
        self.rate_slider.setValue(0)
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(1, 10_000)
        self.rate_spin.setDecimals(0)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.setValue(1)

        rate_controls = QVBoxLayout()
        rate_controls.setContentsMargins(0, 0, 0, 0)
        rate_controls.setSpacing(0)
        rate_controls.addWidget(self.rate_slider)
        rate_controls.addWidget(FrequencyScale())

        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Repeat rate (Hz)"))
        rate_row.addLayout(rate_controls, 1)
        rate_row.addWidget(self.rate_spin)

        buttons = QHBoxLayout()
        buttons.addWidget(self.record_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        buttons.addWidget(self.once_button)
        buttons.addWidget(self.loop_button)
        buttons.addWidget(self.delete_button)
        buttons.addWidget(self.stop_button)

        self.save_output_checkbox = QCheckBox("Save output to wav file")
        self.output_filename_edit = QLineEdit("output.wav")
        output_row = QHBoxLayout()
        output_row.addWidget(self.save_output_checkbox)
        output_row.addWidget(self.output_filename_edit, 1)

        layout = QVBoxLayout()
        layout.addWidget(
            QLabel("Pulse"),
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(self.waveform, 1)
        layout.addLayout(rate_row)
        layout.addLayout(buttons)
        layout.addLayout(output_row)
        layout.addWidget(self.status_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.record_button.clicked.connect(self.toggle_recording)
        self.open_button.clicked.connect(self.open_wav)
        self.once_button.clicked.connect(lambda: self.start_playback(loop=False))
        self.loop_button.clicked.connect(lambda: self.start_playback(loop=True))
        self.stop_button.clicked.connect(self.stop_playback)
        self.delete_button.clicked.connect(self.delete_selection)
        self.waveform.selectionChanged.connect(self.on_selection_changed)
        self.rate_slider.valueChanged.connect(self.on_rate_slider_changed)
        self.rate_spin.valueChanged.connect(self.on_rate_spin_changed)
        self.save_output_checkbox.toggled.connect(self.update_controls)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_status)
        self._timer.start(100)
        self.update_controls()

    def toggle_recording(self) -> None:
        if self._record_stream is not None:
            self.stop_recording()
            return

        self._recorded_chunks = []
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._record_callback,
        )
        try:
            stream.start()
        except Exception as error:
            try:
                stream.close()
            except Exception as cleanup_error:
                self.show_error("Could not close the recording stream", cleanup_error)
            self.show_error("Could not start recording", error)
            return

        self._record_stream = stream
        self._record_started_at = time.monotonic()
        self.record_button.setText("Stop recording")
        self.update_controls()

    def _record_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            self._audio_messages.put(f"Audio input status: {status}")
        self._recorded_chunks.append(indata[:, 0].copy())

    def stop_recording(self) -> None:
        stream = self._record_stream
        if stream is None:
            return

        self._record_stream = None
        stream_error: Exception | None = None
        try:
            stream.stop()
        except Exception as error:
            stream_error = error
        try:
            stream.close()
        except Exception as error:
            if stream_error is None:
                stream_error = error
        if stream_error is not None:
            self.show_error("Could not stop recording cleanly", stream_error)

        self.record_button.setText("Record")
        if self._recorded_chunks:
            self.samples = np.concatenate(self._recorded_chunks)
            self.sample_rate = SAMPLE_RATE
            self.waveform.set_samples(self.samples, self.sample_rate)
            duration = len(self.samples) / self.sample_rate
            self.status_label.setText(
                f"Recorded {duration:.2f} seconds at {self.sample_rate:,} Hz."
            )
        else:
            self.status_label.setText("No audio was captured.")
        self._recorded_chunks = []
        self.update_controls()

    def open_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open WAV sample",
            "",
            "WAV audio (*.wav *.wave)",
        )
        if not path:
            return

        try:
            samples, sample_rate = load_wav(path)
        except Exception as error:
            self.show_error("Could not open WAV file", error)
            return

        self.samples = samples
        self.sample_rate = sample_rate
        self.waveform.set_samples(self.samples, self.sample_rate)
        self.status_label.setText(
            f"Loaded {Path(path).name} — "
            f"{len(samples) / sample_rate:.2f} seconds at {sample_rate:,} Hz."
        )
        self.update_controls()

    def start_playback(self, loop: bool) -> None:
        output_path = None
        if self.save_output_checkbox.isChecked():
            output_path = self.resolve_output_path()
            if output_path is None:
                return

        try:
            self.player.sample_rate = self.sample_rate
            self.player.start(
                self.samples,
                repeat_rate=float(self.rate_spin.value()),
                loop=loop,
                output_path=str(output_path) if output_path is not None else None,
            )
        except Exception as error:
            self.show_error("Could not start playback", error)
            return

        self._active_output_path = output_path
        mode = "looping" if loop else "playing once"
        self.status_label.setText(
            f"{mode.capitalize()} at {self.rate_spin.value():g} full-sample starts per second."
        )
        if output_path is not None:
            self.status_label.setText(
                f"{self.status_label.text()} Saving to {output_path}."
            )
        self.update_controls()

    def resolve_output_path(self) -> Path | None:
        filename = self.output_filename_edit.text().strip()
        if not filename:
            self.show_error(
                "Output filename is required",
                ValueError("Enter a filename for the WAV output."),
            )
            return None

        path = Path(filename).expanduser()
        if not path.suffix:
            path = path.with_suffix(".wav")
            self.output_filename_edit.setText(str(path))
        elif path.suffix.lower() not in {".wav", ".wave"}:
            self.show_error(
                "Invalid output filename",
                ValueError("The output filename must have a .wav or .wave extension."),
            )
            return None

        if not path.exists():
            return path
        if not path.is_file():
            self.show_error(
                "Invalid output path",
                ValueError(f"{path} exists and is not a file."),
            )
            return None

        prompt = QMessageBox(self)
        prompt.setIcon(QMessageBox.Icon.Warning)
        prompt.setWindowTitle("Output file already exists")
        prompt.setText(f"{path} already exists.")
        prompt.setInformativeText(
            "Change the filename or overwrite the existing WAV file?"
        )
        change_name = prompt.addButton("Change name", QMessageBox.ButtonRole.ActionRole)
        overwrite = prompt.addButton("Overwrite", QMessageBox.ButtonRole.AcceptRole)
        prompt.addButton(QMessageBox.StandardButton.Cancel)
        prompt.setDefaultButton(change_name)
        prompt.exec()

        if prompt.clickedButton() is overwrite:
            return path
        if prompt.clickedButton() is change_name:
            self.output_filename_edit.setFocus()
            self.output_filename_edit.selectAll()
            self.status_label.setText(
                "Enter a new output filename, then press Play again."
            )
        return None

    def on_repeat_rate_changed(self, repeat_rate: float) -> None:
        if self.player.is_looping:
            try:
                self.player.set_repeat_rate(repeat_rate)
            except ValueError as error:
                self.show_error("Invalid repeat rate", error)
                return
            self.status_label.setText(
                f"Looping at {repeat_rate:g} full-sample starts per second."
            )

    def on_rate_slider_changed(self, position: int) -> None:
        repeat_rate = repeat_rate_for_slider(position)
        blocker = QSignalBlocker(self.rate_spin)
        self.rate_spin.setValue(repeat_rate)
        del blocker
        self.on_repeat_rate_changed(repeat_rate)

    def on_rate_spin_changed(self, repeat_rate: float) -> None:
        blocker = QSignalBlocker(self.rate_slider)
        self.rate_slider.setValue(slider_position_for_repeat_rate(repeat_rate))
        del blocker
        self.on_repeat_rate_changed(repeat_rate)

    def on_selection_changed(self) -> None:
        self.update_controls()

    def delete_selection(self) -> None:
        selection = self.waveform.selection
        if selection is None:
            return

        start, end = selection
        try:
            self.samples = delete_sample_range(self.samples, start, end)
        except ValueError as error:
            self.show_error("Could not delete the selection", error)
            return

        removed_duration = (end - start) / self.sample_rate
        self.waveform.set_samples(self.samples, self.sample_rate)
        self.status_label.setText(
            f"Deleted {removed_duration:.3f} seconds. "
            f"{len(self.samples) / self.sample_rate:.2f} seconds remain."
        )
        self.update_controls()

    def stop_playback(self) -> None:
        try:
            self.player.stop()
        except Exception as error:
            self.show_error("Could not stop playback cleanly", error)
            self.status_label.setText(
                "Playback stopped, but output could not be finalized."
            )
            self._active_output_path = None
            self.update_controls()
            return

        if self._active_output_path is not None:
            self.status_label.setText(
                f"Playback stopped. Saved to {self._active_output_path}."
            )
            self._active_output_path = None
        else:
            self.status_label.setText("Playback stopped.")
        self.update_controls()

    def refresh_status(self) -> None:
        while True:
            try:
                self.status_label.setText(self._audio_messages.get_nowait())
            except queue.Empty:
                break

        message = self.player.pop_audio_message()
        if message is not None:
            self.status_label.setText(message)

        if self._record_stream is not None:
            elapsed = time.monotonic() - self._record_started_at
            self.status_label.setText(f"Recording: {elapsed:.1f} seconds")

        if not self.player.is_playing and self.stop_button.isEnabled():
            try:
                self.player.stop()
            except Exception as error:
                self.show_error("Could not close the playback stream", error)
                self.status_label.setText(
                    "Playback finished, but output could not be finalized."
                )
                self._active_output_path = None
                self.update_controls()
                return

            if self._active_output_path is not None:
                self.status_label.setText(
                    f"Playback finished. Saved to {self._active_output_path}."
                )
                self._active_output_path = None
            else:
                self.status_label.setText("Playback finished.")
            self.update_controls()

    def update_controls(self) -> None:
        recording = self._record_stream is not None
        playing = self.player.is_playing
        self.record_button.setEnabled(not playing)
        self.open_button.setEnabled(not recording and not playing)
        can_play = not recording and not playing and self.samples.size > 0
        self.once_button.setEnabled(can_play)
        self.loop_button.setEnabled(can_play)
        self.stop_button.setEnabled(playing)
        can_adjust_rate = not recording and (
            not playing or self.player.is_looping
        )
        self.rate_slider.setEnabled(can_adjust_rate)
        self.rate_spin.setEnabled(can_adjust_rate)
        can_edit_output = not recording and not playing
        self.save_output_checkbox.setEnabled(can_edit_output)
        self.output_filename_edit.setEnabled(
            can_edit_output and self.save_output_checkbox.isChecked()
        )
        self.delete_button.setEnabled(
            not recording and not playing and self.waveform.selection is not None
        )

    def show_error(self, title: str, error: Exception) -> None:
        QMessageBox.critical(self, title, str(error))

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            if self._record_stream is not None:
                self.stop_recording()
            self.player.stop()
        except Exception as error:
            self.show_error("Audio cleanup failed", error)
        event.accept()


def main() -> int:
    application = QApplication(sys.argv)
    window = AudioSampleWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
