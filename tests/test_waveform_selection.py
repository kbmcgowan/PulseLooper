import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from pyGrains import WaveformWidget


class WaveformSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.widget = WaveformWidget()

    def test_snaps_to_interpolated_sign_change(self) -> None:
        self.widget.set_samples(np.array([-3, 1], dtype=np.float32), 44_100)

        self.assertEqual(self.widget._snap_sample_index(0), 1)

    def test_snaps_both_selection_ends_to_exact_zero_samples(self) -> None:
        self.widget.set_samples(np.array([1, 0, 0, 1], dtype=np.float32), 44_100)

        self.widget._set_selection(0, 3)

        self.assertEqual(self.widget.selection, (1, 2))

    def test_keeps_pointer_position_when_there_are_no_crossings(self) -> None:
        self.widget.set_samples(np.array([1, 2, 3], dtype=np.float32), 44_100)

        self.assertEqual(self.widget._snap_sample_index(2), 2)


if __name__ == "__main__":
    unittest.main()
