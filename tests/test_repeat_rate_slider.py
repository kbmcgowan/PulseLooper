import unittest

from pyGrains import (
    RATE_SLIDER_STEPS,
    repeat_rate_for_slider,
    slider_position_for_repeat_rate,
)


class LogarithmicRepeatRateSliderTests(unittest.TestCase):
    def test_quarter_positions_span_equal_frequency_ratios(self) -> None:
        positions = [0, 2500, 5000, 7500, RATE_SLIDER_STEPS]

        rates = [repeat_rate_for_slider(position) for position in positions]

        self.assertEqual(rates, [1, 10, 100, 1000, 10_000])

    def test_frequency_maps_back_to_slider_position(self) -> None:
        for position in (0, 2500, 5000, 7500, RATE_SLIDER_STEPS):
            with self.subTest(position=position):
                rate = repeat_rate_for_slider(position)
                self.assertEqual(slider_position_for_repeat_rate(rate), position)

    def test_spinbox_values_are_validated(self) -> None:
        for rate in (0, 10_001):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    slider_position_for_repeat_rate(rate)


if __name__ == "__main__":
    unittest.main()
