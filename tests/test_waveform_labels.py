import unittest

from PulseLooper import format_time_ms

class FormatTimeMillisecondsTests(unittest.TestCase):
    def test_formats_integer_milliseconds_without_trailing_zeroes(self) -> None:
        self.assertEqual(format_time_ms(1250), "1250 ms")

    def test_preserves_sub_millisecond_precision(self) -> None:
        self.assertEqual(format_time_ms(0.020833), "0.021 ms")

    def test_removes_unnecessary_decimal_zeroes(self) -> None:
        self.assertEqual(format_time_ms(125.5), "125.5 ms")

if __name__ == "__main__":
    unittest.main()
