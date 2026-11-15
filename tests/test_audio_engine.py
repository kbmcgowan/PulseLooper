import unittest

import numpy as np
import sounddevice as sd

from audio_engine import (
    OverlapPlayer,
    delete_sample_range,
    mix_overlapping_grains,
    schedule_grain_starts,
)


class DeleteSampleRangeTests(unittest.TestCase):
    def test_removes_selected_samples_and_joins_remaining_audio(self) -> None:
        sample = np.array([0, 1, 2, 3, 4], dtype=np.float32)

        result = delete_sample_range(sample, 1, 4)

        np.testing.assert_array_equal(result, np.array([0, 4], dtype=np.float32))
        np.testing.assert_array_equal(sample, np.array([0, 1, 2, 3, 4], dtype=np.float32))

    def test_allows_deleting_the_entire_sample(self) -> None:
        sample = np.array([0, 1, 2], dtype=np.float32)

        result = delete_sample_range(sample, 0, len(sample))

        self.assertEqual(result.size, 0)

    def test_rejects_empty_or_out_of_bounds_ranges(self) -> None:
        sample = np.array([0, 1, 2], dtype=np.float32)

        for start, end in ((1, 1), (-1, 2), (0, 4)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    delete_sample_range(sample, start, end)


class MixOverlappingGrainsTests(unittest.TestCase):
    def test_mix_single_grain_across_a_partial_block(self) -> None:
        sample = np.array([0.25, 0.5, 0.75], dtype=np.float32)

        result = mix_overlapping_grains(sample, [2], block_start=0, frame_count=6)

        np.testing.assert_array_equal(
            result, np.array([0, 0, 0.25, 0.5, 0.75, 0], dtype=np.float32)
        )

    def test_sums_overlapping_grains_and_handles_block_continuation(self) -> None:
        sample = np.array([1, 2, 3], dtype=np.float32)

        result = mix_overlapping_grains(sample, [0, 1], block_start=2, frame_count=3)

        np.testing.assert_array_equal(result, np.array([5, 3, 0], dtype=np.float32))


class ScheduleGrainStartsTests(unittest.TestCase):
    def test_schedules_ten_thousand_starts_in_one_second(self) -> None:
        starts, next_start = schedule_grain_starts(
            frame_count=44_100,
            sample_rate=44_100,
            repeat_rate=10_000,
            next_start=0,
        )

        self.assertEqual(len(starts), 10_000)
        self.assertEqual(starts[0], 0)
        self.assertLess(starts[-1], 44_100)
        self.assertAlmostEqual(next_start, 0, places=7)

    def test_split_blocks_preserve_fractional_start_phase(self) -> None:
        first, next_start = schedule_grain_starts(512, 44_100, 997, 0)
        second, next_start = schedule_grain_starts(512, 44_100, 997, next_start)
        combined, _ = schedule_grain_starts(1_024, 44_100, 997, 0)

        self.assertEqual(first + [start + 512 for start in second], combined)
        self.assertGreaterEqual(next_start, 0)
        self.assertLess(next_start, 44_100 / 997)


class LiveRepeatRateTests(unittest.TestCase):
    def test_rate_change_applies_at_the_next_callback(self) -> None:
        player = OverlapPlayer(sample_rate=100, block_size=4)
        player._sample = np.ones(1000, dtype=np.float32)
        player._playing = True
        player._loop = True
        player._repeat_rate = 1
        outdata = np.zeros((4, 1), dtype=np.float32)

        player._audio_callback(outdata, 4, None, sd.CallbackFlags())
        self.assertEqual(player._active_starts, [0])

        player.set_repeat_rate(25)
        player._audio_callback(outdata, 4, None, sd.CallbackFlags())

        self.assertEqual(player._repeat_rate, 25)
        self.assertEqual(player._active_starts, [0, 4])

        player._audio_callback(outdata, 4, None, sd.CallbackFlags())
        self.assertEqual(player._active_starts, [0, 4, 8])

        player._audio_callback(outdata, 4, None, sd.CallbackFlags())

        self.assertEqual(player._active_starts, [0, 4, 8, 12])

    def test_rejects_repeat_rates_outside_supported_range(self) -> None:
        player = OverlapPlayer()

        for repeat_rate in (0, 10_001):
            with self.subTest(repeat_rate=repeat_rate):
                with self.assertRaises(ValueError):
                    player.set_repeat_rate(repeat_rate)


if __name__ == "__main__":
    unittest.main()
