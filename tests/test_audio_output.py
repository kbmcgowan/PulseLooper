import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd

from audio_engine import OverlapPlayer
from audio_files import WavOutputWriter


class WavOutputWriterTests(unittest.TestCase):
    def test_writes_playback_blocks_to_mono_pcm_wav(self) -> None:
        expected = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.wav"
            writer = WavOutputWriter(path, sample_rate=48_000)
            writer.append(expected[:1, None])
            writer.append(expected[1:, None])

            writer.finish()
            actual, sample_rate = sf.read(path, dtype="float32")

        self.assertEqual(sample_rate, 48_000)
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_abort_preserves_an_existing_output_file(self) -> None:
        original = np.array([0.1, 0.2], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.wav"
            sf.write(path, original, 44_100, subtype="PCM_16")
            writer = WavOutputWriter(path, sample_rate=44_100)
            writer.append(np.array([[0.8], [0.9]], dtype=np.float32))

            writer.abort()
            actual, _ = sf.read(path, dtype="float32")

        np.testing.assert_allclose(actual, original, atol=1e-4)

    def test_player_writes_the_generated_mixed_audio(self) -> None:
        expected = np.array([0.25, -0.5], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.wav"
            player = OverlapPlayer(sample_rate=100, block_size=4)
            player._sample = expected
            player._playing = True
            player._output_writer = WavOutputWriter(path, sample_rate=100)
            outdata = np.zeros((4, 1), dtype=np.float32)

            player._audio_callback(outdata, 4, None, sd.CallbackFlags())
            player.stop()
            actual, sample_rate = sf.read(path, dtype="float32")

        self.assertEqual(sample_rate, 100)
        np.testing.assert_allclose(
            actual, np.array([0.25, -0.5, 0, 0], dtype=np.float32), atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
