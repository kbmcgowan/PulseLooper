import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_files import load_wav


class LoadWavTests(unittest.TestCase):
    def test_loads_stereo_wav_as_mono_and_preserves_sample_rate(self) -> None:
        stereo = np.array(
            [[0.2, 0.6], [-0.5, 0.1], [0.0, 0.8]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, stereo, 48_000, subtype="PCM_16")

            samples, sample_rate = load_wav(path)

        self.assertEqual(sample_rate, 48_000)
        self.assertEqual(samples.shape, (3,))
        np.testing.assert_allclose(samples, stereo.mean(axis=1), atol=1e-4)

    def test_rejects_empty_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.wav"
            sf.write(path, np.empty((0,), dtype=np.float32), 44_100)

            with self.assertRaises(ValueError):
                load_wav(path)


if __name__ == "__main__":
    unittest.main()
