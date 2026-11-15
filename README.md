# pyGrains - record a pulse and loop it

pyGrains (formerly pyGrains) is a tool for teaching that "pitch"
is a perceptual phenomenon about the way we experience frequency
by taking a single pulse and looping it.  Uses granular synthesis
in python. Designed to support a lesson that teaches:

    1. 'Pitch' is a perceptual phenomenon that happens in the listener, not a fact about the universe
    2. When we talk about the 'frequency' of a sound in Hertz, we're only talking about periodic sounds
    3. 'Frequency' in Hz is just the number of times a sound repeats in 1 second

Things that are implicitly demonstrated but needn't be called-out if it's too much for the room: our experience of frequency is logarithmic, destructive interference (see Quirks, below), 

**Important note**: the granular synthesis method used here means that the perceived amplitude of generated tones will *not* behave as expected (with low frequency tones being substantially quieter than 1,000 Hz tones and then perceived loudness beginning to drop off again above about 5,000 Hz). This is because of the unpredicable way the grains will overlap. With sample-pulses/SinePeriod-2.279ms.wav, for example, pulses will be separate at 100 Hz, but will begin to overlap at 1,000 Hz (so amplitudes may happen to sum constructively or destructively) and will overlap substantially/often at 10,000 Hz. The resulting RMS of the synthesized tone is unpredictable and therefore not a good way to demonstrate the relationship between frequency and loudness. It's up to you if you want to call this out in class or leave it for another time, but if some clever student asks (either right away or when you get to psychoacoustics) why this tool doesn't do what the textbook says it should do you can point out that it DOES, actually, but that the frequency sensitivity diagrams one sees in psychoacoustics textbooks are all created with pure tones that don't have this interference issue. See also: accidental demonstration of phase cancellation which is discussed in the Quirks.

**AI assistance**: Developed with help from Copilot, an AI assistant using Copilot SDK in VS Code

## Requirements 

- Python 3.10 or later
- PySide6, NumPy, sounddevice, and SoundFile
- Working portaudio

## Install & Run

This works on my mac. Your python3 binary might be called something else.

### Install

```sh
python3 -m venv .pyGrains-venv
source .pyGrains-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
deactivate
```

### Run
```sh
source .pyGrains-venv/bin/activate
python pyGrains.py &
```

`deactivate` will exit the venv

## Sample Pulse

Snap your fingers, clap your hands, toot a whistle, strike a cowbell, whatever you want to try that is fun. Save the sampled pulse to a file (see below) and then edit it.  Explain why even sharply-ended sources decay (maybe re-record with the source surrounded by a foam box to show that the decay is echo).

## Sample Pulses

Note: these are not ideal sines

- sample-pulses/SinePeriod-2.279ms.wav: (A4) single period from a 440 Hz tone generator.
- sample-pulses/SinePeriod-3.810ms.wav: (C4) single period from a 262.626 tone generator
- sample-pulses/SinePeriod-4.527ms.wav: (A3) single period from a 220 Hz tone generator.
- sample-pulses/SinePeriod-4.558ms.wav: another A3 to compare
sample-pulses/SyntheticSchwa-140Hz.wav
- SyntheticSchwa-140Hz.wav: single period from a Praat vowel editor, schwa-like formants, 140 Hz F0

## Instructions

- Press **Record** to capture from the default input device, or **Open
WAV** to load a WAV file.
- Multi-channel files are mixed down to mono and retain their original sample rate.
- Select snaps to nearest zero-crossing if available
- **Delete selection** to remove the selected range (there is no undo, but you could save the pulse before editing, see below)
- **Play once** plays a single sample and
- **Play loop** repeats indefinitely;
- **Stop** ends playback. 

### Saving

- check **Save output to wav file** to write the generated mono audio, use this with **Play once** to save the pulse before editing


## Known Quirks and Future Changes

- Infrasound versions of notes (e.g. 1.02198 Hz sine pulse for an infra C and 1.25992 Hz for an infra E note respectively) do not behave as expected.
- Destructive Interference, sorta: Try exactly doubling the looping frequency given by f = 1/ Loop Period and you'll get destructive interference because the grains are 180º out of phase. Single pulses of a sine wav that begin and end neatly at zero-crossings result in complete silence. Maybe a neat way to introduce this idea if you save the output and examine it in Praat/Phonlab/whatever, but this was not something I intended
    - `sample-pulses/SinePeriod-2.268ms.wav` contains 1,024 frames at 44.1 kHz but only the first 100 frames (2.268ms) contain non-zero audio -- implying a cycle near 440 Hz or A4 on a conventionally tuned piano. If the looper is set to 880 starts per second, the starts will be about 50 samples apart—roughly half that sine cycle. The overlapping parts therefore have opposite phase and _mostly_ cancel (SinePeriod-2.268ms.wav is not an ideal sine). To dig into the code:

    - `schedule_grain_starts` sets the spacing of the loop,
    - `mix_overlapping_grains` sums the overlapping copies, and
    - `_audio_callback` assembles the output.
    -  https://en.wikipedia.org/wiki/Granular_synthesis

- perhaps add an output waveform viewer that appears after synthesis?  UI implications are weird, though, and I'm not sure how to handle it. Separate tab?

