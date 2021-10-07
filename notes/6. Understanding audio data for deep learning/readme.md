# Understanding audio data for deep learning

## Sound

- Produced by the vibration of an object
- Vibration determines oscillation of air molecules
- Alternation of air pressure causes a wave

## Waveform

![Untitled](0.png)

Represented with amplitude & time

![Untitled](1.png)

Time that passes until the same wave repeats

![Untitled](2.png)

Frequency is the inverse of period.

![Untitled](3.png)

Distance from 0 amplitude

![Untitled](4.png)

This can be represented with a sine wave

## Frequency/pitch and amplitude/loudness

![Untitled](5.png)

Higher frequency → Higher pitch

![Untitled](6.png)

Higher amplitude → Louder

## Analog digital conversion (ADC)

![Untitled](7.png)

Blue point = Sample. Higher sample rate = more samples per second.

- Signal sampled at uniform time intervals
- Amplitude quantised with limited number of bits

CD:

- Sample rate = 44,100 Hz
- Bit depth = 16 bits/channel

8-bit old-school music soundtracks are called that because their bit depth is 8 bits/channel

## A real-world sound wave (piano key)

![Untitled](8.png)

How can we learn a lot about this sound?

### Fourier transform

- Decompose complex periodic sound into sum of sine waves oscillating at different frequencies

![Untitled](9.png)

The left sound wave is a combination of these two sine waves

![Untitled](10.png)

Amplitude in this case determines how much a specific sound contributes to the result. The bottom one, with 1.5 amplitude, contributes the most.

![Untitled](11.png)

Apply fourier transform to previous example

- From *time domain* to *frequency domain*
- No time information

### Short Time Fourier Transform (STFT)

- Computes several FFT at different intervals
- Preserves time information
- Fixed frame size (e.g., 2048 samples)
- Gives a spectrogram (time + frequency + magnitude)

![Untitled](12.png)

![Untitled](13.png)

STFT is performed by focussing on one frame (number of samples) at a time. We calculate the Fourier Transform and project it onto the spectrogram.

- In the Spectrogram we use dB. With dB we apply a logarithmic function to the magnitude itself
- When we perform the Fourier Transform, what we actually perform is the FFT. Fast Fourier Transform. A variation on FT which is way faster.

![Untitled](14.png)

Why did we learn about spectrograms? Spectrograms are fundamental in processing audio for deep learning. It will be used as an input for our DL model

## Traditional ML pre-processing pipeline for audio data

![Untitled](15.png)

- Feature engineering
- Perform STFT
- Extract time + frequency domain features

With DL we don't need to focus that much on feature engineering, we can just use the spectrogram.

## Mel Frequency Cepstral Coefficients (MFCCs)

- Capture timbral/textural aspects of sound
- Frequency domain feature
- Approximate human auditory system
- 13 to 40 coefficients
- Calculated at each frame

![Untitled](16.png)

Representation of an MFCC

### MFCCs applications

- Speech recognition
- Music genre classification
- Music instrument classification
- ...

We don't need the mathematical and theoretical information about MFCCs and STFT for DL.

### DL Pre-processing pipeline for audio data

![Untitled](17.png)