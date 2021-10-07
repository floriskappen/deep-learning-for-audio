import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "Data/genres_original/blues/blues.00000.wav"

# Waveform
signal, sr = librosa.load(file, sr=22050) # sr * T -> 22050 * 30
librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# FFT -> Spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft) # Contribution to each frequency bin to the overal sound
frequency = np.linspace(0, sr, len(magnitude))

left_requency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

plt.plot(left_requency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

# STFT -> Spectrogram

# These are quite ordinary values when analyzing music.
n_fft = 2048 # Number of samples. The window we consider when performing a single fast fourier transform
hop_length = 512 # Number of samples. The amount we are shifting each fourier transform to the right.

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

# MFCCs
n_mfcc = 13 # Commonly used for analyzingb music
MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
