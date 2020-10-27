from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
#Read the Wav file and plotting it
data,sampleRate = sf.read('102_1b1_Ar_sc_Meditron.wav')
fo, to, Sxxo = signal.spectrogram(data)
fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
plt.pcolormesh(to, fo, Sxxo, shading='gouraud')
ax.set_title('Original Signal')
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.grid(which='both', axis='both')

#Design the Filter and Plot it.
sos = signal.butter(20, [100,5000], btype='bandpass', analog=False, output='sos', fs=sampleRate)
w, h = signal.sosfreqz(sos, 44100, fs=44100)
ax = fig.add_subplot(3, 1, 2)
ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
ax.set_title('Butterworth bandpass frequency response')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Amplitude [dB]')
ax.axis((10, 10000, -100, 10))
ax.grid(which='both', axis='both')

#Filtering the signal and plotting the filtered Signal
y = signal.sosfilt(sos,data)
ff, tf, Sxxf = signal.spectrogram(y,return_onesided=False)
ax = fig.add_subplot(3, 1, 3)
plt.pcolormesh(tf, ff, Sxxf, shading='gouraud')
ax.set_title('Filtered Signal')
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.grid(which='both', axis='both')
plt.show()

sf.write('Filtered_Signal.wav', y, sampleRate)
