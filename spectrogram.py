from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    dir_path = 'overlay_tests'
    
    for i in range(5):
        path = f'{dir_path}/random_overlay_{i}.wav'
        sample_rate, samples = wavfile.read(path)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        spectrogram = 10 * np.log10(spectrogram + 1e-9)
        
        plt.pcolormesh(times, frequencies, spectrogram, shading='auto')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
