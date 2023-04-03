import librosa
import numpy as np


def features(arr:np.ndarray, frame_rate:int):
    mfcc = librosa.feature.mfcc(y=arr, sr=frame_rate)
    zcr = librosa.feature.zero_crossing_rate(y=arr)
    chroma_cqt = librosa.feature.chroma_cqt(y=arr, sr=frame_rate)
    chroma_stft = librosa.feature.chroma_stft(y=arr, sr=frame_rate)
    melspectrogram = librosa.feature.melspectrogram(y=arr, sr=frame_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=arr, sr=frame_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=arr, sr=frame_rate)
    spectral_flatness = librosa.feature.spectral_flatness(y=arr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=arr, sr=frame_rate)
    spectrogram = librosa.feature.melspectrogram(y=arr, sr=frame_rate)#, n_fft=2048, hop_length=1024, n_mels=128)
    tempogram = librosa.feature.tempogram(y=arr, sr=frame_rate)
    
    print(spectrogram.shape)


if __name__ == '__main__':
    from helpers import AudioSampler
    # data = AudioSampler()
    X, y, fr = AudioSampler.sample_array_2(5, True)
    print(X.shape)
    print(y.shape)
    # features(X, fr)
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    S_dB = librosa.power_to_db(X, ref=np.max)

    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=fr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    ax.set(title='Mel-frequency spectrogram')
    
    plt.show()
        # X, y = data.sample_array(1, window_size=100, convert_to_mono=True)
    # print(X.shape, y.shape)
    
    