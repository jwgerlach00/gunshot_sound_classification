from helpers import AudioSampler
import numpy as np

spectrograms, labels = AudioSampler.generate_dataset(10_000)
print(spectrograms.shape)
print(labels.shape)
np.save('dataset/spectrograms_10k.npy', spectrograms)
np.save('dataset/labels_10k.npy', labels)