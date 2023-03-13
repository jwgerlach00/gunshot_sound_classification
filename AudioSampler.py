from pydub import AudioSegment
# from pydub.playback import play
import random
from typing import Tuple, Dict, Union
import os
import yaml
import audiosegment
import pydub
import numpy as np
import scipy
import matplotlib.pyplot as plt


class AudioSampler:
    random.seed(42)
    
    def __init__(self, environment_path:str, overlay_path:int):
        self.environment_path = environment_path
        self.overlay_path = overlay_path
        
    @staticmethod
    def random_overlay(environment_path:str, overlay_path:str) -> Dict[str, Union[AudioSegment, int]]:
        environment = AudioSegment.from_wav(environment_path)
        overlay = AudioSegment.from_wav(overlay_path)
        rand_pos = random.randint(0, len(environment) - len(overlay)) # ms
        rand_volume = random.randint(-10, 10) # dB
        return {
            'audio': environment.overlay(overlay + rand_volume, position=rand_pos),
            'pos': rand_pos,
            'volume': rand_volume
        }

    @staticmethod
    def pydub_data(audio:AudioSegment, convert_to_mono:bool) -> Dict[str, Union[np.ndarray, int]]:
        if convert_to_mono:
            audio = audio.set_channels(1)

        return {
            'arr': np.array(audio.get_array_of_samples()),
            'fr': audio.frame_rate
        }
    
    def sample_generator(self, n):
        value = 0
        while value < n:
            audio = AudioSampler.random_overlay(self.environment_path, self.overlay_path)
            
            meta = {
                'position_ms': audio['pos'],
                'volume_db': audio['volume']
            }
            
            two_channel = AudioSampler.pydub_data(audio['audio'], False)
            two_channel['sound'] = audio['audio']
            two_channel['meta'] = meta
            
            yield two_channel
            value += 1


if __name__ == '__main__':
    path_stem = 'kaggle_sounds'
    environment_path = f'example_from_rainforest_model.wav'
    overlay_path = f'{path_stem}/Zastava M92/9 (1).wav'
    
    out_dir = 'overlay_tests'
    if os.path.exists(f'{out_dir}/metadata.yaml'):
        os.remove(f'{out_dir}/metadata.yaml')
    
    audio = AudioSampler(environment_path, overlay_path)
    
    for i, x in enumerate(audio.sample_generator(5)):
        
        f, t, Sxx = scipy.signal.spectrogram(x['arr'], fs=x['fr'])
        Sxx = 10 * np.log10(Sxx + 1e-9)
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        
        out_path = f'{out_dir}/random_overlay_{i}.wav'
        x['sound'].export(out_path, format='wav')
        meta = { out_path.split('.')[0]: x['meta'] }
        
        with open(f'{out_dir}/metadata.yaml', 'a') as f:
            yaml.dump(meta, f)
