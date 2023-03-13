from pydub import AudioSegment
# from pydub.playback import play
from random import randint
from typing import Tuple
import os
import yaml


def random_overlay(environment_path:str, overlay_path:str) -> Tuple[AudioSegment, int, int]:
    environment = AudioSegment.from_wav(environment_path)
    overlay = AudioSegment.from_wav(overlay_path)
    rand_pos = randint(0, len(environment) - len(overlay)) # ms
    rand_volume = randint(-10, 10) # dB
    return (
        environment.overlay(overlay + rand_volume, position=rand_pos),
        rand_pos,
        rand_volume
    )


if __name__ == '__main__':
    path_stem = 'kaggle_sounds'
    environment_path = f'example_from_rainforest_model.wav'
    overlay_path = f'{path_stem}/Zastava M92/9 (1).wav'
    
    out_dir = 'overlay_tests'
    os.remove(f'{out_dir}/metadata.yaml')
    for i in range(5):
        sound, pos, vol = random_overlay(environment_path, overlay_path)
        
        out_path = f'{out_dir}/random_overlay_{i}.wav'
        sound.export(out_path, format='wav')
        meta = {
            out_path.split('.')[0]: {
                'position_ms': pos,
                'volume_db': vol
            }
        }
        with open(f'{out_dir}/metadata.yaml', 'a') as f:
            yaml.dump(meta, f)
