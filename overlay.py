from pydub import AudioSegment
from pydub.playback import play
from random import randint


def random_overlay(environment_path:str, overlay_path:str) -> AudioSegment:
    environment = AudioSegment.from_wav(environment_path)
    overlay = AudioSegment.from_wav(overlay_path)
    r = randint(0, len(environment) - len(overlay))
    return environment.overlay(overlay, position=r)


if __name__ == '__main__':
    path_stem = 'kaggle_sounds'
    environment_path = f'example_from_rainforest_model.wav'
    overlay_path = f'{path_stem}/Zastava M92/9 (1).wav'
    
    out_dir = 'overlay_tests'
    for i in range(5):
        random_overlay(environment_path, overlay_path).export(f'{out_dir}/random_overlay{i}.wav', format='wav')
