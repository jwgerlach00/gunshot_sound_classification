from pydub import AudioSegment
# from pydub.playback import play
import random
from typing import Dict, Union
import os
import yaml
import numpy as np
import scipy
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tqdm import tqdm

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def trim_audio(audio):
    start_trim = detect_leading_silence(audio)
    end_trim = detect_leading_silence(audio.reverse())

    duration = len(audio)    
    trimmed_sound = audio[start_trim:duration-end_trim]
    return trimmed_sound

class AudioSampler:
    #random.seed(42)
    
    def __init__(self):
        pass
        
    @staticmethod
    def random_overlay(environment_path:str, overlay_path:str, frame_rate_multiplier:bool=False, match_fr=False) -> Dict[str, Union[AudioSegment, int]]:
        '''
        Overlays an audio file on top of another (environmental) audio file.
        Randomizes the position and volume of the overlay.
        '''
        ENV_LENGTH = 5 *1000 # 5 seconds
        environment = AudioSegment.from_wav(environment_path)
        env_clip_start = random.randint(0, len(environment) - ENV_LENGTH)
        env_clip = environment[env_clip_start:env_clip_start+ENV_LENGTH]

        overlay = AudioSegment.from_wav(overlay_path)
        overlay = trim_audio(overlay)
        if match_fr and overlay.frame_rate != match_fr:
            overlay = overlay.set_frame_rate(match_fr) # Give everything a consistent frame rate
        try:
            rand_pos = random.randint(0, ENV_LENGTH - len(overlay)) # ms
        except:
            print(len(overlay), len(env_clip))
        rand_volume = random.randint(-10, 10) # dB
        
        out_clip = env_clip.overlay(overlay + rand_volume, position=rand_pos)
        out_clip = out_clip.low_pass_filter(16000).high_pass_filter(5000)
        y = []
        for x in range(ENV_LENGTH):
            if x < rand_pos or x > rand_pos+len(overlay):
                if frame_rate_multiplier:
                    y.extend([0]*(out_clip.frame_rate//1000)) # 1000 ms
                else:
                    y.append(0)
            else:
                if frame_rate_multiplier:
                    y.extend([1]*(out_clip.frame_rate//1000)) # 1000 ms
                else:
                    y.append(1)

        return {
            'audio': out_clip,
            'pos': rand_pos,
            'volume': rand_volume,
            'y' :  y
        }

    @staticmethod
    def pydub_data(audio:AudioSegment) -> Dict[str, Union[np.ndarray, int]]:
        '''
        Converts pydub audio data to numpy array and frame rate.
        '''
        return {
            'arr': np.array(audio.get_array_of_samples()),
            'fr': audio.frame_rate
        }
        
    @staticmethod
    def spectrogram(arr:np.ndarray, frame_rate:int) -> Figure:
        '''
        Plots a spectrogram of numpy array audio data using matplotlib.
        '''
        f, t, Sxx = scipy.signal.spectrogram(arr, fs=frame_rate)
        Sxx = 10 * np.log10(Sxx + 1e-9)
        
        fig = plt.figure()
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        plt.close()
        
        return f, t, Sxx
    
    def sample_generator(self, n:int, convert_to_mono:bool) -> Dict[str, Union[AudioSegment, int, np.ndarray, dict]]:
        '''
        Generator function for randomizing a set of audio samples.
        Returns metadata and spectrogram and numpy array of each sample in addition to the pydub audio object.
        '''
        value = 0
        while value < n:
            audio = AudioSampler.random_overlay(self.environment_path, self.overlay_path)
            
            if convert_to_mono:
                audio['audio'] = audio['audio'].set_channels(1)
            
            meta = {
                'position_ms': audio['pos'],
                'volume_db': audio['volume']
            }
            
            # sample = 
            # sample['sound'] = audio['audio']
            # sample['meta'] = meta
            
            yield AudioSampler.pydub_data(audio['audio'])['arr'], audio['y']
            value += 1
    
    @staticmethod
    def sample_array_2(n, convert_to_mono):
        # enrionment_path
        environment_dir = 'environment_sounds'
        all_environment_files = []
        for x in os.listdir(environment_dir):
            if x.endswith(".wav"):
                all_environment_files.append(x)

        # gunshot path
        overlay_dir = 'kaggle_sounds'
        all_overlay_files = []
        for x in os.listdir(overlay_dir):
            for y in os.listdir(f'{overlay_dir}/{x}'):
                overlay = AudioSegment.from_wav(f'{overlay_dir}/{x}/{y}')
                if len(overlay) < 5000:
                    all_overlay_files.append(f'{x}/{y}')
                else:
                    print(f'{x}/{y} is too long ({len(overlay)}ms)')
        
        X = []
        Y = []
        # max_length = 0
        max_fr = 0
        for _ in tqdm(range(n)):
            random_environment_file = random.choice(all_environment_files)
            environment_path = f'{environment_dir}/{random_environment_file}'

            random_overlay_file = random.choice(all_overlay_files)
            overlay_path = f'{overlay_dir}/{random_overlay_file}'

            audio = AudioSampler.random_overlay(environment_path, overlay_path, frame_rate_multiplier=True, match_fr=48000)
            if convert_to_mono:
                audio['audio'] = audio['audio'].set_channels(1)
            
            x = AudioSampler.pydub_data(audio['audio'])['arr']
            X.append(x)
            Y.append(audio['y'])
            max_fr = audio['audio'].frame_rate if audio['audio'].frame_rate > max_fr else max_fr
            # max_len = len(x) if len(x) > max_length else max_length
            
        # for i, (x, y) in enumerate(zip(X, Y)):
        #     if len(x) < max_len:
        #         X[i] = np.pad(x, (0, max_len-len(x)), 'constant')
        #         Y[i] = np.pad(y, (0, max_len-len(x)), 'constant')
            
        return (
            np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32),
            max_fr
        )
                    
        
        
            
    def sample_array(self, n:int, window_size:int, convert_to_mono:bool,output_spectrogram=False) -> Dict[str, Union[np.ndarray, int, dict]]:
        '''
        Returns a numpy array of audio samples.
        '''

        # enrionment_path
        environment_dir = 'environment_sounds'
        all_environment_files = []
        for x in os.listdir(environment_dir):
            if x.endswith(".wav"):
                all_environment_files.append(x)

        

        # gunshot path
        overlay_dir = 'kaggle_sounds'
        all_overlay_files = []
        for x in os.listdir(overlay_dir):
            if os.path.isdir(f'{overlay_dir}/{x}'):
                for y in os.listdir(f'{overlay_dir}/{x}'):
                    all_overlay_files.append(f'{x}/{y}')

        

        X = []
        y = []
        print('Generating Dataset...')
        for _ in tqdm(range(n)):
            random_environment_file = random.choice(all_environment_files)
            self.environment_path = f'{environment_dir}/{random_environment_file}'

            random_overlay_file = random.choice(all_overlay_files)
            self.overlay_path = f'{overlay_dir}/{random_overlay_file}'

            audio = AudioSampler.random_overlay(self.environment_path, self.overlay_path)
            
            if convert_to_mono:
                audio['audio'] = audio['audio'].set_channels(1)
            
            meta = {
                'position_ms': audio['pos'],
                'volume_db': audio['volume']
            }
            
            clip_data = AudioSampler.pydub_data(audio['audio'])
            clip = clip_data['arr']
            labels = audio['y']
            frame_rate = clip_data['fr']
            for i in range(len(clip)):
                try:
                    w = len(clip[i:i+window_size])
                    if w == window_size:
                        y.append(labels[i:i+window_size][-1])
                        X.append(clip[i:i+window_size])
                except:
                    pass
        
        X = np.array(X,dtype=np.float32)
        y = np.array(y,dtype=np.float32)
        if output_spectrogram:
            #output the windows as spectrograms
            temp = []
            for windowed_sound_clip in X:
                f, t, Sxx = AudioSampler.spectrogram(windowed_sound_clip, frame_rate)
                temp.append(Sxx)
            return temp,y
        else:
            return X,y



if __name__ == '__main__':
    # Environment Path
    environment_path = f'city.wav'
    
    path_stem = 'kaggle_sounds'
    overlay_path = f'{path_stem}/Zastava M92/9 (1).wav'
    
    out_dir = 'overlay_tests'
    if os.path.exists(f'{out_dir}/metadata.yaml'):
        os.remove(f'{out_dir}/metadata.yaml')
    
    audio = AudioSampler()
    arr = audio.sample_array(5, 5, True, False)
    print(arr)
    
    ''' Not needed now
    for i, x in enumerate(audio.sample_generator(5, True)):
        
        AudioSampler.spectrogram(x['arr'], x['fr'])
        plt.savefig(f'{out_dir}/random_overlay_{i}.png')
        
        out_path = f'{out_dir}/random_overlay_{i}.wav'
        x['sound'].export(out_path, format='wav')
        meta = { out_path.split('.')[0]: x['meta'] }
        
        with open(f'{out_dir}/metadata.yaml', 'a') as f:
            yaml.dump(meta, f)
    '''
