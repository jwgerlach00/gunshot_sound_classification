from pydub import AudioSegment
# from pydub.playback import play
import random
from typing import Dict, Union, Optional
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
    random.seed(42)
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_environment_paths(environment_dir:str='environment_sounds'):
        all_environment_files = []
        for x in os.listdir(environment_dir):
            if x.endswith(".wav"):
                all_environment_files.append(os.path.join(environment_dir, x))
        return all_environment_files
    
    @staticmethod
    def get_gunshot_paths(overlay_dir:str='kaggle_sounds'):
        all_overlay_files = []
        for x in os.listdir(overlay_dir):
            if x != '.DS_Store':
                for y in os.listdir(f'{overlay_dir}/{x}'):
                    if y.endswith(".wav"):
                        all_overlay_files.append(os.path.join(overlay_dir, x, y))
        return all_overlay_files
        
    @staticmethod
    def random_overlay(environment_path:str, overlay_path:str, frame_rate_multiplier:bool=False,
                       match_fr:Optional[int]=None) -> Dict[str, Union[AudioSegment, int]]:
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
        # Db scale (I think)
        Sxx = 10 * np.log10(Sxx + 1e-9)
        return f, t, Sxx
    
    @staticmethod
    def plot_spectrogram(t:np.ndarray, f:np.ndarray, Sxx:np.ndarray, y:Optional[np.ndarray]) -> tuple:
        '''
        Plots the spectrogram and plots the boundaries for the y bit-vector if y is provided.
        '''
        fig, ax = plt.subplots()
        # Shade according to the spectrogram
        ax.pcolormesh(t, f, Sxx, shading='auto')
        
        # Add boundary lines for y bit-vector if provided
        if isinstance(y, np.ndarray) or y:
            # Find the indices of all the elements equal to 1
            indices = np.where(y == 1)[0]
            # Get the index of the first element equal to 1
            first_index = indices[0]
            # Get the index of the last element equal to 1
            last_index = indices[-1]
            
            ax.axvline(x=t[first_index], color='r')
            ax.axvline(x=t[last_index], color='r')
        
        # Add labels
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Time [sec]')
        ax.set_title('Spectrogram')
        return fig, ax
    
    @staticmethod
    def spectrogram_downsample_y(y_ms, x_time):
        '''
        Downsamples y_ms to fit x_time using a nearest-neighbor approach. y is assumed to be in milliseconds while
        x_time is assumed to be in seconds.
        '''
        y_time = np.array(range(0, len(y_ms))) / 1000 # ms to s
        out = []
        for x_t in x_time:
            diff = np.absolute(y_time - x_t)
            index = diff.argmin()
            out.append(y_ms[index])
        return np.array(out)
    
    @staticmethod
    def sample_spectrogram(n:int, convert_to_mono:bool=True, show_plot:bool=False) -> tuple:
        '''
        Generates a random spectrogram and its corresponding y bit-vector (n) times. Randomly chooses an environment
        and overlay file from paths specified within the classmethods.
        '''
        all_environment_files = AudioSampler.get_environment_paths()
        all_overlay_files = AudioSampler.get_gunshot_paths()
        
        spectrograms = []
        labels = []
        for _ in tqdm(range(n)):
            # Randomly choose an environment and overlay file
            random_environment_path = random.choice(all_environment_files)
            random_overlay_path = random.choice(all_overlay_files)

            # Generate the audio data
            audio = AudioSampler.random_overlay(random_environment_path, random_overlay_path,
                                                frame_rate_multiplier=False, match_fr=None)
            
            # Convert to a single channel if specified
            if convert_to_mono:
                audio['audio'] = audio['audio'].set_channels(1)
            
            clip_data = AudioSampler.pydub_data(audio['audio'])
            clip = clip_data['arr']
            frame_rate = clip_data['fr']
            
            # Generate the spectrogram
            f, t, Sxx = AudioSampler.spectrogram(clip, frame_rate)
            
            spectrograms.append(Sxx)
            labels.append(AudioSampler.spectrogram_downsample_y(audio['y'], t))
            
            # Plot the spectrogram if specified
            if show_plot:
                AudioSampler.plot_spectrogram(t, f, Sxx, labels[-1])
                plt.show()

        return spectrograms, labels
                    
            
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
            if x != '.DS_Store':
                for y in os.listdir(f'{overlay_dir}/{x}'):
                    if y != '.DS_Store':
                        all_overlay_files.append(f'{x}/{y}')

        

        X = []
        y = []
        spectrograms = []
        clips = []
        print('Generating Dataset...')
        for _ in tqdm(range(n)):
            random_environment_file = random.choice(all_environment_files)
            self.environment_path = f'{environment_dir}/{random_environment_file}'

            random_overlay_file = random.choice(all_overlay_files)
            self.overlay_path = f'{overlay_dir}/{random_overlay_file}'

            audio = AudioSampler.random_overlay(self.environment_path, self.overlay_path, frame_rate_multiplier=True, match_fr=48000)
            
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
            if output_spectrogram:
                f, t, Sxx = AudioSampler.spectrogram(clip, frame_rate)
                # spectrograms.append((f, t, Sxx))
                clips.append(clip)
                spectrograms.append(Sxx)
                # Sxx = np.array(Sxx).swapaxes(0,1)
                # for i in range(len(Sxx)):
                #     try:
                #         w = len(Sxx[i:i+window_size])
                #         if w == window_size:
                #             y.append(labels[i:i+window_size][-1])
                #             X.append(Sxx[i:i+window_size])
                #     except:
                #         pass
            else:
                for i in range(len(clip)):
                    try:
                        w = len(clip[i:i+window_size])
                        if w == window_size:
                            y.append(labels[i:i+window_size][-1])
                            X.append(clip[i:i+window_size])
                    except:
                        pass
        if output_spectrogram:
            return spectrograms
        X,y = np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)
        return X,y,spectrograms



if __name__ == '__main__':
    spectrograms, labels = AudioSampler.sample_spectrogram(5, True)