import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import time
import datetime
from pytz import timezone
from playsound import playsound
from IPython.display import clear_output
import random
import os.path

warnings.filterwarnings("error")

def print_bytes(byte_array):
    p = "0x"
    for b in byte_array:
        x = str(hex(b))[2:]
        if (len(x) < 2):
            x = "0" + x
        p += x
    return p

def mask_num(mask, num):
    if (num < 0):
        return -(abs(num) & mask)
    else:
        return (num & mask)

def int32_to_frame(left, right):
    return (np.int64(right) << 32) + (left)

def save_clip_stereo(waveform, file_name, sampwidth, framerate, nframes):
    with wave.open(file_name, 'wb') as fp:
        fp.setnchannels(2)
        fp.setsampwidth(sampwidth)
        fp.setframerate(framerate)
        fp.setnframes(nframes)
        for i in range(nframes):
            left  = waveform[0][i]
            right = waveform[1][i]
            try:
                fp.writeframes(int32_to_frame(left, right))
            except RuntimeWarning:
                pass
#                 print("Left <{}>: {} Right <{}>: {}".format(type(left), hex(left), type(right), hex(right)))

class audio_file(object):
    def __init__(self, file_name):
        self.file_name    = file_name
        self.audio_sample = wave.open(self.file_name,'r')
        self.num_channels = self.audio_sample.getnchannels()
        self.byte_depth   = self.audio_sample.getsampwidth()
        self.sample_freq  = self.audio_sample.getframerate()
        self.num_frames   = self.audio_sample.getnframes()
        self.frame_length = self.num_channels * self.byte_depth
        self.load_data()

    def load_data(self):
        self.audio_waveform_32bit = np.zeros(shape=(self.num_channels,self.num_frames), dtype=np.int32)
        self.audio_frames         = []
        up_sampled_bits = 8 * (4 - self.byte_depth)
        for j in range(self.num_frames):
            waveform_sample = self.audio_sample.readframes(1)
            self.audio_frames += [waveform_sample]
            for i in range(self.num_channels):
                self.audio_waveform_32bit[i][j] = int.from_bytes(
                    waveform_sample[i * self.byte_depth: (i + 1) * self.byte_depth], "little", signed=True) << up_sampled_bits

    def info_dump(self):
        print("File Name:     ", self.file_name)
        print("Channels:      ", self.num_channels)
        print("Byte Depth:    ", self.byte_depth)
        print("Sampling rate: ", self.sample_freq)
        print("Frames:        ", self.num_frames)

    def graph_waveform(self, start_sample=0, end_sample=500):
        """Plots left and right waveforms
        """
        plt.figure(figsize=(20,8))
        plt.subplot(2, 1, 1)
        plt.plot(self.audio_waveform_32bit[0][start_sample:end_sample], 'r')
        plt.title("Left")
        plt.subplot(2, 1, 2)
        plt.plot(self.audio_waveform_32bit[1][start_sample:end_sample], 'b')
        plt.title("Right")
        plt.show()


    def save_32bit_waveform(self, file_name):
        """Saves a signal as a 32 bit PCM WAV file
        """
        save_clip_stereo(self.audio_waveform_32bit, file_name, 4, self.sample_freq, self.num_frames)

def down_sample(waveform, samp_rate_ratio):
    """Down samples by a ratio - resulting frequency must be an integer
    """
    return [waveform[0][::samp_rate_ratio], waveform[1][::samp_rate_ratio]]
#     save_clip_stereo(waveform, file_name, 4, self.sample_freq // samp_rate_ratio, self.num_frames // samp_rate_ratio)

def up_sample(samp_rate_ratio, file_name):
    """Deprecated
    """
    waveform = []
    for i in range(self.num_channels):
        x = self.audio_waveform_32bit[i]
        y = np.concatenate((x,x,x))
        re_sampl = signal.resample(y, self.num_frames * 3 * samp_rate_ratio)
        start = self.num_frames * samp_rate_ratio
        end =  2 * (self.num_frames * samp_rate_ratio)
        new_sig = [np.int32(x) for x in re_sampl[start:end]]
        print("Length of resampled sig: {}\nStart: {}\nEnd: {}\nNew Sig Len: {}".format(len(re_sampl), start, end, len(new_sig)))
        waveform += [new_sig]
    with wave.open(file_name, 'wb') as fp:
        fp.setnchannels(self.num_channels)
        fp.setsampwidth(4)
        fp.setframerate(self.sample_freq * samp_rate_ratio)
        fp.setnframes(self.num_frames * samp_rate_ratio)
        for i in range(self.num_frames):
            left  = waveform[0][i]
            right = waveform[1][i]
            fp.writeframes(int32_to_frame(left, right))

def res_down(waveform, num_bits):
    """Reduce Bit Depth
    """
    mask = np.uint32(((0x1 << num_bits) - 1) << (32 - num_bits))
    left  = np.array([mask_num(mask, x) for x in waveform[0]], dtype=np.int32)
    right = np.array([mask_num(mask, x) for x in waveform[1]], dtype=np.int32)
    return [left, right]
#     save_clip_stereo(wf, file_name, 4, self.sample_freq, self.num_frames)

def compare_wavs(file1, file2):
    playsound(file1)
    time.sleep(0.5)
    playsound(file2)

def get_datetime():
    return datetime.datetime.now(tz=datetime.timezone.utc).astimezone(timezone('US/Pacific')).strftime("%m-%d-%Y %H:%M:%S")

def generate_wav_files(file_name, bit_res=[24,16], freq_res=[1,2,4]):
    aud_file = audio_file(file_name + '.wav')
    aud_file.info_dump()

    destroyed_waveforms = {}
    list_file_names= []
    for i in bit_res:
        for j in freq_res:
            fn = "{}_{}bit_{}kHz.wav".format(file_name, i, 44.1 / j)
            list_file_names += [fn]
            if (not os.path.exists(fn)):
                destroyed_waveforms[(i, j)] = res_down(down_sample(aud_file.audio_waveform_32bit, j), i)
                save_clip_stereo(destroyed_waveforms[(i, j)], fn, 4, 44100//j, aud_file.num_frames//j)
    return list_file_names

def run_experiments(num_exp, list_file_names, file_name, debug=False):
    input("Thank you for participating in this audio quality study. In this experiment, you will be asked to compare audio recording " \
          "qualities of the same recording. You will listen to two recordings, and then be asked which one is the 'better' recording. " \
          "This is a completely subjective answer, and you should go with your gut response. To prime your ears, here is the highest " \
          "quality recording, followed by the lowest quality. The pair of recordings will play twice. Press enter to listen to the recording.")
    compare_wavs(list_file_names[0], list_file_names[-1])
    compare_wavs(list_file_names[0], list_file_names[-1])
    num_wavs = len(list_file_names)
    scores = [0] * num_wavs
    name = input("When comparing recordings, you will also be given the option to rate the two as equal. In such a circumstance, you will " \
                 "then be asked if the pair of recordings would be a high quality or low quality recording. You will be able to repeat " \
                 "the same pair of recordings as many times as you would wish. To begin, input your name: ")
    score_file_name = "data/{}_{}_{}.csv".format(name, file_name, get_datetime())

    for x in range(num_exp):
        seed1 = random.randint(0,num_wavs-1)
        seed2 = random.randint(0,num_wavs-1)
        compare_wavs(list_file_names[seed1],
                     list_file_names[seed2])
        print("[{}/{}]".format(x, num_exp))
        i = 0
        while (i not in [1, 2, 3]):
            try:
                clear_output(wait=True)
                i = int(input("Which Recording Sounded Better?\n[1] First\n[2] Second\n[3] Same\n[4] Listen Again\n"))
                if (i == 4):
                    compare_wavs(list_file_names[seed1],
                                 list_file_names[seed2])
            except ValueError:
                i = 0
                print("Invalid Input. Choices are 1, 2, 3, or 4.")
        if (seed1 == seed2):
            pass
        elif (i == 1):
            scores[seed1] += 1
        elif (i == 2):
            scores[seed2] ++ 1
        elif (i == 3):
            h = 0
            while (h not in [1, 2]):
                try:
                    h = int(input("Are they low or high quality?\n[1] Low\n[2] High\n"))
                except ValueError:
                    h = 0
                    print("Invalid Input. Choices are 1 or 2.")
            scores[seed1] += (h-1)
            scores[seed2] += (h-1)

        if (debug):
            print("File 1: {}\nFile 2: {}".format(list_file_names[seed1], list_file_names[seed2]))
            time.sleep(3)
        clear_output(wait=True)

    with open(score_file_name, 'w') as fp:
        for i in range(num_wavs):
            fp.write("{},{}\n".format(list_file_names[i], scores[i]))
            print("File: {} Score: {}".format(list_file_names[i], scores[i]))
