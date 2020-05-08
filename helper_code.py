import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def print_bytes(byte_array):
    p = "0x"
    for b in byte_array:
        x = str(hex(b))[2:]
        if (len(x) < 2):
            x = "0" + x
        p += x
    return p

def int32_to_frame(left, right):
    return (right << 32) + (left)
"""
    x = ((left  & (0xFF << 24)) <<  8).astype(np.uint64) + \
           ((left  & (0xFF << 16)) << 24).astype(np.uint64) + \
           ((left  & (0xFF <<  8)) << 40).astype(np.uint64) + \
           ((left  & (0xFF <<  0)) << 56).astype(np.uint64) + \
           ((right & (0xFF << 24)) >> 24).astype(np.uint64) + \
           ((right & (0xFF << 16)) >>  8).astype(np.uint64) + \
           ((right & (0xFF <<  8)) <<  8).astype(np.uint64) + \
           ((right & (0xFF <<  0)) << 24).astype(np.uint64)
    print(hex(x))
    return x
"""

class audio_file(object):
    def __init__(self, file_name):
        self.file_name    = file_name
        self.audio_sample = wave.open(self.file_name,'r')
        self.num_channels = self.audio_sample.getnchannels()
        self.byte_depth   = self.audio_sample.getsampwidth()
        self.sample_freq  = self.audio_sample.getframerate()
        self.num_frames   = self.audio_sample.getnframes()
        self.frame_length = self.num_channels * self.byte_depth

    def load_data(self):
        self.audio_waveform_32bit = np.zeros(shape=(self.num_channels,self.num_frames), dtype=np.int32)
        self.audio_frames         = []
        print("Generating 32 bit up scaled waveform...")
        up_sampled_bits = 8 * (4 - self.byte_depth)
        for j in range(self.num_frames):
            waveform_sample = self.audio_sample.readframes(1)
            self.audio_frames += [waveform_sample]
            for i in range(self.num_channels):
                self.audio_waveform_32bit[i][j] = int.from_bytes(waveform_sample[i * self.byte_depth: (i + 1) * self.byte_depth], "little", signed=True) << up_sampled_bits
        
    def info_dump(self):
        print("File Name:     ", self.file_name)
        print("Channels:      ", self.num_channels)
        print("Byte Depth:    ", self.byte_depth)
        print("Sampling rate: ", self.sample_freq)
        print("Frames:        ", self.num_frames)
        
    def graph_waveform(self, start_sample=0, end_sample=500):
        plt.figure(figsize=(20,8))
        plt.subplot(2, 1, 1)
        plt.plot(self.audio_waveform_32bit[0][start_sample:end_sample], 'r')
        plt.title("Left")
        plt.subplot(2, 1, 2)
        plt.plot(self.audio_waveform_32bit[1][start_sample:end_sample], 'b')
        plt.title("Right")
        plt.show()
        
    def save_clip(self, file_name, waveform=None):
        """Saves a signal as a 32 bit PCM WAV file
        """
        if (waveform is None):
            waveform = self.audio_waveform_32bit
        with wave.open(file_name, 'wb') as fp:
            fp.setnchannels(self.num_channels)
            fp.setsampwidth(4)
            fp.setframerate(self.sample_freq)
            fp.setnframes(self.num_frames)
            for i in range(self.num_frames):
                left  = waveform[0][i]
                right = waveform[1][i]
                fp.writeframes(int32_to_frame(left, right))
                
    def down_sample(self, samp_rate_ratio, file_name):
        waveform = self.audio_waveform_32bit
        with wave.open(file_name, 'wb') as fp:
            fp.setnchannels(self.num_channels)
            fp.setsampwidth(4)
            fp.setframerate(self.sample_freq // samp_rate_ratio)
            fp.setnframes(self.num_frames // samp_rate_ratio)
            for i in range(self.num_frames):
                if (i % samp_rate_ratio == 0):
                    left  = waveform[0][i]
                    right = waveform[1][i]
                    fp.writeframes(int32_to_frame(left, right))

    def up_sample(self, samp_rate_ratio, file_name):
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