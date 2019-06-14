#!/usr/bin/env python3


import time
from pynput import keyboard
from multiprocessing import Process, Value
import subprocess
import librosa
import os 

if 1: # for AudioRecorder
    import sounddevice as sd
    import soundfile as sf
    import numpy as np  
    import argparse, tempfile, queue, sys, datetime

def reset_audio_sample_rate(filename, dst_sample_rate):
    # dst_sample_rate = 16000, see "def stop_record"
    data, sample_rate = sf.read(filename) 
    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate
    sf.write(filename, data, sample_rate)
    # print(f"Reset sample rate to {dst_sample_rate} for the file: {filename}")
    
class TimerPrinter(object):
    # Print a message with a time gap of "T_gap"
    def __init__(self):
        self.prev_time = -999

    def print(self, s, T_gap):
        curr_time = time.time()
        if curr_time - self.prev_time < T_gap:
            return
        else:
            self.prev_time = curr_time
            print(s)
    
    def reset(self):
        self.prev_time = -999


class AudioRecorder(object):

    def __init__(self):
        self.init_settings()
    
    def init_settings(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.parser.add_argument(
            '-l', '--list-devices', action='store_true',
            help='show list of audio devices and exit')
        self.parser.add_argument(
            '-d', '--device', type=self.int_or_str, default='0',
            help='input device (numeric ID or substring)')
        self.parser.add_argument(
            '-r', '--samplerate', type=int, help='sampling rate')
        self.parser.add_argument(
            '-c', '--channels', type=int, default=1, help='number of input channels')
        self.parser.add_argument(
            'filename', nargs='?', metavar='FILENAME',
            help='audio file to store recording to')
        self.parser.add_argument(
            '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')

        self.args = self.parser.parse_args()

        if self.args.list_devices:
            print(sd.query_devices())
            self.parser.exit(0)
        if self.args.samplerate is None:
            device_info = sd.query_devices(self.args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            self.args.samplerate = int(device_info['default_samplerate'])

    def start_record(self, folder='./'):
        
        # Some settings
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        self.filename = tempfile.mktemp(
            prefix=folder + 'audio_' + self.get_time(),
            suffix='.wav',
            dir='')
        self.audio_time0 = time.time()
        
        # Start
        # self._thread_alive = True # This seems not working
        self.thread_record = Process(
            target=self.record,
            args=())
        self.thread_record.start()

    def stop_record(self, sample_rate=16000):
        '''
        Input:
            sample_rate: desired sample rate. The original audio's sample rate is determined by
                the hardware configuration. Here, to achieve the desired sample rate, 
                this script will read the saved audio from file, resample it, 
                and then save it back to file.
        '''
        
        # Stop thread
        self.thread_record.terminate()
        # self._thread_alive = False # This seems not working
        
        if 0: # Print dashed lines
            print("\n\n" + "/"*80)
            print("Complete writing audio to file:", self.filename)
            print("/"*80 + "\n")

        # Check result
        time_duration = time.time() - self.audio_time0
        self.check_audio(time_duration)
        reset_audio_sample_rate(self.filename, sample_rate)

    def record(self):

        q = queue.Queue()
        
        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            new_val = indata.copy()
            # print(new_val)
            q.put(new_val)

        with sf.SoundFile(self.filename, mode='x', samplerate=self.args.samplerate,
                        channels=self.args.channels, subtype=self.args.subtype) as file:
            with sd.InputStream(samplerate=self.args.samplerate, device=self.args.device,
                                channels=self.args.channels, callback=callback):
                print('#' * 80)
                print('Start recording:')
                print('#' * 80)
                # while True and self._thread_alive:
                while True:
                    file.write(q.get())
            
    def check_audio(self, time_duration, MIN_AUDIO_LENGTH=0.1):
        # Delete file if it's too short
        print("\n")
        if time_duration < MIN_AUDIO_LENGTH:
            self.delete_file(self.filename)
            print("Audio is too short. It's been deleted.")
        else:
            print('Recorded audio is saved to: ' + self.filename)
        print("-"*80  + "\n\n")


    def delete_file(self, filename):
        subprocess.check_call("rm " + filename, shell=True)

    def int_or_str(self, text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    def get_time(self):
        s=str(datetime.datetime.now())[5:].replace(' ','-').replace(":",'-').replace('.','-')[:-3]
        return s # day, hour, seconds: 02-26-15-51-12-556

class KeyboardMonitor(object):
    # https://pypi.org/project/pynput/1.0.4/
    def __init__(self, recording_state, PRINT=False):
        self.recording_state = recording_state
        self.PRINT = PRINT 
        self.thread = None 
        self.prev_state, self.curr_state = False, False

    def get_key_state(self):
        ss = (self.prev_state, self.curr_state)
        return ss 

    def update_key_state(self):
        self.prev_state = self.curr_state
        self.curr_state = self.recording_state.value
        # print("update_key-state", self.get_key_state())

        # print(self.get_key_state())

    def start_listen(self, run_in_new_thread=False): # Collect events until released
        if run_in_new_thread:
            self.thread = Process(
                target=self._start_listen,
                args=())
            self.thread.start()
        else:
            self._start_listen()

    def stop_listen(self):
        if self.thread:
            self.thread.terminate()

    def _start_listen(self):
        with keyboard.Listener(
                on_press=self.callback_on_press,
                on_release=self.callback_on_release) as listener:
            listener.join()
     
    def key2char(self, key):
        try:
            return key.char
        except:
            return str(key)

    def callback_on_press(self, key):
        key = self.key2char(key)
        if self.PRINT:
            print("\nKey {} is pressed".format(key))
        self.on_press(key)

    def callback_on_release(self, key):
        key = self.key2char(key)
        if self.PRINT: 
            print("\nKey {} is released".format(key))
        self.on_release(key)

    def on_press(self, key):
        if key.upper() == 'R':
            self.recording_state.value = 1
        
    def on_release(self, key):
        if key.upper() == 'R':
            self.recording_state.value = 0 

    def is_kept_pressed(self):
        return (self.prev_state, self.curr_state) == (True, True)

    def has_just_pressed(self):
        return (self.prev_state, self.curr_state) == (False, True)

    def has_just_released(self):
        return (self.prev_state, self.curr_state) == (True, False)

    def is_released(self):
        return not self.curr_state
       

if __name__ == '__main__':
    
    # Start keyboard listener
    recording_state = Value('i', 0)
    board = KeyboardMonitor(recording_state, PRINT=False)
    board.start_listen(run_in_new_thread=True)

    # Set up audio recorder
    recorder = AudioRecorder()

    # Others
    tprinter = TimerPrinter() # for print

    # Start loop
    while True:
        tprinter.print("Usage: keep pressing down 'R' to record audio", T_gap=2)

        board.update_key_state()
        if board.has_just_pressed():

            # start recording
            recorder.start_record(folder='./data/data_tmp/') 

            # wait until key release
            while not board.has_just_released():
                board.update_key_state()
                time.sleep(0.001)

            # stop recording
            recorder.stop_record()
            
        time.sleep(0.05)