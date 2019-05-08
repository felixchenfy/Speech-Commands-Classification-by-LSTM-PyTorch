#!/usr/bin/env python3


import time
from pynput import keyboard
from multiprocessing import Process, Value
import subprocess


if 1: # for AudioRecorder
    import sounddevice as sd
    import soundfile as sf
    import numpy as np  
    import argparse, tempfile, queue, sys, datetime

class TimerPrinter(object):
    # Print a message with a time gap of "T_gap"
    def __init__(self):
        self.prev_time = 0

    def print(self, s, T_gap):
        curr_time = time.time()
        if curr_time - self.prev_time < T_gap:
            return
        else:
            self.prev_time = curr_time
            print(s)


class AudioRecorder(object):

    def __init__(self, f_check_state=None):
        # f_check_state: if this function returns false, the recording state will stop
        
        self.init_settings()
        
        if f_check_state is None:
            self.f_check_state = lambda: True
        else:
            self.f_check_state = f_check_state
    
    def init_settings(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.parser.add_argument(
            '-l', '--list-devices', action='store_true',
            help='show list of audio devices and exit')
        self.parser.add_argument(
            '-d', '--device', type=self.int_or_str,
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
    

    def record(self):
        try:
            filename = tempfile.mktemp(
                prefix='audio_' + self.get_time(),
                suffix='.wav',
                dir='')
            
            q = queue.Queue()
            
            def callback(indata, frames, time, status):
                """This is called (from a separate thread) for each audio block."""
                if status:
                    print(status, file=sys.stderr)
                new_val = indata.copy()
                # print(new_val)
                q.put(new_val)

            # Make sure the file is opened before recording anything:
            t0 = time.time()
            with sf.SoundFile(filename, mode='x', samplerate=self.args.samplerate,
                            channels=self.args.channels, subtype=self.args.subtype) as file:
                with sd.InputStream(samplerate=self.args.samplerate, device=self.args.device,
                                    channels=self.args.channels, callback=callback):
                    print('#' * 80)
                    print('Start recording:')
                    print('#' * 80)
                    while True and self.f_check_state():
                        file.write(q.get())

            # Delete file if it's too short
            MIN_AUDIO_LENGTH = 2 # seconds
            t_duration = time.time() - t0 
            if t_duration < MIN_AUDIO_LENGTH:
                self.delete_file(filename)
                print("\nAudio is too short. It's been deleted.")
            else:
                print('\nRecording finished: ' + filename)

        except KeyboardInterrupt:
            print("'Ctrl+C' is pressed")
            # print('\nRecording finished: ' + repr(filename))
            # self.parser.exit(0)
        except Exception as e:
            print(e)
            # self.parser.exit(type(e).__name__ + ': ' + str(e))


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
    
    def run(self): # Collect events until released
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
            self.recording_state.value = True
            
    def on_release(self, key):
        if key.upper() == 'R':
            self.recording_state.value = False 


def thread1_keyboard_monitor(recording_state):
    km = KeyboardMonitor(recording_state, PRINT=False)
    km.run()
    print("Thread1 ends.")

def thread2_audio_recorder(recording_state):

    # Set up audio recorder
    def f_check_state():
        return recording_state.value 
    recorder = AudioRecorder(f_check_state=f_check_state)

    # Some variables and funcs
    tprinter1, tprinter2 = TimerPrinter(), TimerPrinter()
    prev_state, curr_state = recording_state.value, recording_state.value
    is_start_record = lambda: (not prev_state) and curr_state
    is_stop_record = lambda: not curr_state

    # Start listening recording_state to control the recording process
    while True:
        curr_state = recording_state.value

        if is_start_record():
            print("\n")
            recorder.record()
            print("\n"*2 + "/"*80 + "\n" + "Recording completes\n" + "/"*80 + "\n")

        elif is_stop_record():
            tprinter2.print("Usage: Keep pressing 'R' to record audio.", T_gap=3)
        
        prev_state = curr_state
        time.sleep(0.01)

    print("Thread2 ends.")


if __name__ == '__main__':
    recording_state = Value('i', 0)

    # Set keyboard monitor
    thread1 = Process(
        target=thread1_keyboard_monitor,
        args=(recording_state, ))

    # Set audio recorder
    thread2 = Process(
        target=thread2_audio_recorder,
        args=(recording_state, ))

    # Start
    thread1.start()
    thread2.start()

    # Wait
    thread1.join()
    thread2.join()
