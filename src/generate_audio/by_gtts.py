from gtts import gTTS
import subprocess
import soundfile as sf
import librosa
lang = ['en', 'en-uk', 'en-au', 'en-in'][2]
tts = gTTS(text='Result is', lang=lang)
filename = "result_is.wav"
tts.save(filename)


# Load and change speed
# data, sample_rate = sf.read(filename)
data, sample_rate = librosa.load(filename)

def resample(data, sample_rate, new_sample_rate):
    data = librosa.core.resample(data, sample_rate, new_sample_rate)
    return data, new_sample_rate
data, sample_rate = resample(data, sample_rate, 48000)
data = librosa.effects.time_stretch(data, rate=1.5)
librosa.output.write_wav(filename, data, sample_rate)


# Play
subprocess.call(["cvlc", "--play-and-exit", filename])
