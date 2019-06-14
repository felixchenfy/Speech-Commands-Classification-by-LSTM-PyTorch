# Speech Commands Classification
Abstract: Classification of 11 types of audio clips using MFCCs features and LSTM. Pretrained on Speech Command Dataset with intensive data augmentation.   

Final project of EECS 475 - Machine Learning  
student name: **Feiyu Chen**  
time: 2019, June 15th

Report file: [Report.ipynb](Report.ipynb)  

Video file: Report_Video_Demo.mp4  
Video link: https://youtu.be/6Kpfc7uD26w  

# 1. Introduction

The goal of this project is to implement an audio classification system, which: 
1. first reads in an audio clip (containing at most one word),
2. and then recognizes the class(label) of this audio.


### Classes  
11 classes are chosen, namely:   
> one, two, three, four, five, front, back, left, right, stop, none

where "one" means the audio contains the word "one", etc. The only exception is that "none" means the audio contains no word.

### Method  

Features: MFCCs (Mel-frequency cepstral coefficients) are computed from the raw audio. You can think of it as the result of fouriour transformation.

Classifier: LSTM (Long Short-Term Memory) is adopted for classificatioin, which is a type of Recurrent Neural Network.

The model was pretrained on the [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) with intensive data augmentation, including "shift", "amplify", "superpose noise", etc.

The model was then finetuned on my own dataset.
    
### Result  

The test accuracy is 92.4% on Speech Commands Dataset, with a random 0.9/0.1 train/test split.

The model was was then evaluated on my own dataset of 1378 samples.  
The test accuracy is 100.0%, with a random 0.7/0.3 train/test split.  

For more detailed introduction of how I trained this audio classifier, please see my report: [Report.ipynb](Report.ipynb)   

# 3. Datset

See [data/README.md](data/README.md)

# 4. Install dependencies

This project is written in Python 3.7. The main depencencies are listed below.

* Pytorch   
    I use the newest version in 2019/06, which is 1.1.0:
    > $ pip install torch torchvision


* Keyboard io:  
    > $ pip install pynput  

* Audio IO:  
    https://github.com/bastibe/SoundFile  
    > $ pip install soundfile  
    > $ pip install sounddevice  
    > $ sudo apt-get install libsndfile1  

* Extract audio MFCC feature:  
    https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html  
    > $ pip install librosa  

* Synthesize audio (good):  
    > $ pip install gtts  

* Synthesize audio (bad):  
    > $ pip install pyttsx  
    > $ pip install pyttsx3  
    > $ sudo apt-get install espeak  

* Others  
    > pip install matplotlib sklearn scipy

    There may be some other common libraries. You may just install them by pip.


# 5. Main commands

## 5.1 Collect audio from microphone

(tested on Ubuntu 18.04)

> $ source src/s0_record_audio.sh  

## 5.2 Train

> $ python src/s1_pretrain_on_kaggle.py  

Copy one of the weight file ".ckpt" from "checkpoints/" to "good_weights/kaggle.ckpt", and then:

> $ python src/s2_train_on_mine.py  

## 5.3 Test

* Inference from microphone:
    1. Run the main program:  
        > $ python src/s3_inference_microphone.py
    2. Press "r" on your keyboard to start recording the audio.
    3. Say a word among the above 10 words, or say nothing.
    4. Release the key "r" to stop recording.
    5. The program will speak out the recognition result.

* Inference an audio file:  
    For example:
    > $ python src/s4_inference_audio_file.py --path_to_data test_data/audio_front.wav

* Inference a folder containing audio files:  
    For example:
    > $ python src/s4_inference_audio_file.py --path_to_data data/data_train/three/