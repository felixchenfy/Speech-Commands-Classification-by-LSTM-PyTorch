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

The model was then finetuned and evaluated on my own dataset of 1378 samples, with all the parameters fixed except the last FC layer.  
The test accuracy is 100.0%, with a random 0.7/0.3 train/test split.   
This is kind of overfitting, because almost all the words(audio files) were spoken by me, which are similar to each other to some extent.

If you want to use this repo for your own course project, you may need to record audios of your own voice, and then finetune the model. 

For more detailed introduction of how I trained this audio classifier, please see my report: [Report.ipynb](Report.ipynb)   

# 3. Download Data

Please see instructions in "data_train" and "kaggle" in this file: [data/README.md](data/README.md). 

# 4. Install dependencies
See [doc/dependencies.md](doc/dependencies.md)

# 5. Main commands

## 5.1 Collect audio from microphone

> $ source src/s0_record_audio.sh  

Press keyboard key "R" to start recording. Release "R" to finish recording. The recorded audio is saved "data/data_tmp/".

## 5.2 Train

(Before training, you need to download the required [data](data/README.md) and put into the folders: data/data_train/, data/kaggle/)

Train on Speech Commands Dataset:

> $ python src/s1_pretrain_on_kaggle.py  

Copy one of the saved weight file ".ckpt" from "checkpoints/" to "good_weights/kaggle.ckpt", and then:

> $ python src/s2_train_on_mine.py  

## 5.3 Test

* Inference from microphone:
    1. Run the main program:  
        > $ python src/s3_inference_microphone.py
    2. Press "R" on your keyboard to start recording the audio.
    3. Say a word among the above 10 words, or say nothing.
    4. Release the key "R" to stop recording.
    5. The program will speak out the recognition result.

* Inference an audio file:  
    For example:
    > $ python src/s4_inference_audio_file.py --path_to_data test_data/audio_front.wav

* Inference a folder containing audio files:  
    For example:
    > $ python src/s4_inference_audio_file.py --path_to_data test_data/

# 6. Reference

* Record audio  
https://python-sounddevice.readthedocs.io/en/0.3.12/examples.html#recording-with-arbitrary-duration

* LSTM  
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py

* How to augment audio data  
https://www.kaggle.com/CVxTz/audio-data-augmentation

