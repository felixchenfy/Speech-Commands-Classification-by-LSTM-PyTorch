
This project is written in Python 3.7 on Ubuntu 18.04. 

The main depencencies are listed below.  

* All python things
    > conda create -n test_speech python=3.7  
    > conda activate test_speech  
    > pip install matplotlib sklearn scipy numpy opencv-python jupyter
    > pip install pynput soundfile sounddevice librosa gtts pyttsx pyttsx3  

    Finally, install torch. Please go to "https://pytorch.org/" and install the one that matches with your computer.

    I'm using Stable(1,1), Linux, Pip, Python 3.7, CUDA 10.0, and torch-1.1.0, torchvision-0.3.0

    > pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl  
    > pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl  

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