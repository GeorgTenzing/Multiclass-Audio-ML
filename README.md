# Multiclass-Audio-ML
This is a personal project for the ETH Zürich Course: P&S: Python for machine learning
Datasets will be uploaded in the future 


## Overview:
The Notebook main.ipynb allows users to train and test ml models for binary as well as multi-class audio files.
It acts as a wrapper to the actual code, which is hidden under the 'core' folder. 

Models are stored as and loaded from the 'models' folder as .pth files.
To support older, already trained, models, I provide extra code within the python files 
inside the 'core' folde.

Audio files are stored locally, in my case under the folder 'drone_audio_datasets'.
Only a small sample is provided for showcasing. Please refer to the source section for the datasets.

dataset.py: stores the class MultiAudioDataset, which labels the .wav audio files and stores their paths
model.py:   stores the class MultiAudioCNN, which is a convolutional neural network for binary audio classification.
utils.py:   stores usefull functions like process_audio_file used for extracting features from audio files, 
as well as concat_wav_files and split_wav_files for manipulating audio data. 

train.py: is the heart of the model training. It contains the class TrainConfig for ease of use when using 
the main function train_audio_detection_model, which trains an audio classification model, 
using CNN on mel-spectrogram data, as well as train_one_epoch which makes the code more readable.

test.py: contains the class TestConfig similarly to TrainConfig and the function test_multiple_models 
which acts as a wrapper for test_multiclass_dataset, which evaluates models  and prints prediction statistics. 

For legacy models use: core.dataset - AudioCNN
                       core.model   - AudioDataset
                       core.test    - test_binary_datasets
📁 core/

 ┣ dataset.py         # Dataset loaders

 ┣ model.py           # CNN architectures

 ┣ utils.py           # Audio utilities

 ┣ long_detection.py  # Long audio detection logic

 ┣ test.py            # Model evaluation

 ┗ train.py           # Training functions

📁 datasets/          # Training & testing audio data  

📁 models/            # Saved model weights  

main.ipynb           # Jupyter notebook for quick experimentation  

## Requirements 
os 
time
pathlib
typing
dataclasses

numpy
matplotlib
IPython
torch
torchaudio

## Datasets References: 
This project uses several drone audio datasets, including the following:

1. **Real World Object Detection Dataset for Quadcopter Unmanned Aerial Vehicle Detection**  
   M. Ł. Pawełczyk and M. Wojtyra  
   IEEE Access, vol. 8, pp. 174394-174409, 2020  
   DOI: [10.1109/ACCESS.2020.3026192](https://doi.org/10.1109/ACCESS.2020.3026192)

2. **Audio Based Drone Detection and Identification using Deep Learning**  
   Sara A Al-Emadi, Abdulla K Al-Ali, Abdulaziz Al-Ali, Amr Mohamed  
   IWCMC 2019 Vehicular Symposium (IWCMC-VehicularCom 2019), Tangier, Morocco, June 23, 2019

3. **DroneNoise Database**  
   Authors: Carlos Ramos-Romero, Nathan Green, César Asensio, Antonio J Torija Martinez  
   - Collected during sUAS overflight operations in Edzell, Scotland (Aug 17, 2022)
   - DOI: [https://doi.org/10.17866/rd.salford.22133411.v3](https://doi.org/10.17866/rd.salford.22133411.v3)  
   - Publisher URL: [https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411](https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411)
