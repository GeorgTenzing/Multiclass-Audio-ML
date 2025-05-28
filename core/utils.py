import os
from pathlib import Path
import time 

import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


# --- Audio Processing ---
def process_audio_file(path, sample_rate, transform, duration_sec = 1):
    """
    Loads and processes a .wav or .mp3 file:
    - Resamples if needed
    - Crops or pads to 1 second
    - Applies mel-spectrogram and amplitude-to-dB transforms
    """
    path = Path(path)
    waveform, sr = torchaudio.load(path)     # load waveform and sample rate

    if sr != sample_rate:                    # resample if sample rate doesn't match
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # --- Crop or pad to fixed duration ---
    target_samples = int(duration_sec * sample_rate)          # number of samples we want (e.g., 5 sec * 16000 = 80000)
    current_samples = waveform.shape[1]                       # actual number of samples in the loaded waveform

    if current_samples < target_samples:                      # if the waveform is too short
        pad = target_samples - current_samples                # calculate how many samples to pad
        waveform = torch.nn.functional.pad(waveform, (0, pad))# pad at the end with zeros (right-side padding)
    else:
        waveform = waveform[:, :target_samples]               # crop waveform to the target length

    features = transform(waveform)            # apply MelSpectrogram + AmplitudeToDB

    if features.shape[0] != 1:                # ensure features have 1 channel
        features = features.mean(dim=0, keepdim=True)

    return features

# --- Transform ---
def melspec_transform():
       
    return nn.Sequential(                       # preprocessing pipeline
        MelSpectrogram(sample_rate=16000, n_mels=64),  # convert waveform to mel spectrogram
        AmplitudeToDB()                                      # convert amplitude to decibels
    )


# --- 
def concat_wav_files(input_dir, output_path, check_folders, pattern, sample_rate = 16000):

    files = []
    if check_folders:
        folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
        for folder in folders:
            folder_path = os.path.join(input_dir, folder)
            for file in sorted(os.listdir(folder_path)):
                files.append(os.path.join(folder_path, file))
    else:
        files = sorted(os.listdir(input_dir)) 

    audio_data = []
    for file in files:
        if file.endswith(".wav") and (pattern is None or file.startswith(pattern)):
            file_path = os.path.join(input_dir, file)
            waveform, sr = torchaudio.load(file_path)
            if sr != sample_rate:                    # resample if sample rate doesn't match
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] != 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            audio_data.append(waveform)
            print(f"file: {file}, filepath: {file_path}")

    if len(audio_data) == 0:
        print(f"No valid .wav files found in {input_dir} with pattern '{pattern}'")
        return

    concatenated_audio = torch.cat(audio_data, dim=1)  

    output_dir  = os.path.dirname(output_path)
    file_name   = os.path.basename(output_path)
    output_path = os.path.join(output_dir, file_name)

    torchaudio.save(output_path, concatenated_audio, sample_rate=sample_rate)
    print(f"Concatenated {len(files)} files → {file_name}")
    print(f"Output directory: {output_dir}")



def split_wav_files(input_path, output_dir, pattern, num_chunks=10):

    waveform, sr = torchaudio.load(input_path)

    total_samples = waveform.shape[1]
    chunk_size = total_samples // num_chunks

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else total_samples
        chunk = waveform[:, start:end]

        # output_path = os.path.join(output_dir, f"forest_split_{i+1:02d}.wav")
        output_path = os.path.join(output_dir, pattern + f"{i+1:02d}.wav")
        torchaudio.save(output_path, chunk, sample_rate=sr)
        print(f"Saved: {output_path} ({(end - start) / sr:.2f} sec)")
