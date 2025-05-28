# --- Standard Library ---
from dataclasses import dataclass, field
from pathlib import Path
import os 

# --- Third-Party Libraries ---
import numpy as np 
import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
import matplotlib.pyplot as plt
import IPython.display as ipd

# --- Local Application Modules ---
from core.dataset import AudioDataset, MultiAudioDataset
from core.model import AudioCNN, MultiAudioCNN
from core.utils import process_audio_file

# --- Default Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataclass: Configuration for Detection ---
@dataclass
class DetectionConfig:
    test_folder: str = ""
    # test_datasets: list = field(default_factory=list)
    class_names: list = field(default_factory=list) 
    models_dir: str = "models/"
    models: list = field(default_factory=list)
    old_model: bool = False
    save_plot: bool = False
    save_name: str = ""


# --- Split Long Audio into Strides ---   
def stride_splitting(path, sample_rate, stride_samples):
    waveform, sr = torchaudio.load(path)

    # Resample if sampling rates don't match
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    # Convert to mono if multi-channel
    if waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Create list of 1-second stride waveforms
    total_samples = waveform.shape[1]
    num_strides = total_samples // stride_samples

    waveform_strides = [
        waveform[:, i * stride_samples : (i + 1) * stride_samples]
        for i in range(num_strides)
    ]
    return waveform, waveform_strides


# --- Plot Predictions and Audio Waveform for Long Audio ---
def plot_predictions_on_long_audio(filename, file_path, class_names, model, transform, save_plot=False, save_name=None):
    sample_rate = 16000    # Hz
    stride_duration = 1.0  # seconds
    stride_samples = int(stride_duration * sample_rate)

    waveform, waveform_strides = stride_splitting(file_path, sample_rate, stride_samples)

    # Assign colors for plotting different classes
    color_options = ["red", "blue", "green", "black", "orange", "purple", "pink", "brown", "gray"]
    class_colors = color_options[:len(class_names)]

    # Lists for predictions and confidence scores
    predictions = []
    confidences = []
    probabilities = []
    concat_audio  = []

    for waveform_stride in waveform_strides:
        # pad if stride is too short
        if waveform_stride.shape[1] < stride_samples:
            waveform_stride = nn.functional.pad(waveform_stride, (0, stride_samples - waveform_stride.shape[1]))
        
        # Feature Extraction and Prediction
        features = transform(waveform_stride).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            conf, class_idx = torch.max(probs, dim=1)
            
            predictions.append(class_idx.item())
            confidences.append(conf.item())
            probabilities.append(probs)
            concat_audio.append(waveform_stride.squeeze().numpy())


    # --- Plot Predicted Class Confidence Over Time ---
    plt.figure(figsize=(12, 5))
    time_axis = [i * stride_duration for i in range(len(predictions))]
    
    for class_idx, class_name in enumerate(class_names):
        mask = [index for index, prediction in enumerate(predictions) if prediction == class_idx]
        plt.plot(
            [time_axis[m] for m in mask],
            [confidences[m] for m in mask],
            color=class_colors[class_idx],
            label=class_name,
            linestyle="-",
            marker="o",
            markersize=8
        )
    # Confidence Threshold Line
    plt.axhline(y=0.5, color="red", linestyle="--", label="Confidence Threshold 0.5")
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence Of Predicted Class")
    plt.title(f"Predicted Classes Over Time: \"{filename}\"")
    plt.grid(True)
    plt.show()

    # --- Plot Original Audio Waveform ---
    concat_audio_np = np.array(concat_audio) # Efficient conversion          
    concat_audio_np_flat = torch.from_numpy(concat_audio_np).flatten().numpy() 
    
    audio_time = [i / sample_rate for i in range(len(concat_audio_np_flat))]

    plt.figure(figsize=(12, 3))
    plt.plot(audio_time, concat_audio_np_flat)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform of {filename}")
    plt.grid()
    plt.show()

    # --- Play the Original Audio ---
    print(f"Playing full original audio: {filename}")
    ipd.display(ipd.Audio(waveform.squeeze().numpy(), rate=sample_rate))
    

# --- Run Long Audio Detection with Multiple Models ---
def long_audio_detection(
    test_folder, 
    class_names, 
    models_dir, 
    models, 
    old_model, 
    save_plot, 
    save_name
    ):
    # --- Transform ---
    melspec_transform = nn.Sequential(                 # preprocessing pipeline
        MelSpectrogram(sample_rate=16000, n_mels=64),  # convert waveform to mel spectrogram
        AmplitudeToDB()                                # convert amplitude to decibels
    )
     
    test_folder = Path(test_folder)
    models_dir = Path(models_dir)

    # --- Iterate Over All Provided Models ---
    for model_name in models:
        model_path = models_dir / model_name

        # Load Correct Model Architecture
        if old_model:
            model = AudioCNN(num_classes=1).to(device)
        else:
            model = MultiAudioCNN(num_classes=len(class_names)).to(device)

        # Load Model Weights and Evaluate Model
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        print(f"\nTesting model: {model_name}")

        # --- Process Each Audio File in the Test Folder ---
        for filename in os.listdir(test_folder):
            print(f"\nProcessing file: {filename}")

            file_path = Path(test_folder, filename)
            plot_predictions_on_long_audio(
                filename, 
                file_path, 
                class_names, 
                model, 
                melspec_transform, 
                save_plot, 
                save_name
            )