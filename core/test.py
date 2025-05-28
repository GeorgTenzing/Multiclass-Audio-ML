# --- Standard Library ---
import time
from dataclasses import dataclass, field
from typing import List, Callable, Optional
from pathlib import Path

# --- Third-Party Libraries ---
import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

# --- Local Application Modules ---
from core.model import AudioCNN, MultiAudioCNN
from core.utils import process_audio_file

@dataclass
class TestConfig:
    test_folder: str = ""
    test_datasets: list = field(default_factory=list)
    class_names: list = field(default_factory=list) 
    model_folder: str = "models/"
    models: list = field(default_factory=list)
    output_false_pred: bool = False
    old_model: bool = False


# --- Helper Function: Get Device ---
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)} \n")
    else:
        print(f"Using CPU \n")
    return device



def test_binary_datasets(class_names, folder_path, model, transform, output_false_pred, device):
    """
    Evaluates a model on a binary dataset and prints prediction statistics.
    """
    
    positive = class_names[0]
    negative = class_names[1]

    true_positives  = 0
    true_negatives  = 0
    false_positives = 0
    false_negatives = 0

    list_of_false_positive_files = []
    list_of_false_negative_files = []

    tic = time.perf_counter()
    model.eval()
    
    print("\n--- Test Predictions ---")
    positive_dir = folder_path / positive
    negative_dir = folder_path / negative
    
    assert positive_dir.exists(), f"{positive_dir} does not exist"
    assert negative_dir.exists(), f"{negative_dir} does not exist"

    for folder in [positive_dir, negative_dir]:
        for path in folder.glob("*.wav"):
            features = process_audio_file(path, 16000, transform)
            features = features.unsqueeze(0).to(device)  # shape [1, 1, 64, time]

            with torch.no_grad():
                output = model(features)
                prediction = positive if output.item() > 0.5 else negative

            filename = path.name

            if prediction == negative and folder == negative_dir:
                true_negatives += 1
            elif prediction == positive and folder == negative_dir:
                false_positives += 1
                list_of_false_positive_files.append(filename)
            elif prediction == positive and folder == positive_dir:
                true_positives += 1
            elif prediction == negative and folder == positive_dir:
                false_negatives += 1
                list_of_false_negative_files.append(filename)

    toc = time.perf_counter()    

    total    = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total * 100
    
    print(f"{folder_path} completed in {toc - tic:.2f} seconds")
    print(f"False Positive Predictions: {false_positives} / {true_positives } = {false_positives / total * 100:.2f}%")
    print(f"False Negative Predictions: {false_negatives} / {true_negatives} = {false_negatives / total * 100:.2f}%")
    print(f"Out of {total} samples, accuracy is {accuracy:.2f}%")
    
    if output_false_pred:
        for false_positives in list_of_false_positive_files: 
            print(f"False Positive: {filename}, {output.item():.2f}")
        for false_negatives in list_of_false_negative_files: 
            print(f"False Negative: {filename}, {output.item():.2f}")

    return accuracy






def test_multiclass_dataset(data_set, class_names, model, transform, output_false_pred, device, extensions=("*.wav", "*.mp3")):
    correct = 0
    total = 0

    result = {cls: {"correct": 0, "incorrect": 0, "files": []} for cls in class_names}

    tic = time.perf_counter()

    for class_index, class_name in enumerate(class_names):
        class_dir = Path(data_set, class_name)
        assert class_dir.exists(), f"{class_dir} does not exist"   

        for extension in extensions:                       
            for file in class_dir.glob(extension):
                features = process_audio_file(file, 16000, transform)
                features = features.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(features)
                    _, predicted = torch.max(output, 1)

                    if predicted.item() == class_index:
                        correct += 1
                        result[class_name]["correct"] += 1
                    else:
                        result[class_name]["incorrect"] += 1
                        result[class_name]["files"].append(file)
                    total += 1

    toc = time.perf_counter()

    if total == 0:
        print(f"No samples found in {data_set}. Please check the dataset.")
        return 0.0
    
    print(f"Testing completed in {toc - tic:.2f} seconds\n")

    for cls in class_names:
        total_cls = result[cls]["correct"] + result[cls]["incorrect"]
        if total_cls == 0:
            print(f"No samples found in {data_set}. Please check the dataset.")
            continue
        acc_cls = (result[cls]["correct"] / total_cls * 100) 
        print(f"Class {cls:<10}: Accuracy = {acc_cls:.2f}% ({result[cls]['correct']} / {total_cls})")
        
        if output_false_pred and result[cls]["files"]:
            print(f"  Misclassified files: {result[cls]['files'][:3]} ...")
    
    accuracy = correct / total * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}% on {total} samples\n")

    return accuracy








# --- Helper Function: load and eval model ---
def load_and_eval_model(model_folder, model_name, class_names, device, old_model= False):
    model_path = Path(model_folder) / model_name

    if old_model:
        model = AudioCNN(num_classes=1).to(device)
    else:
        model = MultiAudioCNN(num_classes=len(class_names)).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model successfully loaded from: {model_path} \n")
        print(f"Testing model: {model_name} \n")
    except Exception as e:
        print(f"\nFailed to load model from {model_path}: {e}")
        return None
    
    model.eval()
    return model


def test_multiple_models(
    test_folder,
    test_datasets,
    class_names,
    model_folder,
    models,
    # overall_accuracy,
    output_false_pred,
    old_model
    ):
    """
    Tests multiple models across multiple datasets and reports accuracy.
    """
    # --- Transform ---
    melspec_transform = nn.Sequential(                       # preprocessing pipeline
        MelSpectrogram(sample_rate=16000, n_mels=64),        # convert waveform to mel spectrogram
        AmplitudeToDB()                                      # convert amplitude to decibels
    )

    # --- Helper Function: Get Device ---
    device = get_device()

    for model_name in models:

        model = load_and_eval_model(model_folder, model_name, class_names, device, old_model)
        if model is None:
            continue

        all_accuracies = []

        for test_dataset in test_datasets:
            data_set = Path(test_folder, test_dataset) 

            if old_model:
                accuracy = test_binary_datasets(class_names, data_set, model, melspec_transform, output_false_pred, device)
            else:
                accuracy = test_multiclass_dataset(data_set, class_names, model, melspec_transform, output_false_pred, device)
            
            all_accuracies.append(accuracy)

        # if overall_accuracy:
        #     avg = sum(all_accuracies) / len(all_accuracies)
        #     print(f"\nOverall Accuracy: {avg:.2f}%")
        




# --- FUTURE WORK: SAVE BEST MODEL ---
"""

    in_training: bool = False
    model_testing: Optional[Callable] = None

    in_training,
    model_testing
    
        if in_training:
            model = model_testing
        if not in_training:
            print(f"-----------------------------------------------------------------------------")
            print(f"\n Testing model: {model_name} \n")


        if not in_training:
            print(f"-----------------------------------------------------------------------------")

            
                if not in_training:
        print(f"{data_set} completed in {toc - tic:.2f} seconds")
        print(f"Overall Accuracy: {accuracy:.2f}% on {total} samples\n")

    for cls in class_names:
        total_cls = result[cls]["correct"] + result[cls]["incorrect"]
        acc_cls = (result[cls]["correct"] / total_cls * 100) if total_cls > 0 else 0.0
        if not in_training:
            print(f"Class '{cls}': Accuracy = {acc_cls:.2f}% ({result[cls]['correct']} / {total_cls})")
        if output_false_pred and result[cls]["files"]:
            if not in_training:
                print(f"  Misclassified files: {result[cls]['files'][:3]} ...")


"""