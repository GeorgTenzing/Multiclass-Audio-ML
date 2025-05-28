# --- Standard Library ---
import time
from dataclasses import dataclass, field
from pathlib import Path

# --- Third-Party Libraries ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

# --- Local Application Modules ---
from core.dataset import AudioDataset, MultiAudioDataset
from core.model import AudioCNN, MultiAudioCNN

# --- Dataclass: Configuration for Training ---
@dataclass
class TrainConfig:
    train_folder: str = ""
    class_names: list = field(default_factory=list)
    epochs: int = 2
    model_folder: str = "models"
    model_name: str = None
    load_model: bool = False
    save_model: bool = False
    old_model: bool = False
    sample_rate: int = 16000
    batch_size: int = 32
    lr: float = 1e-3

# --- Helper Function: Detect and Validate Class Directories ---
def detect_and_validate_class_dirs(train_folder, class_names=None):
    train_folder = Path(train_folder)

    if not class_names:
        class_names = sorted([folder.name for folder in train_folder.iterdir() if folder.is_dir()])
    
    class_dirs  = [train_folder / class_name for class_name in class_names]

    for class_dir in class_dirs:
        assert class_dir.exists(), f"\n{class_dir} does not exist"

    return class_dirs

# --- Helper Function: Get Device ---
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)} \n")
    else:
        print(f"Using CPU \n")
    return device



# --- Train Function ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()                            # set model to training mode
    total_loss = 0.0                         # accumulate loss
    correct = 0                              # correct prediction counter
    total = 0                                # total samples

    for features, labels in dataloader:      # iterate over batches
        features = features.to(device)       # move inputs to device
        labels   = labels.to(device)         # move labels to device

        optimizer.zero_grad()                # zero gradients
        outputs = model(features)            # forward pass

        # Detect binary vs multi-class
        if outputs.shape[1] == 1:
            labels = labels.float().unsqueeze(1) # shape: [batch, 1]
            preds = (outputs > 0.5).float()      # convert probabilities to binary predictions
        else:
            labels = labels.long()               # shape: [batch]
            _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)    # compute loss
        loss.backward()                      # backpropagation
        optimizer.step()                     # update weights
        
        num_correct = (preds == labels).sum().item()  # number of correct predictions in batch
        correct    += num_correct                     # accumulate total correct predictions
        total_loss += loss.item()            # accumulate loss
        total      += labels.size(0)         # update total count

    return total_loss, correct, total        # return accumulated loss and accuracy


# --- Main Function ---
def train_audio_detection_model(
    train_folder,
    class_names,
    sample_rate,
    batch_size,
    epochs,
    lr,
    model_name,
    load_model,
    save_model,
    old_model,
    model_folder
    ):
    """
    Trains an audio classification model using CNN on mel-spectrogram data.
    Handles model loading, training loop, and optional model saving.
    """
    # --- Helper Function: Detect and Validate Class Directories ---
    class_dirs = detect_and_validate_class_dirs(train_folder, class_names)
    
    # --- Helper Function: Get Device ---
    device = get_device()                   

    # --- Transform ---
    melspec_transform = nn.Sequential(                       # preprocessing pipeline
        MelSpectrogram(sample_rate=sample_rate, n_mels=64),  # convert waveform to mel spectrogram
        AmplitudeToDB()                                      # convert amplitude to decibels
    )
    
    # --- Model Initialization, Dataset and Criterion ---
    if old_model and len(class_dirs) == 2:
        model = AudioCNN(num_classes=1).to(device)  # binary classification  
        dataset = AudioDataset(class_dirs, melspec_transform, sample_rate) 
        criterion = nn.BCELoss()
    else:
        model = MultiAudioCNN(num_classes=len(class_dirs)).to(device)  
        dataset = MultiAudioDataset(class_dirs, melspec_transform, sample_rate) 
        criterion = nn.CrossEntropyLoss()            # cross entropy loss for multi-class classification

    # --- Optimizer, Dataloader ---
    optimizer  = optim.Adam(model.parameters(), lr=lr)   
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Optional: Load Model ---
    if load_model:
        model_path = Path(model_folder) / model_name
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"\nModel successfully loaded from: {model_path} \n")
        except Exception as e:
            print(f"\nFailed to load model from {model_path}: {e}")
            return 

    # --- Training Loop ---
    tic = time.perf_counter()

    for epoch in range(epochs):
        total_loss, correct, total = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.6f} | Accuracy: {100 * correct / total:.2f}%")

    toc = time.perf_counter()

    # --- Print Results ---
    print(f"\nTraining completed in {(toc - tic)/60:.2f} minutes")
    print(f"Time per epoch: {(toc - tic)/epochs:.2f} seconds, Epochs per minute: {60/((toc - tic)/epochs):.2f}")

    # --- Optional: Save Model ---
    if save_model:
        model_save_path = Path(model_folder) / model_name
        model_save_path.parent.mkdir(parents=True, exist_ok=True)  # Create dir if needed
        torch.save(model.state_dict(), model_save_path)
        print(f"\nModel saved to {model_save_path}") 




# --- FUTURE WORK: SAVE BEST MODEL ---
"""

    # --- TESTING ---
    
    from core.test import test_multiple_models
    import sys

    Dataset_dir = Path.cwd().parent
    sys.path.append(str(Path.cwd().parent))
    print(f"Dataset_dir: {Dataset_dir}")
    dataset_path = Dataset_dir / "Dataset"/ "local_georg_datasets"
    print(f"Dataset_path: {dataset_path}")
    # ---------------

    
    # --- TESTING ---
    curr_testing_accuracy = 0.0
    prev_testing_accuracy = 0.0
    # ---------------


    # --- TESTING ---
    from core.test import test_multiple_models
    import sys

    Dataset_dir = Path.cwd().parent
    sys.path.append(str(Path.cwd().parent))
    dataset_path = Dataset_dir / "Dataset"/ "local_georg_datasets"

    curr_testing_accuracy = test_multiple_models(
        test_folder=str(dataset_path),
        test_datasets = ["test/"],
        class_names = ["biden", "city", "cricket", "drone", "forest", "harris", "trump"],
        models_dir = "models/",
        models = [model_save_name],
        overall_accuracy = False,
        output_false_pred = False,
        old_model=False,
        in_training = True,
        model_testing = model
    )
    print(f"Overall testing accuracy: {curr_testing_accuracy:.2f}%")
    # if current testing_accuracy is bigger than the previous one, save the model
    if curr_testing_accuracy > prev_testing_accuracy:
        print(f"Current testing accuracy: {curr_testing_accuracy:.2f}% is better than previous one: {prev_testing_accuracy:.2f}%")
        # model_save_path = Path("models") / model_save_name
        # model_save_path.parent.mkdir(parents=True, exist_ok=True)  # Create dir if needed
        # torch.save(model.state_dict(), model_save_path)
        prev_testing_accuracy = curr_testing_accuracy
    print(f"\n")
    # ----------------
"""