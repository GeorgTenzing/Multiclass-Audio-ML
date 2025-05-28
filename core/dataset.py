# --- Standard Libraries ---
from pathlib import Path

# --- Third-Party Libraries ---
import torch
from torch.utils.data import Dataset

# --- Local Application Modules ---
from core.utils import process_audio_file


# --- Helper Function: Check that this class_dir has at least one valid file ---
def check_file_count(file_count, class_dir, verbose):
    if file_count == 0:
        raise ValueError(f"No .wav or .mp3 files found in directory: {class_dir}")
    elif verbose:
        print(f"Loaded {file_count:4d} files from class '{class_dir.name}'")


# --- Dataset ---
class MultiAudioDataset(Dataset):           
    """
    Custom PyTorch dataset for multi-class audio classification.

    Reads audio files (e.g., .wav, .mp3) from a list of class directories.
    Each subfolder corresponds to a different class.
    """
    def __init__(self, class_dirs, transform, sample_rate, extensions=("*.wav", "*.mp3"), verbose=True):
        self.transform = transform                # audio transform to apply
        self.sample_rate = sample_rate            # sample rate for audio files
        self.samples = []                         # list of audio file paths
        self.labels  = []                         # corresponding labels 

        # --- Load audio files and assign class labels ---       
        for class_index, class_dir in enumerate(class_dirs):  
            file_count = 0  
            for extension in extensions:                        # iterate over allowed audio formats (e.g., .wav, .mp3)        
                for file in class_dir.glob(extension):          # search for matching files in current class directory
                    self.samples.append(file)                   # store file path
                    self.labels.append(class_index)             # assign corresponding class index
                    file_count += 1
            check_file_count(file_count, class_dir, verbose)    # validate and optionally print number of files loaded

    def __len__(self):                            # return dataset size
        return len(self.samples)

    def __getitem__(self, idx):                   # get item by index
        features = process_audio_file(self.samples[idx], self.sample_rate, self.transform)
        return features, self.labels[idx]         # return processed features and label
    

    # --- Dataset ---
class AudioDataset(Dataset):                
    """
    Custom PyTorch dataset for binary audio classification.
    Reads .wav files from positive and negative directories.
    """
    def __init__(self, class_dirs, transform, sample_rate):
        self.transform = transform                  # audio transform to apply
        self.sample_rate = sample_rate              # sample rate for audio files
        self.samples = []                           # list of audio file paths
        self.labels  = []                           # corresponding labels 

        positive_dir = Path(class_dirs[0])         # first directory is positive
        negative_dir = Path(class_dirs[1])         # second directory is negative
   
        for file in positive_dir.glob("*.wav"):     # loop over positive audio files
            self.samples.append(file)               # only use .wav files
            self.labels.append(1)                   # label 1 = positive

        for file in negative_dir.glob("*.wav"):     # loop over negative audio files
            self.samples.append(file)               # only use .wav files
            self.labels.append(0)                   # label 0 = negative

    def __len__(self):                              # return dataset size
        return len(self.samples)

    def __getitem__(self, idx):                     # get item by index
        features = process_audio_file(self.samples[idx], self.sample_rate, self.transform)
        return features, self.labels[idx]           # return processed features and label




