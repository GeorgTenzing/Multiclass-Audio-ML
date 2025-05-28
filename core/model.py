# --- Third-Party Libraries ---
import torch
import torch.nn as nn

# --- Multi-class Model ---
class MultiAudioCNN(nn.Module):                      
    """
    Convolutional neural network for multi class audio classification.
    Processes spectrogram inputs and outputs a probability.
    """
    def __init__(self, num_classes=1):
        super(MultiAudioCNN, self).__init__()
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),    # conv layer 1
            nn.ReLU(),                                     # activation
            nn.MaxPool2d(2),                               # downsample

            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # conv layer 2
            nn.ReLU(),                                     # activation
            nn.MaxPool2d(2),                               # downsample

            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # conv layer 3
            nn.ReLU(),                                     # activation
            nn.AdaptiveAvgPool2d((1, 1)),                  # global pooling to 1x1

            nn.Flatten(),                                  # flatten for dense layer
            nn.Linear(64, num_classes),                    # linear layer to output 1 value
        )

        self.binary_classifier = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.num_classes == 1:                          # Binary classification
            return self.binary_classifier(self.net(x))    
        else:
            return self.net(x)                             # Multi-class classification


# --- Binary Model ---
class AudioCNN(nn.Module):                      
    """
    Convolutional neural network for multi class audio classification.
    Processes spectrogram inputs and outputs a probability.
    """
    def __init__(self, num_classes=1):
        super(AudioCNN, self).__init__()
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),    # conv layer 1
            nn.ReLU(),                                     # activation
            nn.MaxPool2d(2),                               # downsample

            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # conv layer 2
            nn.ReLU(),                                     # activation
            nn.MaxPool2d(2),                               # downsample

            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # conv layer 3
            nn.ReLU(),                                     # activation
            nn.AdaptiveAvgPool2d((1, 1)),                  # global pooling to 1x1

            nn.Flatten(),                                  # flatten for dense layer
            nn.Linear(64, 1),                    # linear layer to output 1 value
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)                             # Multi-class classification
    

