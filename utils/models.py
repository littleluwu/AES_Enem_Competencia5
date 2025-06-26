import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_size, targets=6):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, targets)
        )

    def forward(self, x):
        return self.fc(x)
    
