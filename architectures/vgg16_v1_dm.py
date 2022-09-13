import torch.nn as nn

def get_block(dropout, num_classes=2):
    return nn.Sequential(

        nn.Linear(256, 64),
        nn.ELU(),

        nn.Dropout(p=dropout),

        nn.Linear(64, 16),
        nn.ELU(),

        nn.Linear(16, num_classes)
    )
