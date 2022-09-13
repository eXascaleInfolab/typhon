import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(

        # Disable bias for convolutions direclty followed by a batch norm
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ELU(),

        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ELU(),

        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ELU(),

        nn.Dropout(p=dropout),
        nn.MaxPool2d(2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ELU(),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ELU(),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ELU(),

        nn.Dropout(p=dropout),
        nn.MaxPool2d(2),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ELU(),

        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ELU(),

        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ELU(),

        nn.Dropout(p=dropout),
        nn.MaxPool2d(2),

        nn.Flatten(),

        nn.Linear(2048, 256),
        nn.ELU(),

        nn.Dropout(p=dropout)
    )
