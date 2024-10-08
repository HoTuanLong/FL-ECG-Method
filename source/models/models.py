import os, sys
from libs import *

from .blocks import *

class ResNet18(nn.Module):
    def __init__(self, 
        base_channels = 64, 
        num_classes = 2, 
    ):
        super(ResNet18, self).__init__()
        self.bblock = ResBlock
        self.stem = nn.Sequential(
            nn.Conv1d(
                12, base_channels, 
                kernel_size = 15, padding = 7, stride = 2, 
            ), 
            nn.BatchNorm1d(base_channels), 
            nn.ReLU(), 
            nn.MaxPool1d(
                kernel_size = 3, padding = 1, stride = 2, 
            ), 
        )

        self.stage_1 = nn.Sequential(
            self.bblock(base_channels*1), 
            self.bblock(base_channels*1), 
        )
        self.stage_2 = nn.Sequential(
            self.bblock(base_channels*1, downsample = True), 
            self.bblock(base_channels*2), 
        )
        self.stage_3 = nn.Sequential(
            self.bblock(base_channels*2, downsample = True), 
            self.bblock(base_channels*4), 
        )
        self.stage_4 = nn.Sequential(
            self.bblock(base_channels*4, downsample = True), 
            self.bblock(base_channels*8), 
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(
                base_channels*8, num_classes, 
            ), 
        )

    def forward(self, 
        input, 
    ):
        output = self.stem(input)

        output = self.stage_1(output)
        output = self.stage_2(output)
        output = self.stage_3(output)
        output = self.stage_4(output)

        output = self.pool(output).squeeze(-1)
        output = self.classifier(output)

        return output