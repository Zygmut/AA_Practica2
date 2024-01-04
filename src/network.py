from torch.nn import (
    Module,
    Conv2d,
    Linear,
    MaxPool2d,
    Dropout,
    ReLU,
    Flatten,
    BatchNorm2d,
)
import torch


class CatDogNet(Module):
    def __init__(self):
        super(CatDogNet, self).__init__()
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()

        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.batchnorm1 = BatchNorm2d(16)
        self.batchnorm2 = BatchNorm2d(32)
        self.batchnorm3 = BatchNorm2d(128)

        self.linear1 = Linear(32 * 32 * 32, 128)
        self.linear2 = Linear(128, 2)

    def forward(self, x):
        layer_order = (
            ("conv", self.conv_1),
            ("batc", self.batchnorm1),
            ("relu", self.relu),
            ("maxp", self.maxpool),
            ("conv", self.conv_2),
            ("batc", self.batchnorm2),
            ("relu", self.relu),
            ("maxp", self.maxpool),
            ("drop", self.dropout),
            ("flat", lambda x: torch.flatten(x, 1)),
            ("line", self.linear1),
            ("batc", self.batchnorm3),
            ("relu", self.relu),
            ("drop", self.dropout),
            ("line", self.linear2),
        )

        data = x.clone()

        data_outputs = []
        for name, layer in layer_order:
            data = layer(data)
            data_outputs.append((name, data.clone()))

        return data, data_outputs
