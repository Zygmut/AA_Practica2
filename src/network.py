from torch.nn import (
    Module,
    Conv2d,
    Linear,
    MaxPool2d,
    Dropout,
    ReLU,
    Flatten,
    BatchNorm2d,
    Sequential,
)


class CatDogNet(Module):
    def __init__(self):
        super(CatDogNet, self).__init__()
        self.conv_block1 = CatDogNet.convolutional_block(3, 64)
        self.conv_block2 = CatDogNet.convolutional_block(64, 128)
        self.conv_block3 = CatDogNet.convolutional_block(128, 256)
        self.conv_block4 = CatDogNet.convolutional_block(256, 512)

        self.relu = ReLU()
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()

        self.linear1 = Linear(512 * 8 * 8, 1024)
        self.linear2 = Linear(1024, 2)

        self.layers = Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.dropout,
            self.flatten,
            self.linear1,
            self.relu,
            self.dropout,
            self.linear2,
        )

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def convolutional_block(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        return Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(out_channels),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
        )
