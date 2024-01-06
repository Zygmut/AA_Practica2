from torchvision import models

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
    def __init__(self, num_classes):
        super(CatDogNet, self).__init__()
        self.layers = Sequential(
            CatDogNet.convolutional_block(3, 64),
            CatDogNet.convolutional_block(64, 128),
            CatDogNet.convolutional_block(128, 256),
            CatDogNet.convolutional_block(256, 512),
            Flatten(),
            Linear(512 * 8 * 8, 1024),
            ReLU(),
            Dropout(0.5),
            # By default this will be binary
            Linear(1024, num_classes),
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


class CatDogResnet50(Module):
    def __init__(self, num_classes):
        super(CatDogResnet50, self).__init__()

        base_resnet50_model = models.resnet50(
            weights=models.resnet.ResNet50_Weights.DEFAULT
        )

        # Remove the final layer to replace it with a binary classificatior
        base_model_resnet50 = Sequential(*list(base_resnet50_model.children())[:-1])

        self.layers = Sequential(
            base_model_resnet50,
            Flatten(),
            Linear(512 * 4, 512),
            ReLU(),
            Dropout(0.5),
            Linear(512, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class CatDogResnet34(Module):
    def __init__(self, num_classes):
        super(CatDogResnet34, self).__init__()

        base_resnet_model = models.resnet34(
            weights=models.resnet.ResNet34_Weights.DEFAULT
        )

        # Remove the final layer to replace it with the
        # 37 target outputs
        base_model = Sequential(*list(base_resnet_model.children())[:-1])

        self.layers = Sequential(
            base_model,
            Flatten(),
            Linear(512, 512),
            ReLU(),
            Dropout(0.5),
            Linear(512, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
