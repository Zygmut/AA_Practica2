from torch.nn import Module, Conv2d, Linear, MaxPool2d
from torch.nn.functional import relu


class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
