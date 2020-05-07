import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def conv_layer_out_(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


class ConvNet(nn.Module):
    def __init__(self, in_channels, width, height, outputs, weights_init='xavier'):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        init.xavier_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        init.xavier_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        init.xavier_uniform_(self.conv3.weight)

        convw = conv_layer_out_(conv_layer_out_(conv_layer_out_(width)))
        convh = conv_layer_out_(conv_layer_out_(conv_layer_out_(height)))

        self.head = nn.Linear(convh * convw * 32, outputs)
        init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
