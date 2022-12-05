import torch.nn.functional as F
from torch import nn


class CIFAR10Net(nn.Module):
    """
    customized neural network
    """

    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(480, 84)
        self.fc2 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # block1
        out = self.conv1(x)
        out = self.max_pool(out)
        out = self.relu(out)

        # block2
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.relu(out)

        # block3
        out = self.conv3(out)
        out = self.relu(out)

        # block4
        out = out.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.softmax(out)

        return out


class CNN9Layer(nn.Module):
    """
    9 Layer CNN
    """

    def __init__(self, num_classes, input_shape):
        super(CNN9Layer, self).__init__()
        self.conv1a = nn.Conv2d(input_shape, 128, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv1c = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2c = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3a = nn.Conv2d(256, 512, kernel_size=3, padding=0)
        self.conv3b = nn.Conv2d(512, 256, kernel_size=3, padding=0)
        self.conv3c = nn.Conv2d(256, 128, kernel_size=3, padding=0)

        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1a(x)
        x = self.relu(x)
        x = self.conv1b(x)
        x = self.relu(x)
        x = self.conv1c(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2a(x)
        x = self.relu(x)
        x = self.conv2b(x)
        x = self.relu(x)
        x = self.conv2c(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3a(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.relu(x)
        x = self.conv3c(x)
        x = self.relu(x)
        x = F.avg_pool2d(x, kernel_size=x.shape[2])

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
