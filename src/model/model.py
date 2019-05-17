import torch.nn as nn
from torch.autograd import Variable
import torch

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.build_model()

    def build_model(self):
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.cnn3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=2, stride=1)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(8*3*3, 64)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 10)


    def forward(self, data):

        out = self.cnn1(data)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.cnn3(out)
        out = self.relu3(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.act1(out)

        return out
