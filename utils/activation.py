import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.beta.data.fill_(1.0)

    def forward(self, x):
        x = x * torch.sigmoid(self.beta * x)
        return x


class StarReLU(nn.Module):
    def __init__(self):
        super(StarReLU, self).__init__()
        self.s = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.s.data.fill_(1.0)
        self.b.data.fill_(0.0)

    def forward(self, x):
        x = self.s * torch.pow(torch.relu(x), 2) + self.b
        return x
