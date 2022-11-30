import torch.nn as nn
from torch.nn import init


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)


class DCGAN_MNIST_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, n_extra_layers=0):
        super(DCGAN_MNIST_D, self).__init__()

        self.main = nn.Sequential(
                Unflatten(-1, 1, 28, 28),
                nn.Conv2d(1, 32, 5),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 5),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, 2),
                Flatten(),
                nn.Linear(4 * 4 * 64, 4 * 4 * 64),
                nn.LeakyReLU(),
                nn.Linear(4 * 4 * 64, 1)
            )
    
    def forward(self, input):
        output = self.main(input)
        return output


class DCGAN_MNIST_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(DCGAN_MNIST_G, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(nz, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 7 * 7 * 128),
                nn.ReLU(),
                nn.BatchNorm1d(7 * 7 * 128),
                Unflatten(N=-1, C=128, H=7, W=7),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
                nn.Tanh(),
            )
    
    def forward(self, input):
        output = self.main(input)
        return output
