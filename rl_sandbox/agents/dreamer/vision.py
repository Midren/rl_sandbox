import torch.distributions as td
from torch import nn

class Encoder(nn.Module):

    def __init__(self, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[4, 4, 4]):
        super().__init__()
        layers = []

        channel_step = 96
        in_channels = 3
        for i, k in enumerate(kernel_sizes):
            out_channels = 2**i * channel_step
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=2))
            layers.append(norm_layer(1, out_channels))
            layers.append(nn.ELU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'))
            layers.append(norm_layer(1, out_channels))
            layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        # layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class Decoder(nn.Module):

    def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 6, 6]):
        super().__init__()
        layers = []
        self.channel_step = 48
        # 2**(len(kernel_sizes)-1)*channel_step
        self.convin = nn.Linear(input_size, 32 * self.channel_step)

        in_channels = 32 * self.channel_step  #2**(len(kernel_sizes) - 1) * self.channel_step
        for i, k in enumerate(kernel_sizes):
            out_channels = 2**(len(kernel_sizes) - i - 2) * self.channel_step
            if i == len(kernel_sizes) - 1:
                out_channels = 3
                layers.append(nn.ConvTranspose2d(in_channels, 4, kernel_size=k, stride=2))
            else:
                layers.append(norm_layer(1, in_channels))
                layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k,
                                       stride=2))
                layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        x = self.convin(X)
        x = x.view(-1, 32 * self.channel_step, 1, 1)
        return self.net(x)
        # return td.Independent(td.Normal(self.net(x), 1.0), 3)

class ViTDecoder(nn.Module):

    # def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 5, 3, 5, 3]):
    # def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 5, 5, 3]):
    def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 5, 3, 3]):
        super().__init__()
        layers = []
        self.channel_step = 12
        # 2**(len(kernel_sizes)-1)*channel_step
        self.convin = nn.Linear(input_size, 32 * self.channel_step)

        in_channels = 32 * self.channel_step  #2**(len(kernel_sizes) - 1) * self.channel_step
        for i, k in enumerate(kernel_sizes):
            out_channels = 2**(len(kernel_sizes) - i - 2) * self.channel_step
            if i == len(kernel_sizes) - 1:
                out_channels = 3
                layers.append(nn.ConvTranspose2d(in_channels, 384, kernel_size=k, stride=1, padding=1))
            else:
                layers.append(norm_layer(1, in_channels))
                layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=2, padding=2, output_padding=1))
                layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        x = self.convin(X)
        x = x.view(-1, 32 * self.channel_step, 1, 1)
        return td.Independent(td.Normal(self.net(x), 1.0), 3)



