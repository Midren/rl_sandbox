import torch.distributions as td
from torch import nn


class Encoder(nn.Module):

    def __init__(self, norm_layer: nn.GroupNorm | nn.Identity,
                    channel_step=96,
                    kernel_sizes=[4, 4, 4],
                    double_conv=False,
                    flatten_output=True,
                    in_channels=3,
                ):
        super().__init__()
        layers = []

        for i, k in enumerate(kernel_sizes):
            out_channels = 2**i * channel_step
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=2))
            layers.append(norm_layer(1, out_channels))
            layers.append(nn.ELU(inplace=True))
            if double_conv:
                layers.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'))
                layers.append(norm_layer(1, out_channels))
                layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        if flatten_output:
            layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class Decoder(nn.Module):

    def __init__(self,
                 input_size,
                 norm_layer: nn.GroupNorm | nn.Identity,
                 kernel_sizes=[5, 5, 6, 6],
                 channel_step = 48,
                 output_channels=3,
                 conv_kernel_sizes=[],
                 return_dist=True):
        super().__init__()
        layers = []
        self.channel_step = channel_step
        self.in_channels = 2 **(len(kernel_sizes)-1) * self.channel_step
        in_channels = self.in_channels
        self.convin = nn.Linear(input_size, in_channels)
        self.return_dist = return_dist

        for i, k in enumerate(kernel_sizes):
            out_channels = 2**(len(kernel_sizes) - i - 2) * self.channel_step
            if i == len(kernel_sizes) - 1:
                out_channels = output_channels
                layers.append(nn.ConvTranspose2d(in_channels,
                                                 output_channels,
                                                 kernel_size=k,
                                                 stride=2,
                                                 output_padding=0))
            else:
                layers.append(
                    nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=k,
                                       stride=2,
                                       output_padding=0))
                layers.append(norm_layer(1, out_channels))
                layers.append(nn.ELU(inplace=True))
                for k in conv_kernel_sizes:
                    layers.append(
                        nn.Conv2d(out_channels,
                                  out_channels,
                                  kernel_size=k,
                                  padding='same'))
                    layers.append(norm_layer(1, out_channels))
                    layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        x = self.convin(X)
        x = x.view(-1, self.in_channels, 1, 1)
        if self.return_dist:
            return td.Independent(td.Normal(self.net(x), 1.0), 3)
        else:
            return self.net(x)


class ViTDecoder(nn.Module):

    # def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 5, 3, 5, 3]):
    # def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 5, 5, 3]):
    def __init__(self,
                 input_size,
                 norm_layer: nn.GroupNorm | nn.Identity,
                 kernel_sizes=[5, 5, 5, 3, 3]):
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
                layers.append(
                    nn.ConvTranspose2d(in_channels,
                                       384,
                                       kernel_size=k,
                                       stride=1,
                                       padding=1))
            else:
                layers.append(norm_layer(1, in_channels))
                layers.append(
                    nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=k,
                                       stride=2,
                                       padding=2,
                                       output_padding=1))
                layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        x = self.convin(X)
        x = x.view(-1, 32 * self.channel_step, 1, 1)
        return td.Independent(td.Normal(self.net(x), 1.0), 3)
