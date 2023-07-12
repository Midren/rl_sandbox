from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL.Image import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ResBlock(nn.Module):

    def __init__(self, in_channels, hidden_units=256):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReLU(), nn.Conv2d(in_channels, hidden_units, kernel_size=3,
                                 padding='same'), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_units, in_channels, kernel_size=1, padding='same'))

    def forward(self, X):
        output = self.block(X)
        return X + output


class VAE(nn.Module):

    def __init__(self, latent_dim=3, kl_weight=2.5e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        in_channels = 3
        out_channels = 128

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=4, stride=2,
                      padding=1),  # 32 -> 16
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels // 2),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=4, stride=2,
                      padding=1),  # 16 -> 8
            nn.LeakyReLU(inplace=True),
            ResBlock(out_channels),
            ResBlock(out_channels),
            nn.Conv2d(out_channels, 4, 1),  # 4x8x8
            nn.Flatten())

        self.f_mu = nn.Linear(256, self.latent_dim)
        self.f_log_sigma = nn.Linear(256, self.latent_dim)

        self.decoder_1 = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(inplace=True),
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(4, out_channels, 1),
            ResBlock(out_channels),
            ResBlock(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels,
                               out_channels // 2,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels // 2),
            nn.ConvTranspose2d(out_channels // 2,
                               in_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, X):
        z_h = self.encoder(X)

        z_mu = self.f_mu(z_h)
        z_log_sigma = self.f_log_sigma(z_h)

        device = next(self.f_mu.parameters()).device
        z = z_mu + z_log_sigma.exp() * torch.rand_like(z_mu).to(device)

        x_h_1 = self.decoder_1(z)
        x_h = self.decoder_2(x_h_1.view(-1, 4, 8, 8))
        return x_h, z_mu, z_log_sigma

    def calculate_loss(self, x, x_h, z_mu, z_log_sigma) -> dict[str, torch.Tensor]:
        # loss = log p(x | z) + KL(q(z) || p(z))
        # p(z) = N(0, 1)
        L_rec = torch.nn.MSELoss()

        loss_kl = -1 * torch.mean(torch.sum(
            z_log_sigma + 0.5 * (1 - z_log_sigma.exp()**2 - z_mu**2), dim=1),
                                  dim=0)
        loss_rec = L_rec(x, x_h)

        return {
            'loss': loss_rec + self.kl_weight * loss_kl,
            'loss_rec': loss_rec,
            'loss_kl': loss_kl
        }


def image_preprocessing(img: Image):
    return torchvision.transforms.ToTensor()(img)


if __name__ == "__main__":
    import torch.multiprocessing

    # fix for "unable to open shared memory on mac"
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_data = torchvision.datasets.CIFAR10(str(Path() / 'data' / 'cifar10'),
                                              download=True,
                                              train=True,
                                              transform=image_preprocessing)
    test_data = torchvision.datasets.CIFAR10(str(Path() / 'data' / 'cifar10'),
                                             download=True,
                                             train=False,
                                             transform=image_preprocessing)
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   num_workers=8)
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logger = SummaryWriter(log_dir=f"vae_tmp/{current_time}_{socket.gethostname()}")

    device = 'mps'
    model = VAE(latent_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in tqdm(range(100)):

        logger.add_scalar('epoch', epoch, epoch)

        for sample_num, (img, target) in enumerate(train_data_loader):
            recovered_img, z_mu, z_log_sigma = model(img.to(device))

            losses = model.calculate_loss(img.to(device), recovered_img, z_mu,
                                          z_log_sigma)
            loss = losses['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for loss_kind in losses:
                logger.add_scalar(f'train/{loss_kind}', losses[loss_kind].cpu().detach(),
                                  epoch * len(train_data_loader) + sample_num)

        val_losses = defaultdict(list)
        for img, target in test_data_loader:
            recovered_img, z_mu, z_log_sigma = model(img.to(device))
            losses = model.calculate_loss(img.to(device), recovered_img, z_mu,
                                          z_log_sigma)

            for loss_kind in losses:
                val_losses[loss_kind].append(losses[loss_kind].cpu().detach())

        for loss_kind in val_losses:
            logger.add_scalar(f'val/{loss_kind}', np.mean(val_losses[loss_kind]), epoch)
            logger.add_image(f'val/example_image', img.cpu().detach()[0], epoch)
            logger.add_image(f'val/example_image_rec',
                             recovered_img.cpu().detach()[0], epoch)
