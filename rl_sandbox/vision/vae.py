from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL.Image import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, latent_dim=2, kl_weight=2.5e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.encoder = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1), # 1x28x28 -> 8x13x13
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(8),

                nn.Conv2d(8, 32, kernel_size=3, stride=2), # 8x13x13 -> 32x6x6
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(32),

                nn.Conv2d(32, 8, kernel_size=1), # 32x6x6 -> 8x6x6
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(8),

                nn.Flatten(), # 8x6x6 -> 36*8
                )

        self.f_mu = nn.Linear(288, self.latent_dim)
        self.f_log_sigma = nn.Linear(288, self.latent_dim)

        self.decoder_1 = nn.Sequential(
                nn.Linear(self.latent_dim, 288),
                nn.LeakyReLU(inplace=True),
                )

        self.decoder_2 = nn.Sequential(
                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8, 32, kernel_size=1),
                nn.LeakyReLU(inplace=True),

                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 8, kernel_size=3, stride=2),
                nn.LeakyReLU(inplace=True),

                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, output_padding=1),
                nn.LeakyReLU(inplace=True),
                )

    def forward(self, X):
        z_h = self.encoder(X)

        z_mu = self.f_mu(z_h)
        z_log_sigma = self.f_log_sigma(z_h)

        z = z_mu + z_log_sigma.exp()*torch.rand_like(z_mu).to('mps')

        x_h_1 = self.decoder_1(z)
        x_h = self.decoder_2(x_h_1.view(-1, 8, 6, 6))
        return x_h, z_mu, z_log_sigma

    def calculate_loss(self, x, x_h, z_mu, z_log_sigma) -> dict[str, torch.Tensor]:
        # loss = log p(x | z) + KL(q(z) || p(z))
        # p(z) = N(0, 1)
        L_rec = torch.nn.MSELoss()

        loss_kl = -1 * torch.mean(torch.sum(z_log_sigma + 0.5*(1 - z_log_sigma.exp()**2 - z_mu**2), dim=1), dim=0)
        loss_rec = L_rec(x, x_h)

        return {'loss': loss_rec + self.kl_weight * loss_kl, 'loss_rec': loss_rec, 'loss_kl': loss_kl}

def image_preprocessing(img: Image):
    return torchvision.transforms.ToTensor()(img)

if __name__ == "__main__":
    train_mnist_data = torchvision.datasets.MNIST(str(Path()/'data'/'mnist'),
                                                  download=True,
                                                  train=True,
                                                  transform=image_preprocessing)
    test_mnist_data = torchvision.datasets.MNIST(str(Path()/'data'/'mnist'),
                                                 download=True,
                                                 train=False,
                                                 transform=image_preprocessing)
    train_data_loader = torch.utils.data.DataLoader(train_mnist_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(test_mnist_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=8)
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logger = SummaryWriter(log_dir=f"vae_tmp/{current_time}_{socket.gethostname()}")

    device = 'mps'
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(100)):

        logger.add_scalar('epoch', epoch, epoch)

        for sample_num, (img, target) in enumerate(train_data_loader):
            recovered_img, z_mu, z_log_sigma = model(img.to(device))

            losses = model.calculate_loss(img.to(device), recovered_img, z_mu, z_log_sigma)
            loss = losses['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for loss_kind in losses:
                logger.add_scalar(f'train/{loss_kind}', losses[loss_kind].cpu().detach(), epoch*len(train_data_loader)+sample_num)

        val_losses = defaultdict(list)
        for img, target in test_data_loader:
            recovered_img, z_mu, z_log_sigma = model(img.to(device))
            losses = model.calculate_loss(img.to(device), recovered_img, z_mu, z_log_sigma)

            for loss_kind in losses:
                val_losses[loss_kind].append(losses[loss_kind].cpu().detach())

        for loss_kind in val_losses:
            logger.add_scalar(f'val/{loss_kind}', np.mean(val_losses[loss_kind]), epoch)
            logger.add_image(f'val/example_image', img.cpu().detach()[0], epoch)
            logger.add_image(f'val/example_image_rec', recovered_img.cpu().detach()[0], epoch)
