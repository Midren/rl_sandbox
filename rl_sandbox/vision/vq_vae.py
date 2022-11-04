from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL.Image import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl_sandbox.vision.vae import ResBlock


class VQ_VAE(nn.Module):

    def __init__(self, latent_space_size, latent_dim, beta=0.25):
        super().__init__()
        # amount of the discrete vectors
        self.latent_space_size = latent_space_size
        # dimensionality of each category
        self.latent_dim = latent_dim
        self.beta = beta

        self.latent_space = torch.nn.Parameter(
            torch.empty(size=(self.latent_space_size, self.latent_dim)))
        torch.nn.init.kaiming_uniform_(self.latent_space)

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
            nn.Conv2d(out_channels, latent_dim, 1),  # Dx8x8
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, out_channels, 1),
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

    def quantize(self, z):
        # z <- BxDxHxW
        # Pytorch BUG: https://github.com/pytorch/pytorch/issues/84206
        # .to(memory_format=torch.contiguous_format) should be used instead of .contigious() on mac m1
        latents = torch.permute(z, (0, 2, 3, 1)).to(memory_format=torch.contiguous_format)  # BxHxWxD
        flatten = latents.view(-1, self.latent_dim)  # BHWxD

        # use the property that (a - b)^2 = a^2 - 2ab + b^2
        l2_dist = torch.sum(flatten**2, dim=1, keepdim=True) - 2 * (
            flatten @ self.latent_space.T) + torch.sum(self.latent_space**2, dim=1) # BHWxK

        ks = torch.argmin(l2_dist, dim=1)

        flatten_quantized_latents = torch.index_select(self.latent_space, 0, ks) # BHWxD
        e = flatten_quantized_latents.view(latents.shape).permute((0, 3, 1, 2)).to(memory_format=torch.contiguous_format)
        z.retain_grad()
        e.grad = z.grad
        return e


    def forward(self, X):
        z = self.encoder(X)
        e = self.quantize(z)
        x_h = self.decoder(e)
        return x_h, z, e

    def calculate_loss(self, x, x_h, z, e) -> dict[str, torch.Tensor]:
        # loss = log p(x | z) + || stop_grad(e) - z ||_2 + beta *|| e - stop_grad(z) ||_2
        L_rec = torch.nn.MSELoss()

        loss_reg = torch.norm(e.detach() - z,
                              p=2) + self.beta * torch.norm(e - z.detach(), p=2)
        loss_rec = L_rec(x, x_h)

        return {'loss': loss_rec + loss_reg, 'loss_rec': loss_rec, 'loss_reg': loss_reg}


def image_preprocessing(img: Image):
    return torchvision.transforms.ToTensor()(img)


if __name__ == "__main__":
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
    model = VQ_VAE(latent_space_size=256, latent_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in tqdm(range(100)):

        logger.add_scalar('epoch', epoch, epoch)

        for sample_num, (img, target) in enumerate(train_data_loader):
            recovered_img, z, e = model(img.to(device))

            losses = model.calculate_loss(img.to(device), recovered_img, z, e)
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
