import torch
import typing as t
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from rl_sandbox.vision.vae import ResBlock

class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, seq_num: int, n_dim: int, n_iter: int):
        super().__init__()

        self.seq_num = seq_num
        self.n_slots = num_slots
        self.n_iter = n_iter
        self.n_dim = n_dim
        self.scale = self.n_dim**(-1/2)
        self.epsilon = 1e-8

        # self.norm = LayerNorm()
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.n_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.n_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.slots_proj = nn.Linear(n_dim, n_dim)
        self.slots_proj_2 = nn.Linear(n_dim, n_dim)
        self.slots_norm = nn.LayerNorm(self.n_dim)
        self.slots_norm_2 = nn.LayerNorm(self.n_dim)
        self.slots_reccur = nn.GRUCell(input_size=self.n_dim, hidden_size=self.n_dim)

        self.inputs_proj = nn.Linear(n_dim, n_dim*2, )
        self.inputs_norm = nn.LayerNorm(self.n_dim)

    def forward(self, X: Float[torch.Tensor, 'batch seq h w']) -> Float[torch.Tensor, 'batch num_slots n_dim']:
        batch, seq, _, _ = X.shape
        slots = self.slots_logsigma.exp() + self.slots_mu*torch.randn(batch, self.n_slots, self.n_dim, device=X.device)

        for _ in range(self.n_iter):
            slots_prev = slots
            k, v = self.inputs_proj(self.inputs_norm(X).permute(0, 2, 3, 1).reshape(batch, -1, self.n_dim)).chunk(2, dim=-1)
            slots = self.slots_norm(slots)
            q = self.slots_proj(slots)

            attn = F.softmax(self.scale*torch.einsum('bik,bjk->bij', q, k), dim=1) + self.epsilon

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bij,bjk->bik', attn, v) / self.n_slots
            slots = self.slots_reccur(updates.reshape(-1, self.n_dim), slots_prev.reshape(-1, self.n_dim)).reshape(batch, self.n_slots, self.n_dim)
            slots = slots + self.slots_proj_2(self.slots_norm_2(slots))
        return slots

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)


class PositionalEmbedding(nn.Module):
    def __init__(self, n_dim: int):
        super().__init__()
        self.n_dim = n_dim
        self.proj = nn.Linear(4, n_dim)
        self.grid = torch.from_numpy(build_grid((64, 64)))

    def forward(self, X) -> torch.Tensor:
        return X + self.proj(self.grid.to(X.device))
        # xs = torch.arange(self.n_dim)
        # pos = torch.zeros_like(xs, dtype=torch.float)
        # for i in range(self.n_dim):
        #     pos += torch.sin(xs / (1e4**(2*i/self.n_dim)))
        # return X + pos

class SlottedAutoEncoder(nn.Module):
    def __init__(self, num_slots: int, n_iter: int):
        super().__init__()
        in_channels = 3
        latent_dim = 16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding='same'),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding='same'),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding='same'),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding='same'),
            nn.LeakyReLU(inplace=True),
        )
        seq_num = latent_dim
        n_dim = 64
        self.positional_augmenter = PositionalEmbedding(n_dim)
        self.slot_attention = SlotAttention(num_slots, seq_num, n_dim, n_iter)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=(2, 2), padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=(2, 2), padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=(2, 2), padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=(1, 1), padding=1),
        )

    def forward(self, X: Float[torch.Tensor, 'batch 3 h w']) -> t.Tuple[Float[torch.Tensor, 'batch 3 h w'], Float[torch.Tensor, 'batch num_slots 4 h w']]:
        features = self.encoder(X) # -> batch D 64
        features_with_pos_enc = self.positional_augmenter(features) # -> batch D 64

        slots = self.slot_attention(features_with_pos_enc) # -> batch num_slots 64
        slots = slots.flatten(0, 1).reshape(-1, 1, 1, 64)
        slots = slots.repeat((1, 8, 8, 1))

        decoded_imgs, masks = self.decoder(slots.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(X.shape[0], -1, *X.shape[2:], 4).split([3, 1], dim=-1)

        decoded_imgs = decoded_imgs * F.softmax(masks, dim=1)
        rec_img = torch.sum(decoded_imgs, dim=1)
        return rec_img.permute(0, 3, 1, 2), decoded_imgs.permute(0, 1, 4, 2, 3)

if __name__ == '__main__':
    device = 'cuda'
    ToTensor = tv.transforms.Compose([tv.transforms.ToTensor(),
                                      tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ]
                                     )
    train_data = tv.datasets.ImageFolder('~/rl_old/rl_sandbox/crafter_data_2/', transform=ToTensor)
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=64,
                                                    shuffle=True,
                                                    num_workers=8)
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logger = SummaryWriter(log_dir=f"vae_tmp/{current_time}_{socket.gethostname()}", comment="Added lr scheduler")

    number_of_slots = 7
    slots_iter_num = 3

    total_steps = 5e5
    warmup_steps = 1e4
    decay_rate = 0.5
    decay_steps = 1e5

    model = SlottedAutoEncoder(number_of_slots, slots_iter_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    lr_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=int(warmup_steps))
    lr_decay_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay_rate**(epoch/decay_steps))
    # lr_decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate**(1/decay_steps))
    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([lr_warmup_scheduler, lr_decay_scheduler])

    global_step = 0
    epoch = 0
    pbar = tqdm(total=total_steps, desc='Training')
    while global_step < total_steps:
        epoch += 1
        logger.add_scalar('epoch', epoch, epoch)

        for sample_num, (img, target) in enumerate(train_data_loader):
            recovered_img, _ = model(img.to(device))

            reg_loss = F.mse_loss(img.to(device), recovered_img)

            optimizer.zero_grad()
            reg_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            logger.add_scalar('train/loss', reg_loss.cpu().detach(), epoch * len(train_data_loader) + sample_num)
            pbar.update(1)

        for i in range(3):
            img, target = next(iter(train_data_loader))
            recovered_img, imgs_per_slot = model(img.to(device))
            unnormalize = tv.transforms.Compose([
                tv.transforms.Normalize((0, 0, 0), (1/0.229, 1/0.224, 1/0.225)),
                tv.transforms.Normalize((-0.485, -0.456, -0.406), (1., 1., 1.))
            ])
            logger.add_image(f'val/example_image', unnormalize(img.cpu().detach()[0]), epoch*3 + i)
            logger.add_image(f'val/example_image_rec', unnormalize(recovered_img.cpu().detach()[0]), epoch*3 + i)
            for i in range(6):
                logger.add_image(f'val/example_image_rec_{i}', unnormalize(imgs_per_slot.cpu().detach()[0][i]), epoch*3 + i)
