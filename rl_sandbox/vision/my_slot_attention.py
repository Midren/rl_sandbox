import torch
import typing as t
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from rl_sandbox.vision.dino import ViTFeat
from rl_sandbox.utils.logger import Logger

class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, seq_num: int, n_dim: int, n_iter: int):
        super().__init__()

        self.seq_num = seq_num
        self.n_slots = num_slots
        self.n_iter = n_iter
        self.n_dim = n_dim
        self.scale = self.n_dim**(-1/2)
        self.epsilon = 1e-8

        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.n_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.n_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.slots_proj = nn.Linear(n_dim, n_dim)
        self.slots_proj_2 = nn.Sequential(
                nn.Linear(n_dim, n_dim*4),
                nn.ReLU(inplace=True),
                nn.Linear(n_dim*4, n_dim),
            )
        self.slots_norm = nn.LayerNorm(self.n_dim)
        self.slots_norm_2 = nn.LayerNorm(self.n_dim)
        self.slots_reccur = nn.GRUCell(input_size=self.n_dim, hidden_size=self.n_dim)

        self.inputs_proj = nn.Linear(n_dim, n_dim*2)
        self.inputs_norm = nn.LayerNorm(self.n_dim)

    def forward(self, X: Float[torch.Tensor, 'batch seq n_dim']) -> Float[torch.Tensor, 'batch num_slots n_dim']:
        batch, _, _ = X.shape
        k, v = self.inputs_proj(self.inputs_norm(X)).chunk(2, dim=-1)

        slots = self.slots_mu + self.slots_logsigma.exp() * torch.randn(batch, self.n_slots, self.n_dim, device=X.device)

        for _ in range(self.n_iter):
            slots_prev = slots
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
    def __init__(self, n_dim: int, res: t.Tuple[int, int]):
        super().__init__()
        self.n_dim = n_dim
        self.proj = nn.Linear(4, n_dim)
        self.register_buffer('grid', torch.from_numpy(build_grid(res)))

    def forward(self, X) -> torch.Tensor:
        return X + self.proj(self.grid).permute(0, 3, 1, 2)

class SlottedAutoEncoder(nn.Module):
    def __init__(self, num_slots: int, n_iter: int):
        super().__init__()
        in_channels = 3
        latent_dim = 16
        self.n_dim = 196
        self.lat_dim = int(self.n_dim**0.5)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n_dim, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.n_dim, self.n_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_dim, self.n_dim)
        )

        seq_num = latent_dim
        self.dino_vit = ViTFeat("/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth", feat_dim=384, vit_arch='small', patch_size=16)
        self.positional_augmenter_inp = PositionalEmbedding(self.n_dim, (13, 13))
        self.positional_augmenter_dec = PositionalEmbedding(self.n_dim, (self.lat_dim, self.lat_dim))
        self.positional_augmenter_vit_dec = PositionalEmbedding(self.n_dim, (14, 14))
        self.slot_attention = SlotAttention(num_slots, seq_num, self.n_dim, n_iter)
        self.img_decoder = nn.Sequential( # Dx14x14 -> (3+1)x112x112
            nn.ConvTranspose2d(self.n_dim, 64, kernel_size=5, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=(1, 1), padding=1),
        )

        self.vit_decoder = nn.Sequential( # Dx1x1 -> (384+1)x14x14
            nn.ConvTranspose2d(self.n_dim, 192, kernel_size=5, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 192, kernel_size=5, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 384, kernel_size=5, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 576, kernel_size=3, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(576, 385, kernel_size=3, stride=(1, 1), padding=1),
        )

        # self.vit_decoder_mlp = nn.Sequential(
        #         nn.Linear(self.n_dim, 1024),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(1024, 1024),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(1024, 1024),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(1024, 385),
        #         nn.ReLU(inplace=True)
        # )

    def forward(self, X: Float[torch.Tensor, 'batch 3 h w']) -> t.Tuple[Float[torch.Tensor, 'batch 3 h w'], Float[torch.Tensor, 'batch num_slots 4 h w']]:
        features = self.encoder(X) # -> batch D h w
        features_with_pos_enc = self.positional_augmenter_inp(features) # -> batch D h w

        batch, seq, _, _ = X.shape
        vit_features = self.dino_vit(X)
        vit_res_num = int(vit_features.shape[-1]**0.5)
        vit_features = vit_features.reshape(batch, -1, vit_res_num, vit_res_num)

        pre_slot_features = self.mlp(features_with_pos_enc.permute(0, 2, 3, 1).reshape(batch, -1, self.n_dim))

        slots = self.slot_attention(pre_slot_features) # -> batch num_slots D
        slots = slots.flatten(0, 1).reshape(-1, 1, 1, self.n_dim).permute(0, 3, 1, 2)

        # slots_with_vit_pos_enc = self.positional_augmenter_vit_dec(slots.flatten(2, 3).repeat((1, 1, 196)).reshape(-1, self.n_dim, 14, 14)).flatten(2, 3)
        # decoded_features, vit_masks =self.vit_decoder_mlp(slots_with_vit_pos_enc).reshape(batch, -1, vit_res_num, vit_res_num, 385).split([384, 1], dim=-1)

        decoded_features, vit_masks = self.vit_decoder(slots).permute(0, 2, 3, 1).reshape(batch, -1, vit_res_num, vit_res_num, 385).split([384, 1], dim=-1)
        vit_mask = F.softmax(vit_masks, dim=1)

        rec_features = (decoded_features * vit_mask).sum(dim=1)

        slots = slots.repeat((1, 1, self.lat_dim, self.lat_dim)) # -> batch*num_slots D sqrt(D) sqrt(D)
        slots_with_pos_enc = self.positional_augmenter_dec(slots)

        decoded_imgs, masks = self.img_decoder(slots_with_pos_enc).permute(0, 2, 3, 1).reshape(batch, -1, *(np.array(X.shape[2:])//2), 4).split([3, 1], dim=-1)
        img_mask = F.softmax(masks, dim=1)

        decoded_imgs = decoded_imgs * img_mask
        rec_img = torch.sum(decoded_imgs, dim=1)
        return rec_img.permute(0, 3, 1, 2), decoded_imgs.permute(0, 1, 4, 2, 3), F.mse_loss(rec_features.permute(0, 3, 1, 2), vit_features), vit_mask

if __name__ == '__main__':
    device = 'cuda'
    debug = False
    ToTensor = tv.transforms.Compose([tv.transforms.ToTensor(),
                                      tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      tv.transforms.Resize(224, antialias=True),
                                    ])
    train_data = tv.datasets.ImageFolder('~/rl_sandbox/crafter_data/', transform=ToTensor)
    if debug:
        train_data_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=4,
                                                        prefetch_factor=1,
                                                        shuffle=False,
                                                        num_workers=2)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=32,
                                                        shuffle=True,
                                                        num_workers=8)

    import socket
    from datetime import datetime
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    comment = "Added vit masks logging, lambda=0.1, return old dino".replace(" ", "_")
    logger = Logger(None if debug else 'tensorboard', message=comment, log_dir=f"vae_tmp/{current_time}_{socket.gethostname()}_{comment}")

    number_of_slots = 7
    slots_iter_num = 3

    total_steps = 5e5
    warmup_steps = 1e4
    decay_rate = 0.5
    decay_steps = 1e5
    val_every = 1e4

    model = SlottedAutoEncoder(number_of_slots, slots_iter_num).to(device)
    # model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    lr_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=int(warmup_steps))
    lr_decay_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay_rate**(epoch/decay_steps))
    # lr_decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate**(1/decay_steps))
    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([lr_warmup_scheduler, lr_decay_scheduler])

    global_step = 0
    prev_global_step = 0
    epoch = 0
    pbar = tqdm(total=total_steps, desc='Training')
    while global_step < total_steps:
        for sample_num, (img, target) in enumerate(train_data_loader):
            recovered_img, _, vit_rec_loss, _ = model(img.to(device))

            reg_loss = F.mse_loss(img.to(device)[:, :, ::2, ::2], recovered_img)

            lambda_ = 0.1
            loss = lambda_ * reg_loss + (1 - lambda_) * vit_rec_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            logger.add_scalar('train/img_rec_loss', reg_loss.cpu().detach(), epoch * len(train_data_loader) + sample_num)
            logger.add_scalar('train/vit_rec_loss', vit_rec_loss.cpu().detach(), epoch * len(train_data_loader) + sample_num)
            logger.add_scalar('train/loss', loss.cpu().detach(), epoch * len(train_data_loader) + sample_num)
            pbar.update(1)
        global_step += len(train_data_loader)

        epoch += 1
        logger.add_scalar('epoch', epoch, epoch)

        if global_step - prev_global_step > val_every:
            prev_global_step = global_step
        else:
            continue

        for i in range(3):
            img, target = next(iter(train_data_loader))
            recovered_img, imgs_per_slot, _, vit_mask = model(img.to(device))
            upscale = tv.transforms.Resize(224, antialias=True)
            unnormalize = tv.transforms.Compose([
                tv.transforms.Normalize((0, 0, 0), (1/0.229, 1/0.224, 1/0.225)),
                tv.transforms.Normalize((-0.485, -0.456, -0.406), (1., 1., 1.))
            ])
            logger.add_image(f'val/example_image', unnormalize(img.cpu().detach()[0]), epoch*3 + i)
            logger.add_image(f'val/example_image_rec', unnormalize(recovered_img.cpu().detach()[0]), epoch*3 + i)
            per_slot_img = unnormalize(imgs_per_slot.cpu().detach())[0].permute((1, 2, 0, 3)).flatten(2, 3)
            logger.add_image(f'val/example_image_slot_rec', per_slot_img, epoch*3 + i)
            upscaled_mask = upscale(vit_mask.permute(0, 1, 4, 2, 3).squeeze())
            per_slot_vit = (upscaled_mask.unsqueeze(2) * img.to(device).unsqueeze(1))[0].permute(1, 2, 0, 3).flatten(2, 3)
            logger.add_image(f'val/example_vit_slot_mask', per_slot_vit/upscaled_mask.max(), epoch*3 + i)

