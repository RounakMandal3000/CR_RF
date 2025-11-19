import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips  # ← from the pip package, not taming
from vae_cloudless import AutoEncoder, AutoEncoderParams

class LPIPSReconstructionLoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=0.0001, pixelloss_weight=1.0, perceptual_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        # LPIPS perceptual loss model
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
        self.perceptual_loss = self.perceptual_loss.to("cuda") 
        # Output log variance parameter
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, mu, logvar, weights=None, split="train"):
        # Pixel-wise L1 reconstruction loss
        
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        # Perceptual LPIPS loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(2*inputs.contiguous()-1, 2*reconstructions.contiguous()-1)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # Variance-weighted NLL-style reconstruction loss
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        if weights is not None:
            nll_loss = weights * nll_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = nll_loss.mean()

        # KL loss from the posterior
        def kl_divergence(mean, logvar):
            # Compute per-sample mean, then batch mean
            kl_per_sample = -0.5 * torch.sum(
                1 + logvar - mean.pow(2) - logvar.exp(),
                dim=[1, 2, 3]
            )
            return kl_per_sample.mean()

        kl_loss = kl_divergence(mu, logvar)
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # Total loss
        # print(self.pixel_weight * nll_loss, self.kl_weight * kl_loss)
        loss = self.pixel_weight * nll_loss + self.kl_weight * kl_loss

        # Logs for monitoring
        log = {
            f"{split}/total_loss": loss.detach().mean(),
            f"{split}/logvar": self.logvar.detach(),
            f"{split}/kl_loss": kl_loss.detach().mean(),
            f"{split}/nll_loss": nll_loss.detach().mean(),
            f"{split}/rec_loss": rec_loss.detach().mean(),
        }

        return loss, log

params = AutoEncoderParams(
    resolution=256,
    in_channels=3,
    ch=64,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=4,
    scale_factor=0.3611,
    shift_factor=0.1159,
)

@dataclass
class TrainingParams:
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_interval: int = 10
    log_interval: int = 10

tp = TrainingParams()


class S2VAEDataset(Dataset):
    def __init__(self, folder_path, transform=True):
        self.folder_path = folder_path
        self.files = []
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname.lower().endswith(".tif"):
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, folder_path)
                    self.files.append(rel_path)
        self.files = sorted(self.files)
        print(f"Found {len(self.files)} .tif files under {folder_path}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.files[idx])
        with rasterio.open(img_path) as src:
            s2 = src.read().astype(np.float32)  # (13, H, W)

        if self.transform:
            s2 = s2 / 10000.0  # Sentinel-2 reflectance scaling
            s2 = s2[[3,2,1],...]
            mean = s2.mean(axis=(1, 2), keepdims=True)
            std = s2.std(axis=(1, 2), keepdims=True) + 1e-6
            s2 = (s2 - mean) / std

        return torch.from_numpy(s2)


def train():
    model = AutoEncoder(params, sample_z=True).to(tp.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tp.learning_rate, betas=(0.5, 0.9))
    # optimizer = torch.optim.AdamW(model.parameters(), lr=tp.learning_rate, betas=(0.9, 0.99))
    warmup_epochs = 5

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,          # 0.1 * 1e-4 = 1e-5 at epoch 0
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tp.num_epochs - warmup_epochs,
        eta_min=tp.learning_rate / 10,   # 1e-5
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    dataset = S2VAEDataset(folder_path="/content/")
    dataloader = DataLoader(dataset, batch_size=tp.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Training on {len(dataset)} images for {tp.num_epochs} epochs.")
    model.train()
    loss_fn = LPIPSReconstructionLoss(perceptual_weight=1, kl_weight=1e-5)

    # scaler = torch.cuda.amp.GradScaler()
    accum_steps = 16
    optimizer.zero_grad()
    import time
    for epoch in range(tp.num_epochs):
        running_loss = 0.0
        st = time.time()
        for batch_idx, data in enumerate(dataloader):
            data = data.to(tp.device)

            # with torch.cuda.amp.autocast():
              # recon_batch, data_, mu, logvar = model(data)
              # loss, log = loss_fn(recon_batch, data, mu, logvar)
              # loss = loss / accum_steps

            # scaler.scale(loss).backward()
            
            recon_batch, data_, mu, logvar = model(data)
            loss, log = loss_fn(recon_batch, data, mu, logvar)
            loss = loss / accum_steps
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
              optimizer.step()
              optimizer.zero_grad()
            if(batch_idx + 1)%100:
                print("=")
            running_loss += loss.item()

            
        en = time.time()
        print(f" Epoch time: {en - st:.2f} seconds")
        # Step scheduler after each epoch
        print(
            f"Epoch [{epoch+1}/{tp.num_epochs}] "
            f"Loss: {running_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step()
        # Save checkpoint
        if (epoch + 1) % tp.save_interval == 0:
            save_path = f"vae_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f" Saved checkpoint → {save_path}")

    print("Training complete ")

if __name__ == "__main__":
    train()
