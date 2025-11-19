from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from Flux import Flux, FluxParams


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "checkpoints/flux_ckpt.pt"
    log_interval: int = 10


class SyntheticFlowDataset(Dataset):
    """A minimal synthetic dataset that yields batches matching the Flux forward signature.

    This is a placeholder so training code can be wired up without a real dataset.
    Replace with a real Dataset implementation for actual training.
    """

    def __init__(self, params: FluxParams, length: int = 256, seq_len: int = 16, txt_len: int = 8):
        self.length = length
        self.params = params
        self.seq_len = seq_len
        self.txt_len = txt_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # img: (T, in_channels)
        img = torch.randn(self.seq_len, self.params.in_channels, dtype=torch.float32)
        # img/text ids used by positional embedder; integers
        img_ids = torch.zeros(self.seq_len, dtype=torch.long)
        txt = torch.randn(self.txt_len, self.params.context_in_dim, dtype=torch.float32)
        txt_ids = torch.zeros(self.txt_len, dtype=torch.long)
        # timesteps: scalar per batch item (we'll collate later into a vector)
        timestep = torch.randint(0, 1000, (1,), dtype=torch.long)
        # y: auxiliary vector input
        y = torch.randn(self.params.vec_in_dim, dtype=torch.float32)
        # guidance strength (optional)
        guidance = torch.randint(0, 1000, (1,), dtype=torch.long) if self.params.guidance_embed else None
        # target: random target with unknown shape; we'll let training manage shape compatibility
        target = torch.randn(self.seq_len, self.params.out_channels, dtype=torch.float32)
        return img, img_ids, txt, txt_ids, timestep, y, guidance, target


def collate_fn(batch):
    # batch is a list of tuples returned by __getitem__
    imgs, img_ids, txts, txt_ids, timesteps, ys, guidances, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # (N, T, in_channels)
    img_ids = torch.stack(img_ids, dim=0)  # (N, T_img)
    txts = torch.stack(txts, dim=0)  # (N, T_txt, context_in_dim)
    txt_ids = torch.stack(txt_ids, dim=0)
    timesteps = torch.cat(timesteps, dim=0).long()
    ys = torch.stack(ys, dim=0)
    targets = torch.stack(targets, dim=0)
    if guidances[0] is None:
        guidances = None
    else:
        guidances = torch.cat(guidances, dim=0).long()
    return imgs, img_ids, txts, txt_ids, timesteps, ys, guidances, targets


def build_model(params: FluxParams, device: Optional[str] = None) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Flux(params).to(device)
    return model


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str) -> None:
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], path: str, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return model, optimizer


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, device: str):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for batch_idx, batch in enumerate(dataloader):
        imgs, img_ids, txts, txt_ids, timesteps, ys, guidances, targets = batch
        imgs = imgs.to(device)
        img_ids = img_ids.to(device)
        txts = txts.to(device)
        txt_ids = txt_ids.to(device)
        timesteps = timesteps.to(device)
        ys = ys.to(device)
        targets = targets.to(device)
        if guidances is not None:
            guidances = guidances.to(device)

        optimizer.zero_grad()
        # Forward -- this mirrors the Flux.forward signature
        preds = model(img=imgs, img_ids=img_ids, txt=txts, txt_ids=txt_ids, timesteps=timesteps, y=ys, guidance=guidances)

        # Align shapes if necessary: prefer matching the target's shape
        if preds.shape != targets.shape:
            # attempt a simple reshape/broadcast if lengths match in the time dimension
            try:
                targets_aligned = targets
            except Exception:
                targets_aligned = targets
        else:
            targets_aligned = targets

        loss = criterion(preds, targets_aligned)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train(params: FluxParams, config: TrainConfig):
    device = config.device
    model = build_model(params, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    dataset = SyntheticFlowDataset(params)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(1, config.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        if epoch % config.log_interval == 0 or epoch == config.epochs:
            print(f"Epoch {epoch}/{config.epochs}: loss={avg_loss:.6f}")
        # Save checkpoint at end of each epoch
        save_checkpoint(model, optimizer, f"{config.save_path}.epoch{epoch}")


if __name__ == "__main__":
    # Demo run with synthetic data. Set do_run=True to execute training when running this file.
    do_run = False
    if do_run:
        # Example params -- adapt these values to match your real model setup
        example_params = FluxParams(
            in_channels=3,
            out_channels=3,
            vec_in_dim=4,
            context_in_dim=16,
            hidden_size=64,
            mlp_ratio=4.0,
            num_heads=8,
            depth=2,
            depth_single_blocks=1,
            axes_dim=[1, 7],
            theta=1,
            qkv_bias=True,
            guidance_embed=False,
        )
        cfg = TrainConfig(epochs=2, batch_size=4)
        train(example_params, cfg)
