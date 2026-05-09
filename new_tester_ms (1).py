import os
import glob
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from einops import rearrange
from diffusers import AutoencoderKL
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# ==========================================
# 1. VAE & Embeddings
# ==========================================

def count_parameters(model):
    """Counts the total and trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    return total_params, trainable_params

class VAEWrapper(nn.Module):
    def __init__(self, pretrained_model_name="stabilityai/sd-vae-ft-ema"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name)

    def encode(self, x):
        latent_dist = self.vae.encode(x).latent_dist
        return latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, z):
        z = z / self.vae.config.scaling_factor
        return self.vae.decode(z).sample

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)

def get_sd3_spatial_pos_embed(h, w, dim, base_resolution_S=1024, device='cpu'):
    scale = 256.0 / base_resolution_S
    y_coords = (torch.arange(h, dtype=torch.float32, device=device) - (h - 1) / 2.0) * scale
    x_coords = (torch.arange(w, dtype=torch.float32, device=device) - (w - 1) / 2.0) * scale
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    dim_half = dim // 4 
    freqs = torch.exp(-math.log(10000) * torch.arange(dim_half, dtype=torch.float32, device=device) / dim_half)
    
    emb_y = y_grid.flatten()[:, None] * freqs[None, :]
    emb_x = x_grid.flatten()[:, None] * freqs[None, :]
    pos_embed = torch.cat([torch.sin(emb_y), torch.cos(emb_y), torch.sin(emb_x), torch.cos(emb_x)], dim=-1)
    
    if pos_embed.shape[-1] < dim:
        pad = torch.zeros((pos_embed.shape[0], dim - pos_embed.shape[-1]), device=device)
        pos_embed = torch.cat([pos_embed, pad], dim=-1)
    return pos_embed

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x

class ModulatedLayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
    def forward(self, x, shift, scale):
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ==========================================
# 2. MM-DiT Single-Timestamp Architecture
# ==========================================
class MMDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.adaLN_modulation_c = nn.Linear(hidden_size, 6 * hidden_size)
        self.adaLN_modulation_x = nn.Linear(hidden_size, 6 * hidden_size)

        self.norm_c1 = ModulatedLayerNorm(hidden_size)
        self.norm_x1 = ModulatedLayerNorm(hidden_size)
        self.norm_c2 = ModulatedLayerNorm(hidden_size)
        self.norm_x2 = ModulatedLayerNorm(hidden_size)

        self.qkv_c = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.qkv_x = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        self.proj_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_x = nn.Linear(hidden_size, hidden_size, bias=False)

        self.mlp_c = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.SiLU(), nn.Linear(hidden_size * 4, hidden_size))
        self.mlp_x = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.SiLU(), nn.Linear(hidden_size * 4, hidden_size))

    def forward(self, c, x, y):
        mod_c = self.adaLN_modulation_c(F.silu(y)).chunk(6, dim=1)
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = mod_c
        
        mod_x = self.adaLN_modulation_x(F.silu(y)).chunk(6, dim=1)
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = mod_x

        c_mod = self.norm_c1(c, shift_msa_c, scale_msa_c)
        x_mod = self.norm_x1(x, shift_msa_x, scale_msa_x)

        qkv_c = self.qkv_c(c_mod)
        qkv_x = self.qkv_x(x_mod)

        q_c, k_c, v_c = rearrange(qkv_c, 'b l (qkv d) -> qkv b l d', qkv=3)
        q_x, k_x, v_x = rearrange(qkv_x, 'b l (qkv d) -> qkv b l d', qkv=3)

        q_c = rearrange(q_c, 'b l (h d) -> b h l d', h=self.num_heads)
        k_c = rearrange(k_c, 'b l (h d) -> b h l d', h=self.num_heads)
        v_c = rearrange(v_c, 'b l (h d) -> b h l d', h=self.num_heads)
        q_x = rearrange(q_x, 'b l (h d) -> b h l d', h=self.num_heads)
        k_x = rearrange(k_x, 'b l (h d) -> b h l d', h=self.num_heads)
        v_x = rearrange(v_x, 'b l (h d) -> b h l d', h=self.num_heads)

        k = torch.cat([k_c, k_x], dim=-2)
        v = torch.cat([v_c, v_x], dim=-2)

        attn_c = F.scaled_dot_product_attention(q_c, k, v)
        attn_x = F.scaled_dot_product_attention(q_x, k, v)

        attn_c = rearrange(attn_c, 'b h l d -> b l (h d)')
        attn_x = rearrange(attn_x, 'b h l d -> b l (h d)')

        c = c + gate_msa_c.unsqueeze(1) * self.proj_c(attn_c)
        x = x + gate_msa_x.unsqueeze(1) * self.proj_x(attn_x)

        c_mod = self.norm_c2(c, shift_mlp_c, scale_mlp_c)
        x_mod = self.norm_x2(x, shift_mlp_x, scale_mlp_x)

        c = c + gate_mlp_c.unsqueeze(1) * self.mlp_c(c_mod)
        x = x + gate_mlp_x.unsqueeze(1) * self.mlp_x(x_mod)

        return c, x

class CloudRemovalSingleMMDiT(nn.Module):
    def __init__(self, latent_channels=4, hidden_size=768, depth=12, num_heads=12, patch_size=1, base_resolution_S=1024):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.base_resolution_S = base_resolution_S

        self.t_embedder = TimestepEmbedding(hidden_size)

        # Target Patcher (Noisy clean image)
        self.x_patcher = PatchEmbed(latent_channels, hidden_size, patch_size)
        
        # Context Patcher (Cloudy Latent (4) + SAR Latent (4) + Prob Mask (1) = 9 channels)
        self.c_patcher = PatchEmbed((latent_channels * 2) + 1, hidden_size, patch_size)

        self.blocks = nn.ModuleList([MMDiTBlock(hidden_size, num_heads) for _ in range(depth)])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_modulation_final = nn.Linear(hidden_size, 2 * hidden_size)
        self.unpatch_linear = nn.Linear(hidden_size, patch_size * patch_size * latent_channels)

    def forward(self, x_t, t, cloudy_latent, sar_latent, mask):
        # All latents are [B, 4, H, W], mask is [B, 1, img_H, img_W]
        B, C, latent_h, latent_w = cloudy_latent.shape
        
        y = self.t_embedder(t, self.hidden_size)

        # Target Stream
        x_tokens = self.x_patcher(x_t)
        spatial_pe = get_sd3_spatial_pos_embed(latent_h, latent_w, self.hidden_size, self.base_resolution_S, device=x_t.device)
        x_tokens = x_tokens + spatial_pe.unsqueeze(0)

        # Context Stream Setup (No Time Loop)
        # Downsample probabilistic mask to match latent space
        mask_down = F.interpolate(mask, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        
        # Concatenate: [Cloudy(4), SAR(4), Mask(1)]
        c_input = torch.cat([cloudy_latent, sar_latent, mask_down], dim=1) 
        
        c_tokens = self.c_patcher(c_input)
        c_tokens = c_tokens + spatial_pe.unsqueeze(0)

        # Pass through DiT Blocks
        c_curr, x_curr = c_tokens, x_tokens
        for block in self.blocks:
            c_curr, x_curr = block(c_curr, x_curr, y)

        # Output Projection
        mod_final = self.adaLN_modulation_final(F.silu(y)).chunk(2, dim=1)
        shift_final, scale_final = mod_final
        x_out = self.final_norm(x_curr) * (1 + scale_final.unsqueeze(1)) + shift_final.unsqueeze(1)
        
        x_out = self.unpatch_linear(x_out)
        x_out = rearrange(x_out, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                          h=latent_h, w=latent_w, p1=self.patch_size, p2=self.patch_size)

        return x_out

def get_rectified_flow_target(x_0, t):
    x_1 = torch.randn_like(x_0)
    t_expand = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
    return x_t, x_1 - x_0

# ==========================================
# 3. Dataloader for Multimodal Single Images
# ==========================================
class SAROpticalCloudDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, is_training=True):
        super().__init__()
        self.crop_size = crop_size
        self.is_training = is_training
        self.samples = []
        
        # Updated specific subdirectories based on your exact layout
        s2_dir = os.path.join(root_dir, "s2")                          # Clean Target
        s2_cloudy_dir = os.path.join(root_dir, "s2_cloudy")            # Cloudy Input
        s1_dir = os.path.join(root_dir, "s1")                          # SAR Input
        mask_dir = os.path.join(root_dir, "cloudy_prob_success")       # NPY Probabilities
        
        # Use s2 as the reference directory
        s2_files = glob.glob(os.path.join(s2_dir, "*.png"))
        
        for s2_path in s2_files:
            filename = os.path.basename(s2_path)
            base_name = os.path.splitext(filename)[0] 
            # Example base_name: "ROIs1158_spring_s2_106_p106"
            
            # String replacement to construct exact expected paths
            # 1. Cloudy: replace "_s2_" with "_s2_cloudy_"
            cloudy_name = base_name.replace("_s2_", "_s2_cloudy_") + ".png"
            cloudy_path = os.path.join(s2_cloudy_dir, cloudy_name)
            
            # 2. SAR: replace "_s2_" with "_s1_"
            sar_name = base_name.replace("_s2_", "_s1_") + ".png"
            sar_path = os.path.join(s1_dir, sar_name)
            
            # 3. Mask: replace "_s2_" with "_s2_cloudy_" and append "_cloudprob.npy"
            mask_name = base_name.replace("_s2_", "_s2_cloudy_") + "_cloudprob.npy"
            mask_path = os.path.join(mask_dir, mask_name)
            
            # Verify all modalities exist for this specific sample before adding it
            if os.path.exists(cloudy_path) and os.path.exists(sar_path) and os.path.exists(mask_path):
                self.samples.append({
                    'clean': s2_path,
                    'cloudy': cloudy_path,
                    'sar': sar_path,
                    'mask': mask_path
                })
        
        # Print only on the main GPU
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Loaded {len(self.samples)} optical-SAR paired samples from {root_dir}")

    def _read_rgb(self, path):
        """Explicitly normalizes an 8-bit PNG (0-255) to the VAE's expected [-1.0, 1.0] range."""
        img = Image.open(path).convert('RGB')
        img_np = np.array(img, dtype=np.float32)
        
        img_normalized = img_np / 255.0               # Scale to [0.0, 1.0]
        img_scaled = img_normalized * 2.0 - 1.0       # Scale to [-1.0, 1.0]
        
        tensor = torch.from_numpy(img_scaled).permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        return tensor

    def _read_npy_mask(self, path):
        """Reads probabilistic mask. Leaves values exactly as they are [0.0, 1.0]."""
        np_mask = np.load(path).copy()
        tensor = torch.from_numpy(np_mask).float()
        
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0) # Ensure channel dimension: [1, H, W]
        return tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        clean = self._read_rgb(sample['clean'])
        cloudy = self._read_rgb(sample['cloudy'])
        sar = self._read_rgb(sample['sar'])
        mask = self._read_npy_mask(sample['mask'])
        
        # Consistent Data Augmentation across all 4 spatial tensors
        if self.is_training:
            _, h, w = clean.shape
            if h > self.crop_size or w > self.crop_size:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
                clean = TF.crop(clean, top, left, self.crop_size, self.crop_size)
                cloudy = TF.crop(cloudy, top, left, self.crop_size, self.crop_size)
                sar = TF.crop(sar, top, left, self.crop_size, self.crop_size)
                mask = TF.crop(mask, top, left, self.crop_size, self.crop_size)

            if random.random() > 0.5:
                clean = TF.hflip(clean)
                cloudy = TF.hflip(cloudy)
                sar = TF.hflip(sar)
                mask = TF.hflip(mask)
                
            if random.random() > 0.5:
                clean = TF.vflip(clean)
                cloudy = TF.vflip(cloudy)
                sar = TF.vflip(sar)
                mask = TF.vflip(mask)
        else:
            clean = TF.center_crop(clean, self.crop_size)
            cloudy = TF.center_crop(cloudy, self.crop_size)
            sar = TF.center_crop(sar, self.crop_size)
            mask = TF.center_crop(mask, self.crop_size)

        return {
            "clean": clean,
            "cloudy": cloudy,
            "sar": sar,
            "mask": mask
        }

# ==========================================
# 4. Distributed Training Loop
# ==========================================
def train_rectified_flow(data_root, output_dir="checkpoints", epochs=100, batch_size=1, accumulation_steps=16, learning_rate=1e-4):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
    vae = VAEWrapper().to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Initialize Single-Timestamp MMDiT
    model = CloudRemovalSingleMMDiT(latent_channels=4, hidden_size=512, depth=8, num_heads=8).to(device)
    if local_rank == 0:
        print("\n--- Model Architecture Stats ---")
        count_parameters(model)
        print("--------------------------------\n")
    model.train()
    
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999)).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    dataset = SAROpticalCloudDataset(data_root, is_training=True)
    train_sampler = DistributedSampler(dataset, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, drop_last=True)
    
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        optimizer.zero_grad() 
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=local_rank != 0)
        
        for step, batch in enumerate(progress_bar):
            x_clean = batch["clean"].to(device).float().contiguous()
            x_cloudy = batch["cloudy"].to(device).float().contiguous()
            x_sar = batch["sar"].to(device).float().contiguous()
            mask = batch["mask"].to(device).float().contiguous()
            B = x_clean.shape[0]

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=False): 
                    # Encode Clean, Cloudy, and SAR images
                    clean_latent = vae.encode(x_clean)
                    cloudy_latent = vae.encode(x_cloudy)
                    # Treating SAR PNG as standard image for spatial compression
                    sar_latent = vae.encode(x_sar) 

            t = torch.sigmoid(torch.randn((B,), device=device))
            x_t, v_target = get_rectified_flow_target(clean_latent, t)

            with torch.amp.autocast('cuda'):
                # Pass all streams to the model
                v_pred = model(x_t, t, cloudy_latent, sar_latent, mask)
                loss = F.mse_loss(v_pred, v_target)
                
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                ema_model.update_parameters(model.module)
                optimizer.zero_grad()

            display_loss = loss.item() * accumulation_steps
            epoch_loss += display_loss
            if local_rank == 0:
                progress_bar.set_postfix({"Loss": display_loss})

        if local_rank == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
            if (epoch + 1) % 10 == 0:
                torch.save(model.module.state_dict(), os.path.join(output_dir, f"mmdit_sar_epoch_{epoch+1}.pt"))

    dist.destroy_process_group()

if __name__ == "__main__":
    # Point this to your folder containing *_clean.png, *_cloudy.png, *_sar.png, *_mask.npy
    train_rectified_flow(data_root="/home1/rounak_m/MTP_2/output")
