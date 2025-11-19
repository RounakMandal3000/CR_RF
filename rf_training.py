import torch
import torch.nn as nn
from torch import Tensor
from PIL import Image
import numpy as np
from einops import rearrange, repeat
import math
from traitlets import Callable
from torch.utils.data import Dataset
import os
from Cloud_removal_Flux import Flux, FluxParams
from vae_cloudless import AutoEncoder, AutoEncoderParams
import tifffile as tiff
from utils import get_noise, unpack, denoise_train
import torch
import torch.nn.functional as F
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


class RectifiedFlowLoss(nn.Module):
    def __init__(self, lpips_fn=None, w_rf=1.0, w_l1=1.0, w_lpips=1.0, w_mask=2.0):
        super().__init__()
        
        self.lpips_fn = lpips_fn
        self.w_rf = w_rf
        self.w_l1 = w_l1
        self.w_lpips = w_lpips
        self.w_mask = w_mask
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
        self.perceptual_loss = self.perceptual_loss.to("cuda") 

    def forward(self, x0, x1, xt, t, pred_v, gt_image, pred_image, cloud_prob):
        """
        x0: clean target image (N,C,H,W)
        x1: noisy endpoint sample (N,C,H,W)
        xt: interpolated sample at t   (unused for now)
        t:  timestep in [0,1] shape (N,)
        pred_v: model-predicted velocity
        pred_image: final reconstructed image
        cloud_prob: cloud probability map in [0,1], shape (N,1,H,W)
        """

        v_gt = x1 - x0
        loss_rf = ((pred_v - v_gt).pow(2)).mean()

        mask = 1 + self.w_mask * cloud_prob
        mask = mask / mask.mean()   

        loss_l1 = (mask * (pred_image - gt_image).abs()).mean()

        loss_lp = 0.0
        lp = self.perceptual_loss((pred_image * 2 - 1), (gt_image * 2 - 1))
        loss_lp = lp.mean()

        loss = self.w_rf * loss_rf + self.w_l1 * loss_l1 + self.w_lpips * loss_lp

        return loss, {
            "rf": loss_rf.item(),
            "l1": loss_l1.item(),
            "lpips": loss_lp if isinstance(loss_lp, float) else loss_lp.item(),
        }


class CloudDataset(Dataset):
    def __init__(self, root_cloudy, root_sar, root_mask, root_clean, transform=None):
        self._img_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}
        self.root_cloudy = root_cloudy
        self.root_sar = root_sar
        self.root_mask = root_mask
        self.root_clean = root_clean
        self.transform = transform

        def list_files(root, exts):
            files = [
                f for f in sorted(os.listdir(root))
                if os.path.splitext(f)[1].lower() in exts
            ]
            return files

        self.cloudy_files = list_files(root_cloudy, self._img_exts)
        self.sar_files = list_files(root_sar, self._img_exts)
        self.mask_files = list_files(root_mask, self._img_exts)
        self.clean_files = list_files(root_clean, self._img_exts)

    def load_any(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            x = np.load(path)
        else:
            if ext in {".tif", ".tiff"}:
                x = tiff.imread(path)
            else:
                print("hello, not .tif")
                from PIL import Image
                im = Image.open(path)
                # convert to RGB for 3-channel images, leave single-channel as-is
                if im.mode == "RGB":
                    x = np.array(im)  # (H,W,3)
                else:
                    x = np.array(im)  # could be (H,W) or (H,W,4)
        x = x.astype(np.float32)
        return x

    def __len__(self):
        return len(self.cloudy_files)

    def __getitem__(self, idx):
        cloudy_path = os.path.join(self.root_cloudy, self.cloudy_files[idx])
        sar_path = os.path.join(self.root_sar, self.sar_files[idx])
        mask_path = os.path.join(self.root_mask, self.mask_files[idx])
        clean_path = os.path.join(self.root_clean, self.clean_files[idx])

        cloudy = self.load_any(cloudy_path)  # could be (H,W,3) or (3,H,W) or (H,W)
        sar = self.load_any(sar_path)        # expected 2 channels (H,W,2) or (2,H,W)
        mask = self.load_any(mask_path)      # .npy often (H,W) or (1,H,W)
        clean = self.load_any(clean_path)    # clean image (H,W,3) or (3,H,W)

        # normalize array shapes to (C, H, W)
        def to_chw(x):
            if x.ndim == 2:
                x = x[None, ...]
            elif x.ndim == 3:
                if x.shape[0] in (1, 2, 3, 4) and x.shape[1] > 4:
                    pass
                else:
                    x = np.transpose(x, (2, 0, 1))
            else:
                raise ValueError(f"Unexpected array shape: {x.shape}")
            return x
        
        cloudy = to_chw(cloudy)  
        sar = to_chw(sar)
        mask = to_chw(mask)
        clean = to_chw(clean)

        VV_db = sar[0:1, ...]
        VH_db = sar[1:2, ...]
        VV_lin = 10 ** (VV_db / 10.0)
        VH_lin = 10 ** (VH_db / 10.0)
        ratio = (VV_db + VH_db)/2
        sar = np.concatenate([VV_db, VH_db, ratio], axis=0)
        mean_sar = sar.mean(axis=(1, 2), keepdims=True)
        std_sar = sar.std(axis=(1, 2), keepdims=True) + 1e-6
        sar = (sar - mean_sar) / std_sar

        cloudy = cloudy / 10000.0  # Sentinel-2 reflectance scaling
        cloudy = cloudy[[3,2,1],...]
        mean_cloudy = cloudy.mean(axis=(1, 2), keepdims=True)
        std_cloudy = cloudy.std(axis=(1, 2), keepdims=True) + 1e-6
        cloudy = (cloudy - mean_cloudy) / std_cloudy

        
        clean = clean / 10000.0  # Sentinel-2 reflectance scaling
        clean = clean[[3,2,1],...]
        mean_clean = clean.mean(axis=(1, 2), keepdims=True)
        std_clean = clean.std(axis=(1, 2), keepdims=True) + 1e-6
        clean = (clean - mean_clean) / std_clean


        mean_mask = mask.mean(axis=(1, 2), keepdims=True)
        std_mask = mask.std(axis=(1, 2), keepdims=True) + 1e-6
        mask = (mask - mean_mask) / std_mask


        cloudy = torch.from_numpy(cloudy).float()
        sar = torch.from_numpy(sar).float()
        mask = torch.from_numpy(mask).float()
        clean = torch.from_numpy(clean).float()

        sample = {
            "cloudy": cloudy,
            "sar": sar,
            "mask": mask, 
            "clean": clean
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return cloudy, sar, mask, clean


if __name__ == "__main__":
    params_flux=FluxParams(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=768,
        mlp_ratio=4.0,
        num_heads=6,
        depth=4,
        depth_single_blocks=8,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
    )
    model = Flux(params_flux)
    params_ae = AutoEncoderParams(
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
    ae = AutoEncoder(params_ae)
    sar_ae = AutoEncoder(params_ae)
    ae.load_state_dict(torch.load("model_ae.pth", map_location="cuda"))
    sar_ae.load_state_dict(torch.load("sar_model_ae.pth", map_location="cuda"))
    model.train()
    ae.eval()       # freeze VAE encoder
    sar_ae.eval()   # freeze SAR VAE encoder
    training_params = {
        'batch_size': 4,
        'learning_rate': 2e-5,
        'num_epochs': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_interval': 10,
        'log_interval': 10,
        # validation settings
        'val_fraction': 0.1,
        'val_interval': 1,  # run validation every N epochs
    }
    
    tp = training_params
    device = torch.device(tp['device'])
    model.to(device)
    ae.to(device)
    sar_ae.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tp['learning_rate'])

    # instantiate loss class and move to device
    loss_fn = RectifiedFlowLoss()
    loss_fn = loss_fn.to(device)


    dataset = CloudDataset(
        root_cloudy="data/cloudy_images/",
        root_sar="data/sar_images/",
        root_mask="data/cloud_masks/",
        root_clean="data/clean_images/",
    )
    # split into train and validation sets
    val_fraction = tp.get('val_fraction', 0.1)
    val_len = max(1, int(len(dataset) * val_fraction))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tp['batch_size'], shuffle=True, num_workers=2, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=tp['batch_size'], shuffle=False, num_workers=2, pin_memory=False)

    accump_step = 8
    for epoch in range(tp['num_epochs']):
        print(f"Epoch {epoch}")
        for batch_idx, batch in enumerate(train_loader):
            cloudy, sar, cloudmask, clean = batch
            cloudy = cloudy.to(device)
            sar = sar.to(device)
            cloudmask = cloudmask.to(device)
            clean = clean.to(device)

            B, C, target_height, target_width = cloudy.shape # batch size
            
            latent_img = get_noise(
                B,
                target_height,
                target_width,
                device=device,
                dtype=torch.bfloat16,
                seed=42
            )   
            latent_img = latent_img.to(torch.bfloat16)
            latent_img = rearrange(latent_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            bs, c, h, w = latent_img.shape
            latent_img_ids = torch.zeros(h // 2, w // 2, 3)
            latent_img_ids[..., 1] = latent_img_ids[..., 1] + torch.arange(h // 2)[:, None]
            latent_img_ids[..., 2] = latent_img_ids[..., 2] + torch.arange(w // 2)[None, :]
            latent_img_ids = repeat(latent_img_ids, "h w c -> b (h w) c", b=bs)



            with torch.no_grad():
                latent_cond = ae.encode(cloudy)      # (B, C, H, W)
            latent_cond = latent_cond.to(torch.bfloat16)
            latent_tokens = rearrange(latent_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            bs, c, h, w = latent_cond.shape
            latent_cond_img_ids = torch.zeros(h // 2, w // 2, 3, device=device)
            latent_cond_img_ids[..., 1] = latent_cond_img_ids[..., 1] + torch.arange(h // 2, device=device)[:, None]
            latent_cond_img_ids[..., 2] = latent_cond_img_ids[..., 2] + torch.arange(w // 2, device=device)[None, :]
            latent_cond_img_ids = repeat(latent_cond_img_ids, "h w c -> b (h w) c", b=bs)




            with torch.no_grad():
                sar_latent = sar_ae.encode(sar)      # (B, C, H, W)
            sar_latent = sar_latent.to(torch.bfloat16)
            sar_seq = rearrange(sar_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            bs, c, h, w = sar_latent.shape
            sar_img_ids = torch.zeros(h // 2, w // 2, 3, device=device)
            sar_img_ids[..., 1] = sar_img_ids[..., 1] + torch.arange(h // 2, device=device)[:, None]
            sar_img_ids[..., 2] = sar_img_ids[..., 2] + torch.arange(w // 2, device=device)[None, :]
            sar_img_ids = repeat(sar_img_ids, "h w c -> b (h w) c", b=bs)
            txt_ids = sar_img_ids

            t = torch.rand(B, device=device)

            with torch.no_grad():
                clean_latent = ae.encode(clean)      # (B, C, H, W)
            clean_latent = clean_latent.to(torch.bfloat16)
            clean_latent = rearrange(clean_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            
            x_t = (1-t)[:, None, None, None] * clean_latent + t[:, None, None, None] * latent_img

            pred_vel = denoise_train(
                model=model, 
                img = x_t, 
                img_ids=latent_img_ids, 
                sar_img_cond_seq = sar_seq,
                vec = cloudmask, 
                timesteps = t,  
                guidance = 2.5,
                cloudy_img_cond_seq = latent_tokens,
                cloudy_img_cond_seq_ids = latent_cond_img_ids
            )
            pred_img = x_t - t[:, None, None, None] * pred_vel
            x_pred = unpack(pred_img.float(), target_height, target_width) # height and width what here?
            x_pred = ae.decode(pred_img.float())
            
# torch.Size([1, 4, 32, 32])

# x0, x1, xt, t, pred_v, gt_image, pred_image, cloud_prob
            loss, loss_dict = loss_fn(
                x0=clean_latent,
                x1=latent_img,
                xt=x_t,
                t=t,
                pred_v=pred_vel,
                gt_image=clean,
                pred_image=x_pred,
                cloud_prob=cloudmask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print("epoch", epoch, "batch", batch_idx, "loss", loss.item())
        if(epoch + 1)%5==0:
            torch.save(model.state_dict(), f"rf_epoch_{epoch+1}.pth")
            print(f" Saved checkpoint → rf_epoch_{epoch+1}.pth")






