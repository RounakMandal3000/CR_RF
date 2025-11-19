import torch
import torch.nn as nn
from torch import Tensor
from PIL import Image
import numpy as np
from einops import rearrange, repeat
import math
from typing import Callable
from torch.utils.data import Dataset
import os
from Cloud_removal_Flux import Flux, FluxParams
from vae_cloudless import AutoEncoder, AutoEncoderParams
import tifffile as tiff

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def prepare_kontext(
    ae,
    sar_ae,
    sar_img_cond_path: str,
    cloud_mask_path: str,
    cloud_img_path: str,
    seed: int,
    device: torch.device,
    target_width: 256,
    target_height: 256,
    bs: int = 1,
) -> tuple[dict[str, Tensor], int, int]:
    # load and encode the conditioning image

    return_dict = {}
    sar_img_cond = Image.open(sar_img_cond_path)
    sar_img_cond = np.array(sar_img_cond)
    vv_db_sar_img_cond = sar_img_cond[..., 0]
    vh_db_sar_img_cond = sar_img_cond[..., 1]
    sar_img_cond = np.stack([vv_db_sar_img_cond, vh_db_sar_img_cond, 0.5*(vh_db_sar_img_cond+vv_db_sar_img_cond)], axis=-1)
    sar_img_cond = Image.fromarray(sar_img_cond.astype(np.uint8)).convert("RGB")
    sar_img_cond = sar_img_cond.resize((target_height, target_width), Image.Resampling.LANCZOS)
    sar_img_cond = np.array(sar_img_cond)
    sar_img_cond = torch.from_numpy(sar_img_cond).float() / 127.5 - 1.0
    sar_img_cond = rearrange(sar_img_cond, "h w c -> 1 c h w")
    sar_img_cond_orig = sar_img_cond.clone()
    with torch.no_grad():
        sar_img_cond = sar_ae.encode(sar_img_cond.to(device))
    sar_img_cond = sar_img_cond.to(torch.bfloat16)
    sar_img_cond = rearrange(sar_img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if sar_img_cond.shape[0] == 1 and bs > 1:
        sar_img_cond = repeat(sar_img_cond, "1 ... -> bs ...", bs=bs)
    
    sar_img_cond_ids = torch.zeros(height // 2, width // 2, 3) # what is height and width here?
    sar_img_cond_ids[..., 0] = 1
    sar_img_cond_ids[..., 1] = sar_img_cond_ids[..., 1] + torch.arange(height // 2)[:, None]
    sar_img_cond_ids[..., 2] = sar_img_cond_ids[..., 2] + torch.arange(width // 2)[None, :]
    sar_img_cond_ids = repeat(sar_img_cond_ids, "h w c -> b (h w) c", b=bs)
    return_dict["sar_img_cond_seq"] = sar_img_cond
    return_dict["sar_img_cond_seq_ids"] = sar_img_cond_ids.to(device)
    return_dict["sar_img_cond_orig"] = sar_img_cond_orig



    cloud_img_cond = Image.open(cloud_img_path).convert("RGB")
    cloud_img_cond = cloud_img_cond.resize((target_height, target_width), Image.Resampling.LANCZOS)
    cloud_img_cond = np.array(cloud_img_cond)
    cloud_img_cond = torch.from_numpy(cloud_img_cond).float() / 127.5 - 1.0
    cloud_img_cond = rearrange(cloud_img_cond, "h w c -> 1 c h w")
    cloud_img_cond_orig = cloud_img_cond.clone()
    with torch.no_grad():
        cloud_img_cond = ae.encode(cloud_img_cond.to(device))
    cloud_img_cond = cloud_img_cond.to(torch.bfloat16)
    cloud_img_cond = rearrange(cloud_img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if cloud_img_cond.shape[0] == 1 and bs > 1:
        cloud_img_cond = repeat(cloud_img_cond, "1 ... -> bs ...", bs=bs)

    cloud_img_cond_ids = torch.zeros(height // 2, width // 2, 3) # what is height and width here?
    cloud_img_cond_ids[..., 0] = 1
    cloud_img_cond_ids[..., 1] = cloud_img_cond_ids[..., 1] + torch.arange(height // 2)[:, None]
    cloud_img_cond_ids[..., 2] = cloud_img_cond_ids[..., 2] + torch.arange(width // 2)[None, :]
    cloud_img_cond_ids = repeat(cloud_img_cond_ids, "h w c -> b (h w) c", b=bs)
    return_dict["cloud_img_cond_seq"] = cloud_img_cond
    return_dict["cloud_img_cond_seq_ids"] = cloud_img_cond_ids.to(device)
    return_dict["cloud_img_cond_orig"] = cloud_img_cond_orig


    cloud_img_mask = Image.open(cloud_mask_path).convert("RGB")
    cloud_img_mask = cloud_img_mask.resize((target_height, target_width), Image.Resampling.LANCZOS)
    cloud_img_mask = np.array(cloud_img_mask)
    cloud_img_mask = torch.from_numpy(cloud_img_mask).float() / 127.5 - 1.0
    cloud_img_mask = rearrange(cloud_img_mask, "h w c -> 1 c h w")
    
    cloud_img_mask = cloud_img_mask.to(torch.bfloat16)
    if cloud_img_mask.shape[0] == 1 and bs > 1:
        cloud_img_mask = repeat(cloud_img_mask, "1 ... -> bs ...", bs=bs)
    return_dict["cloud_img_mask"] = cloud_img_mask.to(device)
    return_dict["vec"] = cloud_img_mask.to(device)


    img = get_noise(
        1,
        target_height,
        target_width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)
    bs, c, h, w = img.shape
    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    return_dict["img"] = img
    return_dict["img_ids"] = img_ids.to(img.device)
    return return_dict, target_height, target_width

def denoise_train(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    sar_img_cond_seq: Tensor,
    sar_img_cond_seq_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    cloudy_img_cond_seq: Tensor | None = None,
    cloudy_img_cond_seq_ids: Tensor | None = None
):
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    img_input = img
    img_input_ids = img_ids
    if img_cond is not None:
        img_input = torch.cat((img, img_cond), dim=-1)
    if cloudy_img_cond_seq is not None:
        assert (
            cloudy_img_cond_seq_ids is not None
        ), "You need to provide either both or neither of the sequence conditioning"
        img_input = torch.cat((img_input, cloudy_img_cond_seq), dim=1)
        img_input_ids = torch.cat((img_input_ids, cloudy_img_cond_seq_ids), dim=1)
    pred = model(
        img=img_input,
        img_ids=img_input_ids,
        txt=sar_img_cond_seq,
        txt_ids=sar_img_cond_seq_ids,
        y=vec,
        timesteps=timesteps,
        guidance=guidance_vec,
    )
    if img_input_ids is not None:
        pred = pred[:, : img.shape[1]]

    return pred

def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    sar_img_cond_seq: Tensor,
    sar_img_cond_seq_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    cloudy_img_cond_seq: Tensor | None = None,
    cloudy_img_cond_seq_ids: Tensor | None = None,
):
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if cloudy_img_cond_seq is not None:
            assert (
                cloudy_img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, cloudy_img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, cloudy_img_cond_seq_ids), dim=1)
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=sar_img_cond_seq,
            txt_ids=sar_img_cond_seq_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img

def main(
    model,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 30,
    guidance: float = 2.5,
    sar_img_cond_path: str = "input",
    cloud_mask_path: str = "mask",
    cloud_img_path: str = "cloud_input",
    output_dir: str = "output",
    seed: int = 42,
    width: int | None = None,
    height: int | None = None,
):
    torch_device = torch.device(device)
    inp, height, width = prepare_kontext(
        sar_ae=sar_ae,
        ae=ae,
        sar_img_cond_path=sar_img_cond_path,
        cloud_mask_path=cloud_mask_path,
        cloud_img_path=cloud_img_path,
        target_width=width,
        target_height=height,
        bs=1,
        seed=seed,
        device=torch_device,
    )
    timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=True)
    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)
    x = unpack(x.float(), height, width)
    x = ae.decode(x)
    return x

