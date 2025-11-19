import os
import re
import time
import math
import numpy as np
from dataclasses import dataclass
from PIL import Image
from typing import TYPE_CHECKING
from einops import rearrange, repeat
from torch import Tensor
from typing import Callable

class SamplingOptions:
    prompt: str
    width: int | None
    height: int | None
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str

# Safe torch device detection: fall back to "cpu" if torch is not available
try:
    import torch
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch_device = "cpu"

# Default values for variables referenced below; replace with real values as needed
name = "default_flow_model"
offload = False

def load_t5(device, max_length=512):
    # Placeholder implementation; replace with actual T5 model loading code
    return {"model": "t5", "device": device, "max_length": max_length}

def load_clip(device):
    # Placeholder implementation; replace with actual CLIP model loading code
    return {"model": "clip", "device": device}

def load_flow_model(model_name, device="cpu"):
    # Placeholder implementation; replace with actual flow model loading code
    return {"model": model_name, "device": device}

def load_ae(model_name, device="cpu"):
    # Placeholder implementation; replace with actual autoencoder loading code
    return {"model": model_name, "device": device}

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

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/ar <width>:<height>' will set the aspect ratio of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/ar"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, ratio_prompt = prompt.split()
            if ratio_prompt == "auto":
                options.width = None
                options.height = None
                print("Setting resolution to input image resolution.")
            else:
                options.width, options.height = aspect_ratio_to_height_width(ratio_prompt)
                print(f"Setting resolution to {options.width} x {options.height}.")
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            if height == "auto":
                options.height = None
            else:
                options.height = 16 * (int(height) // 16)
            if options.height is not None and options.width is not None:
                print(
                    f"Setting resolution to {options.width} x {options.height} "
                    f"({options.height * options.width / 1e6:.2f}MP)"
                )
            else:
                print(f"Setting resolution to {options.width} x {options.height}.")
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options


def parse_img_cond_path(options: SamplingOptions | None) -> SamplingOptions | None:
    if options is None:
        return None

    user_question = "Next input image (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write a path to an image directly, leave this field empty "
        "to repeat the last input image or write a command starting with a slash:\n"
        "- '/q' to quit\n\n"
        "The input image will be edited by FLUX.1 Kontext creating a new image based"
        "on your instruction prompt."
    )

    while True:
        img_cond_path = input(user_question)

        if img_cond_path.startswith("/"):
            if img_cond_path.startswith("/q"):
                print("Quitting")
                return None
            else:
                if not img_cond_path.startswith("/h"):
                    print(f"Got invalid command '{img_cond_path}'\n{usage}")
                print(usage)
            continue

        if img_cond_path == "":
            break

        if not os.path.isfile(img_cond_path) or not img_cond_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            print(f"File '{img_cond_path}' does not exist or is not a valid image file")
            continue

        options.img_cond_path = img_cond_path
        break

    return options

def aspect_ratio_to_height_width(aspect_ratio: str, area: int = 1024**2) -> tuple[int, int]:
    width = float(aspect_ratio.split(":")[0])
    height = float(aspect_ratio.split(":")[1])
    ratio = width / height
    width = round(math.sqrt(area * ratio))
    height = round(math.sqrt(area / ratio))
    return 16 * (width // 16), 16 * (height // 16)

def prepare_kontext(
    t5: HFEmbedder,
    clip: HFEmbedder,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    seed: int,
    device: torch.device,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
) -> tuple[dict[str, Tensor], int, int]:
    # load and encode the conditioning image
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    width, height = img_cond.size
    aspect_ratio = width / height
    # Kontext is trained on specific resolutions, using one of them is recommended
    _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
    width = 2 * int(width / 16)
    height = 2 * int(height / 16)

    img_cond = img_cond.resize((8 * width, 8 * height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")
    img_cond_orig = img_cond.clone()

    with torch.no_grad():
        img_cond = ae.encode(img_cond.to(device))

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    # image ids are the same as base image with the first dimension set to 1
    # instead of 0
    img_cond_ids = torch.zeros(height // 2, width // 2, 3)
    img_cond_ids[..., 0] = 1
    img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(height // 2)[:, None]
    img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(width // 2)[None, :]
    img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)

    if target_width is None:
        target_width = 8 * width
    if target_height is None:
        target_height = 8 * height

    img = get_noise(
        1,
        target_height,
        target_width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond_seq"] = img_cond
    return_dict["img_cond_seq_ids"] = img_cond_ids.to(device)
    return_dict["img_cond_orig"] = img_cond_orig
    return return_dict, target_height, target_width

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img

def main(
    name: str = "flux-dev-kontext",
    aspect_ratio: str | None = None,
    seed: int | None = None,
    prompt: str = "replace the logo with the text 'Black Forest Labs'",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 30,
    loop: bool = False,
    guidance: float = 2.5,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/cup.png",
):
    t5 = load_t5(torch_device, max_length=512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    width, height = aspect_ratio_to_height_width(aspect_ratio)
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        img_cond_path=img_cond_path,
    )
    opts = parse_prompt(opts)
    opts = parse_img_cond_path(opts)
    inp, height, width = prepare_kontext(
        t5=t5,
        clip=clip,
        prompt=opts.prompt,
        ae=ae,
        img_cond_path=opts.img_cond_path,
        target_width=opts.width,
        target_height=opts.height,
        bs=1,
        seed=opts.seed,
        device=torch_device,
    )
    timesteps = torch.linspace(1, 0, num_steps + 1)
    mu = get_lin_function(y1=0.5, y2=1.15)(inp["img"].shape[1])
    timesteps = time_shift(mu, 1.0, timesteps)
    timesteps = timesteps.tolist()

    x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)
if __name__ == "__main__":
    main()