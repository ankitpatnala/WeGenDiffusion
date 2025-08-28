# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import xarray as xr
from copy import deepcopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from init_ddp import init_distributed_mode
import json



def save_sample(
    sample,
    path_prefix,
    save_png=True,
    save_npy=True,
    colormap="plasma",
    lon_range=(0, 360),
    lat_range=(-90, 90),
    temp_min=None,
    temp_max=None
):
    """
    Save each climate field sample as .npy and/or projected world map .png (Kelvin scale).

    Args:
        sample (Tensor): (N, C, H, W)
        path_prefix (str): Output file path prefix.
        save_png (bool): Save PNG images with Cartopy map.
        save_npy (bool): Save raw tensor as .npy.
        colormap (str): Matplotlib colormap (e.g., 'plasma').
        lon_range (tuple): Longitude span.
        lat_range (tuple): Latitude span.
        temp_min (float): Minimum temperature for colormap scaling.
        temp_max (float): Maximum temperature for colormap scaling.
    """
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    sample = sample.cpu()

    for i in range(sample.size(0)):
        img_tensor = sample[i]  # (C, H, W)

        # Save raw tensor
        if save_npy:
            np.save(f"{path_prefix}_{i}.npy", img_tensor.numpy())

        # Save PNG with map
        if save_png:
            img_np = img_tensor.numpy()
            if img_np.shape[0] == 1:
                img_np = img_np[0]  # (H, W)
            else:
                img_np = np.mean(img_np, axis=0)  # (H, W)

            # Generate lat/lon grid
            h, w = img_np.shape
            lons = np.linspace(lon_range[0], lon_range[1], w)
            lats = np.linspace(lat_range[0], lat_range[1], h)
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            # Setup figure
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_global()
            ax.coastlines()
            ax.set_title(f"Sample {i}", fontsize=10)

            # Normalize using dataset min/max, if provided
            vmin = temp_min if temp_min is not None else img_np.min()
            vmax = temp_max if temp_max is not None else img_np.max()

            # Plot
            im = ax.pcolormesh(lon_grid, lat_grid, img_np, cmap=colormap, shading="auto", transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, orientation="horizontal", pad=0.05, aspect=50)
            cbar.set_label("K")  # Kelvin label

            # Save and close
            plt.savefig(f"{path_prefix}_{i}.png", bbox_inches='tight')
            plt.close()


def main(args):
    #dist.init_process_group("nccl")
    init_distributed_mode()
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)
     
    data_path = "/p/project1/training2533/patnala1/WeGenDiffusion/data/2011_t2m_era5_2deg.nc"
    ds = xr.open_dataset(data_path)
    mean = ds.mean()['t2m'].values
    std = ds.std()['t2m'].values


    model = DiT_models[args.model](
        input_size=args.image_size,
        num_classes=args.num_classes,
    )
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="250")  # e.g. ddim25 or 250 steps

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model"]
    if not list(state_dict.keys())[0].startswith("module."):
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    batch_size = args.batch_size
    image_size = args.image_size

    # Class labels (dummy 0 if unconditional)
    y = torch.zeros(batch_size, dtype=torch.long).to(device)

    # Sample from diffusion
    for i in range(args.num_batches):
        noise = torch.randn(batch_size, 1, image_size[0], image_size[1]).to(device)
        model_kwargs = dict(y=y)
        sample = diffusion.p_sample_loop(
            model,
            (batch_size, 1, image_size[0], image_size[1]),
            device=device,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )

        sample = sample * mean + std 
        out_path = os.path.join(args.output_dir, f"sample_rank{rank}_batch{i}.npy")
        save_sample(
            sample,
            path_prefix=out_path.replace(".npy", ""),
            save_png=True,
            save_npy=True,
            colormap="plasma",
            lon_range=(0, 360),
            lat_range=(-90, 90),
        )
        if rank == 0:
            print(f"Saved {out_path}")

    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-B/2", choices=list(DiT_models.keys()))
    parser.add_argument("--image-size", type=int, nargs=2, default=(90, 180))
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="/fast/project/HFMI_HClimRep/nishant.kumar/dit_hackathon/samples")
    args = parser.parse_args()
    main(args)
