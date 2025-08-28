#################################################################################
#                          Import Packages                                      #
#################################################################################

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from models import DiT_models
from diffusion import create_diffusion
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from init_ddp import init_distributed_mode
import json


class MyDataset(Dataset):
    def __init__(self, data, target):#, mean, std):
        #self.data = (data - mean) / std
        self.data = data
        self.target = target
    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).float()
        y = torch.tensor(self.target[index], dtype=torch.long)
        return x, y
    def __len__(self):
        return len(self.data)
    

def update_ema(ema_model, model, decay=0.9999):
    with torch.no_grad():
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(log_path=None):
    if dist.get_rank() == 0:
        os.makedirs(log_path, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_path, "train.log"))
            ]
        )
        return logging.getLogger(__name__)
    else:
        return logging.getLogger("null")


#################################################################################
#                          Training Loop                                        #
#################################################################################

def main(args):
    #dist.init_process_group("nccl")
    init_distributed_mode()
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed * dist.get_world_size() + rank)

    if rank == 0:
        experiment_path = os.path.join(args.results_dir, f"{args.model.replace('/', '-')}")
        os.makedirs(experiment_path, exist_ok=True)
    else:
        experiment_path = None

    logger = create_logger(experiment_path if rank == 0 else None)

    model = DiT_models[args.model](input_size=args.image_size, num_classes=args.num_classes)
    cur_epoch = args.cur_epoch
    if args.checkpoint is not None:
        ckpt = torch.load(args.resume_from_ckpt)
        model.load(ckpt['model'])
    model = DDP(model.to(device), device_ids=[rank])
    ema = deepcopy(model.module).to(device)
    requires_grad(ema, False)

    diffusion = create_diffusion(timestep_respacing="")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if args.checkpoint is not None:
        opt.load(ckpt["opt"])


    #ds_train = xr.open_dataset("/fast/project/HFMI_HClimRep/nishant.kumar/dit_hackathon/data/2011_t2m_era5_2deg.nc")
    ds_train = xr.open_dataset("./data/2011_t2m_era5_2deg.nc")
    x_train = ds_train['t2m'].values
    #t2m_mean = float(x_train.mean())
    #t2m_std = float(x_train.std())
    #with open("t2m_norm_stats.json", "w") as f:
        #json.dump({"mean": t2m_mean, "std": t2m_std}, f)

    y_train = np.zeros(len(x_train))

    train_dataset = MyDataset(x_train, y_train)#, mean=t2m_mean, std=t2m_std)

    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=dist.get_world_size(), 
        rank=rank, 
        shuffle=True, 
        seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(args.global_batch_size // dist.get_world_size()), 
        sampler=train_sampler, 
        num_workers=4, 
        drop_last=True     
    )

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    logger.info("Starting training...")
    steps, running_loss, start = 0, 0, time()

    for epoch in range(args.cur_epoch,args.epochs):
        train_sampler.set_epoch(epoch)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, dict(y=y))
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            steps += 1

            if steps % args.log_every == 0:
                torch.cuda.synchronize()
                avg_loss = torch.tensor(running_loss / args.log_every, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss /= dist.get_world_size()
                logger.info(f"Epoch {epoch} Step {steps}: Loss = {avg_loss:.4f}, Steps/sec = {args.log_every / (time() - start):.2f}")
                running_loss, start = 0, time()

            if steps % args.ckpt_every == 0 and rank == 0:
                ckpt_path = os.path.join(experiment_path, f"ckpt_{steps:07d}.pt")
                torch.save({"model": model.module.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict()}, ckpt_path)
                logger.info(f"Checkpoint saved to {ckpt_path}")
            dist.barrier()

    model.eval()
    logger.info("Training complete.")
    cleanup()

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-path", type=str, required=False, default="")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, default=(90,180))
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--cur_epoch", type=int, default=0)
    parser.add_argument("--resume_from_ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
