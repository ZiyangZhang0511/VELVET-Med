import os
import yaml
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from torchprofile import profile_macs

from pytorch_lightning import seed_everything

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import collate_fn
from datasets.m3d_cap import M3DCAPDataset
from datasets.text_utils import SentBasedTokenizer

from modeling.visiontext_model import VisionTextSSL
from modeling.m3dcap_model import CLIP3DSSL

from utilities.constants import ConfigPath, DataPath
from utilities.vision_transforms import MonaiTransforms
from utilities.utils import count_params

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/visiontext_topitc_BtSv/vl_ssl-swinvit-m3d_cap-dr1.0-E30-best/pytorch_model.bin")
    parser.add_argument("--config_path", type=str, default="./config_ssl/visiontext_topitc_BtSv.yaml")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args

def get_dataloader(config_filepath):
    if "visiontext" in config_filepath:
        data_root = os.path.abspath(DataPath.M3D_CAP)
        data_dir = os.path.join(data_root, "nii_down")
        json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")
        
        mt_transforms = MonaiTransforms(num_samples=1)
        transforms = mt_transforms.load_ssl_transforms(mode="local")
        global_transforms = mt_transforms.load_ssl_transforms(mode="global")
        
        dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=0.1, mode="val", pretrained_type="vl_ssl", config_path=config_filepath)
        # print(len(dataset))
        
    else:
        data_root = os.path.abspath(DataPath.M3D_CAP)
        data_dir = os.path.join(data_root, "nii_down")
        json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")

        mt_transforms = MonaiTransforms(num_samples=1)
        transforms = mt_transforms.load_ssl_transforms(mode="clip3d")
        global_transforms = mt_transforms.load_ssl_transforms(mode="clip3d")

        dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=0.1, mode="val", pretrained_type="vl_ssl", text_model_type="bert")
    
    g = torch.Generator()
    g.manual_seed(1) # seed 4, 70 tokens; seed 11, 63 tokens; seed 12, 79 tokens; seed 19, 68 tokens; 
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, collate_fn=collate_fn, generator=g)
    return dataloader

def step(model, batch):
    with autocast():                # remove if FP32
        out = model(batch)         # forward
        # if training:                # backward pass if needed
        #     loss = criterion(out, targets)
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()


def main():

    args = get_args()

    ### create model ###
    if "visiontext" in args.config_path:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
        config["vision_ssl"]["vision_encoder"]["swinvit"]["use_checkpoint"] = False
        model = VisionTextSSL(config).to(args.device)
        ### load weights ###
        ckpt = torch.load(args.ckpt_path, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
    else:
        model = CLIP3DSSL().to(args.device)
    model.eval()

    ### prepare dataloader ###
    dataloader = get_dataloader(args.config_path)
    first_batch = next(iter(dataloader))
    for key in first_batch.keys():
        first_batch[key] = first_batch[key].to(args.device)

    ### calculate MACs FLOPs ###
    batch_size = first_batch["volume_vl"].shape[0]
    macs = profile_macs(model, first_batch) / batch_size / 1000000000
    flops = 2 * macs
    print(f"MACs: {macs:.2f}G, FLOPs: {flops:.2f}G")

    ### count trainable parameters ###
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
    print(f"Number of trainable parameters: {num_trainable_params:.2f}M")

    num_trainable_params = count_params(model.vision_encoder)
    print(f"Number of trainable parameters of vision encoder: {num_trainable_params:.2f}M")
    num_trainable_params = count_params(model.text_encoder)
    print(f"Number of trainable parameters of text model: {num_trainable_params:.2f}M")


    ### compute Throughput ###
    warmup, iters = 10, 100   
    #warm‑up
    for _ in range(warmup):
        step(model, first_batch)

    #timed loop
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step(model, first_batch)
    torch.cuda.synchronize()

    elapsed = time.time() - t0
    throughput = iters * batch_size / elapsed
    print(f"Throughput: {throughput:.2f} samples/s")


if __name__ == "__main__":

    main()