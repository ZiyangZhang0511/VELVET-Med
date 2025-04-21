import os
from tqdm.auto import tqdm
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR 

import numpy as np
import matplotlib.pyplot as plt

import monai
from monai.losses import DiceCELoss
import monai.transforms as mt
from monai.config import print_config
from monai.networks.nets import SwinUNETR
# from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    Dataset as mDataset,
    DataLoader as mDataLoader,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

from modeling.segmenter import load_segmenter
from utilities.vision_transforms import MonaiTransforms
from utilities.constants import SliceMap
from .utils import *

train_slice_map = {
    "img0001.nii.gz": 17,
    "img0002.nii.gz": 23,
    "img0003.nii.gz": 20,
    "img0004.nii.gz": 20,
    "img0005.nii.gz": 17,
    "img0006.nii.gz": 23,
    "img0007.nii.gz": 20,
    "img0008.nii.gz": 20,
    "img0009.nii.gz": 20,
    "img0010.nii.gz": 18,
}

val_slice_map = {
    "img0035.nii.gz": 170,
    "img0036.nii.gz": 230,
    "img0037.nii.gz": 204,
    "img0038.nii.gz": 204,
    "img0039.nii.gz": 204,
    "img0040.nii.gz": 180,
}




def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="btcv")
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--segmenter_type", default="swinunetr")
    parser.add_argument("--num_samples", default=4, type=int)

    parser.add_argument("--initial_lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--num_workers", default=8 , type=int)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--restart_optimizer", action="store_true")
    parser.add_argument("--cur_epoch", default=-1, type=int)
    parser.add_argument("--resume_pretraining", action="store_true")

    parser.add_argument("--ckpt_save_dir", default="./checkpoints/segmentation/m3d_cap-dr0.25", type=str)
    parser.add_argument("--pretrained_weights", default=None, type=str)

    args = parser.parse_args()

    return args


def main():

    args = get_args()
    device = args.device

    os.makedirs(args.ckpt_save_dir, exist_ok=True)

    if args.dataset_name == "btcv":
        slice_map = SliceMap.btcv_slice_map

    ###======== prepare dataloader ========###
    train_list, val_list, test_list = load_data_list(args.dataset_name)

    transforms = MonaiTransforms(num_samples=args.num_samples)
    train_mt = transforms.load_supervised_seg_transforms("train")
    val_mt = transforms.load_supervised_seg_transforms("val")

    train_dataset = CacheDataset(
        data=train_list,
        transform=train_mt,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    val_dataset = CacheDataset(
        data=val_list,
        transform=val_mt,
        cache_num=6,
        cache_rate=1.0,
        num_workers=4,
    )

    
    # visualize_transformed_data(train_dataset, 2, slice_map["train"], save=True)

    train_dataloader = ThreadDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = ThreadDataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    ###======== prepare model ========###
    model = load_segmenter(
        args.segmenter_type,
        args.pretrained_weights,
    )
    model.to(device)

    ###======== training loop ========###
    checkpoint_path = os.path.join(
        args.ckpt_save_dir,
        f"{args.segmenter_type}-{args.dataset_name}-finetune{args.finetune}.pth"
    )

    criterion = DiceCELoss(softmax=True, to_onehot_y=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.98), weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, len(train_dataloader)*args.num_epochs, eta_min=1e-8)

    # step = 0
    best_val_loss = 1e4
    for epoch in range(args.num_epochs):

        epoch_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc="Train loss=X.X", dynamic_ncols=True)
        model.train()
        
        for i, batch in enumerate(epoch_iterator):

            # step += 1

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits_map = model(images)

            loss = criterion(logits_map, labels)
            epoch_loss += loss.item() / len(epoch_iterator)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_iterator.set_description(
                f"Train loss={loss:2.5f}"
            )
        
        print(f"Epoch {epoch}: train loss: {epoch_loss:2.5f}")

        ###======== validating model ========###
        model.eval()
        val_metric_dict = test_model(model, val_dataloader, criterion, per_class=False, device=args.device)

        val_loss = val_metric_dict["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_metric_dict": val_metric_dict,
            }, checkpoint_path)
            print(f"saved checkpoint at epoch {epoch}.")

        if epoch % 5 == 0:
            print(val_metric_dict)

        model.eval()
        visualize_predicition(model, val_dataset, slice_map["val"], case_num=1, save=True, device=device)

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    best_metric_dict = checkpoint["val_metric_dict"]
    print("Best metric:", best_metric_dict)




if __name__ == "__main__":

    main()