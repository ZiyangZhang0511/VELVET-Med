import os
import json
import shutil

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    Dataset as mDataset,
    DataLoader as mDataLoader,
    load_decathlon_datalist
)

from .constants import DataPath
from .seg_utils import load_seg_data_list
from .vision_transforms import MonaiTransforms

from datasets import collate_fn
from datasets.m3d_cap import M3DCAPDataset
from datasets.m3d_vqa import M3DVQADataset
from datasets.tcia_covid19 import TCIACOVID19Dataset


def count_params(model):
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
    return num_trainable_params
    # print(f"Number of trainable parameters: {num_trainable_params:.2f}M")


def remove_directory(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Error occurred while removing the directory: {e}")

def get_dt_dataset(args):

    if args.dataset_name == "m3d_vqa" and args.downstream_type in ["close_vqa_yn", "open_vqa_yn", "close_vqa_mc", "open_vqa_mc", "report_gen"]:
        data_root = os.path.abspath(DataPath.M3D_CAP)
        data_dir = os.path.join(data_root, "nii_down")
        csv_dir = os.path.abspath(DataPath.M3D_VQA)
        
        mt_transforms = MonaiTransforms(num_samples=1)
        global_transforms = mt_transforms.load_vqa_transforms(mode="global")

        train_dataset = M3DVQADataset(data_dir, csv_dir, global_transforms, data_ratio=args.data_ratio, task_type=args.downstream_type, mode="train", config_path=args.config_path)
        val_dataset = M3DVQADataset(data_dir, csv_dir, global_transforms, data_ratio=args.data_ratio, task_type=args.downstream_type, mode="val", config_path=args.config_path)

    elif args.dataset_name == "m3d_vqa" and args.downstream_type in ["cls_vqa_yn"]:
        data_root = os.path.abspath(DataPath.M3D_CAP)
        data_dir = os.path.join(data_root, "nii_down")
        csv_dir = os.path.abspath(DataPath.M3D_VQA)
        
        mt_transforms = MonaiTransforms(num_samples=1)
        global_transforms = mt_transforms.load_vqa_transforms(mode="global")

        train_dataset = M3DVQADataset(data_dir, csv_dir, global_transforms, data_ratio=args.data_ratio, task_type="cls_vqa_yn", mode="train", config_path=args.config_path)
        val_dataset = M3DVQADataset(data_dir, csv_dir, global_transforms, data_ratio=args.data_ratio, task_type="cls_vqa_yn", mode="val", config_path=args.config_path)

    elif args.dataset_name in ["btcv", "abdomenct1k", "ctorg", "totalsegmentator"] and args.downstream_type in ["seg"]:
        train_list, val_list, test_list = load_seg_data_list(args.dataset_name)

        transforms = MonaiTransforms(num_samples=args.num_samples_per_volume)
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

    # elif args.dataset_name == "abdomenct1k" and args.downstream_type in ["seg"]:
    #     train_list, val_list, test_list = load_seg_data_list(args.dataset_name)

    #     transforms = MonaiTransforms(num_samples=args.num_samples_per_volume)
    #     train_mt = transforms.load_supervised_seg_transforms("train")
    #     val_mt = transforms.load_supervised_seg_transforms("val")

    #     train_dataset = CacheDataset(
    #         data=train_list,
    #         transform=train_mt,
    #         cache_num=24,
    #         cache_rate=1.0,
    #         num_workers=8,
    #     )
    #     val_dataset = CacheDataset(
    #         data=val_list,
    #         transform=val_mt,
    #         cache_num=6,
    #         cache_rate=1.0,
    #         num_workers=4,
    #     )
    return train_dataset, val_dataset

def get_dt_dataloader(train_dataset, val_dataset, args):

    if args.dataset_name in ["tcia_covid19"]:
        train_dataloader = mDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        val_dataloader = mDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    elif args.dataset_name in ["m3d_cap", "m3d_vqa"] or args.downstream_type in ["close_vqa_yn", "open_vqa_yn", "close_vqa_mc", "open_vqa_mc", "report_gen"]:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=8,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

    elif args.dataset_name in ["abdomenct1k", "btcv", "ctorg", "totalsegmentator"] and args.downstream_type in ["seg"]:
        train_dataloader = ThreadDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dataloader = ThreadDataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    return train_dataloader, val_dataloader



def get_ssl_dataset(args):

    if args.dataset_name == "tcia_covid19" and args.pretrain_type == "vitautoenc_ssl":

        mt_transforms = MonaiTransforms(num_samples=args.num_samples_per_volume)
        transforms = mt_transforms.load_ssl_transforms(args)

        data_root = os.path.abspath(DataPath.TCIA_COVID19)
        json_path = os.path.join(data_root, "dataset_TCIAcovid19_0.json")

        with open(json_path, "r") as json_f:
            json_data = json.load(json_f)

        train_datalist = json_data["training"]
        val_datalist = json_data["validation"]

        for idx, _each_d in enumerate(train_datalist):
            train_datalist[idx]["image"] = os.path.join(data_root, train_datalist[idx]["image"])

        for idx, _each_d in enumerate(val_datalist):
            val_datalist[idx]["image"] = os.path.join(data_root, val_datalist[idx]["image"])

        # print("Total Number of Training Data Samples: {}".format(len(train_datalist)))
        # print(train_datalist)
        # print("#" * 10)
        # print("Total Number of Validation Data Samples: {}".format(len(val_datalist)))
        # print(val_datalist[:2])
        # print("#" * 10)

        train_dataset = mDataset(data=train_datalist, transform=transforms)
        val_dataset = mDataset(data=val_datalist, transform=transforms)

    elif args.dataset_name == "tcia_covid19" and args.pretrain_type == "vis_ssl":

        mt_transforms = MonaiTransforms(num_samples=args.num_samples_per_volume)
        transforms = mt_transforms.load_ssl_transforms(args)
        
        train_dataset = TCIACOVID19Dataset(transforms, mode="train")
        val_dataset = TCIACOVID19Dataset(transforms, mode="val")


    elif args.dataset_name == "m3d_cap" and args.pretrain_type == "vis_ssl":

        data_root = os.path.abspath(DataPath.M3D_CAP)
        data_dir = os.path.join(data_root, "nii_down")
        json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")

        mt_transforms = MonaiTransforms(num_samples=args.num_samples_per_volume)
        transforms = mt_transforms.load_ssl_transforms(args)
        global_transforms = mt_transforms.load_ssl_transforms(args, mode="global")
        
        train_dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=args.data_ratio, mode="train", pretrained_type="vis_ssl")
        val_dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=args.data_ratio, mode="val", pretrained_type="vis_ssl")
    
    elif args.dataset_name == "m3d_cap" and args.pretrain_type == "txt_ssl":

        data_root = os.path.abspath(DataPath.M3D_CAP)
        data_dir = os.path.join(data_root, "nii_down")
        json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")
        
        transforms = None
        
        train_dataset = M3DCAPDataset(data_dir, json_filepath, transforms, data_ratio=args.data_ratio, mode="train", pretrained_type="txt_ssl")
        val_dataset = M3DCAPDataset(data_dir, json_filepath, transforms, data_ratio=args.data_ratio, mode="val", pretrained_type="txt_ssl")
    
    elif args.dataset_name == "m3d_cap" and args.pretrain_type == "vl_ssl":

        data_root = os.path.abspath(DataPath.M3D_CAP)
        data_dir = os.path.join(data_root, "nii_down")
        json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")
        
        mt_transforms = MonaiTransforms(num_samples=args.num_samples_per_volume)
        transforms = mt_transforms.load_ssl_transforms(args)
        global_transforms = mt_transforms.load_ssl_transforms(args, mode="global")

        train_dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=args.data_ratio, mode="train", pretrained_type="vl_ssl", config_path=args.config_path)
        val_dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=args.data_ratio, mode="val", pretrained_type="vl_ssl", config_path=args.config_path)

    elif args.dataset_name == "m3d_cap" and args.pretrain_type == "clip3d_ssl":

        data_root = os.path.abspath(DataPath.M3D_CAP)
        data_dir = os.path.join(data_root, "nii_down")
        json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")

        mt_transforms = MonaiTransforms(num_samples=args.num_samples_per_volume)
        transforms = mt_transforms.load_ssl_transforms(args)
        global_transforms = mt_transforms.load_ssl_transforms(args, mode="clip3d")

        train_dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=args.data_ratio, mode="train", pretrained_type="vl_ssl", text_model_type="bert")
        val_dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=args.data_ratio, mode="val", pretrained_type="vl_ssl", text_model_type="bert")


    return train_dataset, val_dataset

def get_dataloader(train_dataset, val_dataset, args):

    if args.dataset_name in ["tcia_covid19"]:
        train_dataloader = mDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        val_dataloader = mDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    elif args.dataset_name in ["m3d_cap"] or args.pretrain_type in ["vl_ssl", "vis_ssl", "txt_ssl"]:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=8,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

    
    return train_dataloader, val_dataloader


def visualize_transformed_data(dataset, case_num, slice_map, save=False):

    if isinstance(dataset[case_num], dict):
        data_sample = dataset[case_num]
    elif isinstance(dataset[case_num], list):
        data_sample = dataset[case_num][0]
    # print(data_sample["image"].meta)

    if isinstance(data_sample["image"], torch.Tensor):
        img_name = list(slice_map.keys())[case_num]
    elif not data_sample["image"].meta.get("filename_or_obj", None):
        img_name = list(slice_map.keys())[case_num]
    else:
        img_name = os.path.split(data_sample["image"].meta["filename_or_obj"])[1]

    img = data_sample["image"]
    # label = data_sample["label"]
    print(f"image shape: {img.shape}")
    print(f"image max: {img.max()}, image min: {img.min()}, image mean: {img.mean()}")
    # print(f"label max: {label.max()}, label min: {label.min()}, label mean: {label.mean()}")

    if save:
        save_image(img.permute(3, 0, 1, 2), "./check_data/image.jpg")

    # plt.figure("image", (12, 6))

    # plt.subplot(1, 2, 1)
    # plt.title("image")
    # plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")

    # # plt.subplot(1, 2, 2)
    # # plt.title("label")
    # # plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())

    # if save:
    #     plt.savefig("./check_data/image.jpg", format="jpg", bbox_inches="tight", dpi=300)
    # else:
    #     plt.show()


def print_dict_content(dictionary, header=" ", accelerator=None):

    if accelerator:
        accelerator.print(f"{header}", end="\n")
        for key, value in dictionary.items():
            if value:
                accelerator.print(f"{key}: {value:2.4f},", end="\n")
            else:
                accelerator.print(f"{key}: {value},", end="\n")
        # accelerator.print()
    
    else:
        print(f"{header}", end="\n")
        for key, value in dictionary.items():
            if value:
                print(f"{key}: {value:2.4f},", end="\n")
            else:
                print(f"{key}: {value},", end="\n")
        # print()


@torch.no_grad()
def test_model_for_ssl(model, dataloader, args, mode="val"):

    epoch_loss_dict = {}
    metric_dict = {}

    for i, batch_data in enumerate(dataloader):
        loss_dict, output_dict = model.test_one_step(batch_data)

        for loss_name, loss_value in loss_dict.items():
            if loss_name not in epoch_loss_dict.keys():
                epoch_loss_dict[loss_name] = 0
            epoch_loss_dict[loss_name] += loss_value / len(dataloader)

    print(f"{mode}", end=" ")
    for loss_name, loss_value in epoch_loss_dict.items():
        print(f"{loss_name}: {loss_value:2.5f},", end=" ")
    print()

    metric_dict["loss"] = epoch_loss_dict["loss"]

    return metric_dict

