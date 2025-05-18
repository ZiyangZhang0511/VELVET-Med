import os
import yaml
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from modeling.vqa_model import VQAModel
from modeling.vqa_baseline import VQABaseline

from datasets import collate_fn
from datasets.m3d_vqa import M3DVQADataset, M3DVQABaselineDataset

from utilities.constants import ConfigPath, DataPath
from utilities.vision_transforms import MonaiTransforms


def get_args():
    parser = argparse.ArgumentParser()  
    
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints_dt/rep_allitc_mmssl_visssl_txtssl_BtSv/report_gen-m3d_cap-dr0.3-FvisTruetxtTruemmFalse-E4/pytorch_model.bin")
    parser.add_argument("--config_path", type=str, default="./config_dt/rep_allitc_mmssl_visssl_txtssl_BtSv.yaml")

    parser.add_argument("--save_dir", type=str, default="./check_data")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args

def get_dataloader(config_path):
    g = torch.Generator()
    g.manual_seed(4)

    data_root = os.path.abspath(DataPath.M3D_CAP)
    data_dir = os.path.join(data_root, "nii_down")
    csv_dir = os.path.abspath(DataPath.M3D_VQA)
    
    mt_transforms = MonaiTransforms(num_samples=1)

    if "baseline" not in config_path:
        global_transforms = mt_transforms.load_vqa_transforms(mode="global")
        dataset = M3DVQADataset(data_dir, csv_dir, global_transforms, data_ratio=0.1, task_type="report_gen", mode="train", config_path=config_path)
    else:
        print("vqa_baseline")
        global_transforms = mt_transforms.load_vqa_transforms(mode="clip3d")
        dataset = M3DVQABaselineDataset(data_dir, csv_dir, global_transforms, data_ratio=0.1, task_type="report_gen", mode="train", config_path=config_path)
    
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_fn, shuffle=True, generator=g)

    return dataloader


def main():
    args = get_args()

    ### create model ###
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    if "baseline" not in args.config_path:
        model = VQAModel(config).to(args.device)
    else:
        print("vqa_baseline")
        model = VQABaseline(config).to(args.device)
    
    ### load weights ###
    ckpt = torch.load(args.ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)

    ### get dataloader ###
    dataloader = get_dataloader(args.config_path)
    first_batch = next(iter(dataloader))
    for key in first_batch.keys():
        if key not in "answer":
            first_batch[key] = first_batch[key].to(args.device)
    
    
    re = model.test_one_step(first_batch)
    output_dict = re[1]
    print("pred", output_dict["pred_captions_batch"])
    print("gt", output_dict["gt_captions_batch"])


if __name__ == "__main__":
    main()