import os
import io
import lmdb
import json
import yaml
from pathlib import Path
from tqdm.auto import tqdm

import cv2
import numpy as np
import pandas as pd
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.utils import save_image

import monai.transforms as mt

from utilities.vision_transforms import MonaiTransforms
from utilities.constants import DataPath, ConfigPath

from datasets import collate_fn
from datasets.text_utils import SentBasedTokenizer, custom_split



class M3DCAPDataset(Dataset):

    def __init__(
        self,
        data_dir,
        json_filepath,
        transforms=None,
        global_transforms=None,
        data_ratio=0.25,
        mode="train",
        pretrained_type="vl_ssl",
        text_model_type="sentbert",
        config_path="./config_ssl/text_ssl.yaml",
    ):
        
        self.data_dir = data_dir

        with open(json_filepath, "r") as json_f:
            json_data = json.load(json_f)
        data_list = json_data[mode]
        
        if mode in ["train", "val"]:
            self.data_list = data_list[:int(data_ratio*len(data_list))]
        else:
            self.data_list = data_list
        
        if pretrained_type in ["vl_ssl", "vis_ssl"]:
            self.transforms = transforms
            self.global_transforms = global_transforms
            self.image_loader = mt.LoadImage(image_only=True, ensure_channel_first=False, dtype=np.uint8)

        if pretrained_type in ["vl_ssl", "txt_ssl"]:
            config_filepath = os.path.abspath(config_path)
            with open(config_filepath, "r") as f:
                config = yaml.safe_load(f)
            if "text_ssl" in list(config.keys()):
                self.tokenizer = SentBasedTokenizer(config["text_ssl"]["tokenizer"])
            else: 
                self.tokenizer = SentBasedTokenizer(config["tokenizer"])
        
        self.pretrained_type = pretrained_type
        self.text_model_type = text_model_type


    def __getitem__(self, idx):

        return_dict = {}

        ### transform medical volume ###
        if self.pretrained_type in ["vl_ssl", "vis_ssl"]:
            nii_filepath = os.path.join(self.data_dir, self.data_list[idx]["image"])
            volume_tensor = self.image_loader(nii_filepath)
            # print(volume_tensor.meta["affine"])
            # volume_tensor.meta["affine"][2, 2] = 0.5
            # print(volume_tensor.shape, volume_tensor.dtype, (volume_tensor/1.).mean())
            # save_image((volume_tensor.permute(3, 0, 1, 2)[:16])/255., "./check_data/volume_raw.jpg", nrow=4)

            volume_tensor_transformed = self.transforms(volume_tensor)
            # print(volume_tensor_transformed[0].shape, volume_tensor_transformed[0].dtype, (volume_tensor_transformed[0]/1.).mean())
            # save_image(volume_tensor_transformed[0].permute(3, 0, 1, 2), "./check_data/volume_tf.jpg")
            # save_image(volume_tensor_transformed[1].permute(3, 0, 1, 2), "./check_data/volume_tf_.jpg")

            volume_tensor_global = self.global_transforms(volume_tensor)
            # print(volume_tensor_global.shape, volume_tensor_global.dtype, (volume_tensor_global).mean())
            # save_image((volume_tensor_global.permute(3, 0, 1, 2)[:16]), "./check_data/volume_global.jpg", nrow=4)


            return_dict["volume_vis"] = volume_tensor_transformed[0]
            return_dict["volume_vl"] = volume_tensor_global


        ### tokenize medical report ###
        if self.pretrained_type in ["vl_ssl", "txt_ssl"]:
            txt_filepath = os.path.join(self.data_dir, self.data_list[idx]["text"])
            # print(txt_filepath)
            with open(txt_filepath, "r") as f:
                text = f.read()
            sentences_list = custom_split(text)
            print(sentences_list)

            if self.text_model_type == "sentbert":
                input_ids, token_type_ids, attention_mask = self.tokenizer.encode_report(sentences_list)
            
            elif self.text_model_type == "bert":
                whole_report = "".join(sentences_list)
                encoded_dict = self.tokenizer.base_tokenizer(
                    whole_report,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.max_len_report,
                )
                input_ids = encoded_dict["input_ids"]
                token_type_ids = encoded_dict["token_type_ids"]
                attention_mask = encoded_dict["attention_mask"]

            # print(input_ids.shape, input_ids[..., :5])
            # print(token_type_ids.shape, token_type_ids[..., :5])
            # print(attention_mask.shape, attention_mask[..., :5])

            return_dict["input_ids"] = input_ids
            return_dict["token_type_ids"] = token_type_ids
            return_dict["attention_mask"] = attention_mask

        return return_dict


    def __len__(self):
        return len(self.data_list)



if __name__ == "__main__":

    data_dir = "../../M3D_CAP/ct_case"
    info_filepath = "./data/m3d_cap/ct_case_info_1.csv"
    
    mt_transforms = MonaiTransforms(num_samples=1)
    transforms = mt_transforms.load_ssl_transforms(mode="local")
    global_transforms = mt_transforms.load_ssl_transforms(mode="global")
    
    # dataset = M3DCAPRawDataset(data_dir, info_filepath, transforms)
    # dataloader = DataLoader(dataset, batch_size=2, num_workers=8, collate_fn=collate_fn)
    # for batch in tqdm(dataloader):
    #     pass


    # lmdb_dir = "/home/olg7848/p32335/my_research/vlp_ct/data/m3d_cap/lmdb"
    # lmdb_dataset = M3DCAPLMDBDataset(lmdb_dir, info_filepath, transforms)
    # # lmdb_dataset[110]
    # dataloader = DataLoader(lmdb_dataset, batch_size=2, num_workers=4, collate_fn=collate_fn)
    # for batch in tqdm(dataloader):
    #     # print(batch["subvolumes"].size())
    #     # break
    #     pass

    data_root = os.path.abspath(DataPath.M3D_CAP)
    data_dir = os.path.join(data_root, "nii_down")
    json_filepath = os.path.join(data_root, "m3d_cap_split.json")
    dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=0.05, mode="train", pretrained_type="vl_ssl", text_model_type="sentbert")
    # print(len(dataset))
    dataset[8]

    # dataloader = DataLoader(dataset, batch_size=8, num_workers=8, collate_fn=collate_fn)
    # for batch in tqdm(dataloader):
    #     # for key in batch.keys():
    #     #     print(key, batch[key].size(), batch[key].dtype, batch[key].device)
    #     pass



