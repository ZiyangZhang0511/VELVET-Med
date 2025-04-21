import os
import io
import json
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
from utilities.constants import DataPath

from datasets import collate_fn

class TCIACOVID19niiDataset(Dataset):

    def __init__(self, transforms=None, mode="train"):
        
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

        if mode == "train":
            self.datalist = train_datalist
        else:
            self.datalist = val_datalist

        # print(self.datalist[:2])
        self.transforms = transforms

        self.image_loader = mt.LoadImage(image_only=True, ensure_channel_first=True, dtype=np.uint8)

        self.full_transforms = mt.Compose([
            self.image_loader,
            self.transforms,
        ])


    def __getitem__(self, idx):

        return_dict = {}

        nii_filepath = self.datalist[idx]["image"]

        volume_tensor_transformed = self.full_transforms(nii_filepath)

        # volume_tensor = self.image_loader(nii_filepath)
        
        # img = nib.load(nii_filepath)
        # volume_np = img.get_fdata()
        # print(volume_np.shape, volume_np.dtype, volume_np.mean())


        # print(volume_tensor.shape, volume_tensor.dtype, (volume_tensor/1.).mean())
        # save_image((volume_tensor.permute(3, 0, 1, 2))/255., "./check_data/volume_raw.jpg")

        # volume_tensor_transformed = self.transforms(volume_tensor)
        # save_image(volume_tensor_transformed[0].permute(3, 0, 1, 2), "./check_data/volume_tf.jpg")
        # save_image(volume_tensor_transformed[1].permute(3, 0, 1, 2), "./check_data/volume_tf_.jpg")

        return_dict["subvolumes"] = volume_tensor_transformed
        return return_dict

    def __len__(self):
        return len(self.datalist)



class TCIACOVID19Dataset(Dataset):

    def __init__(self, transforms, mode="train"):

        data_root = os.path.abspath(DataPath.TCIA_COVID19)
        json_path = os.path.join(data_root, "dataset_TCIAcovid19_0.json")

        with open(json_path, "r") as json_f:
            json_data = json.load(json_f)

        if mode == "train":
            datalist = json_data["training"]
        else:
            datalist = json_data["validation"]

        for idx, _each_d in enumerate(datalist):
            datalist[idx]["image"] = os.path.join(data_root, datalist[idx]["image"])
        self.datalist = datalist

        self.transforms = transforms
        self.image_loader = mt.LoadImage(image_only=True, ensure_channel_first=True, dtype=np.uint8)

    def __getitem__(self, idx):

        return_dict = {}

        nii_filepath = self.datalist[idx]["image"]
        # print(nii_filepath)
        volume_tensor = self.image_loader(nii_filepath)
        # print(volume_tensor.meta["affine"])
        # volume_tensor.meta["affine"][2, 2] = 0.5
        # print(volume_tensor.shape, volume_tensor.dtype, (volume_tensor/1.).mean())
        # save_image((volume_tensor.permute(3, 0, 1, 2))/255., "./check_data/volume_raw.jpg")

        volume_tensor_transformed = self.transforms(volume_tensor)
        # print(volume_tensor_transformed[0].shape, volume_tensor_transformed[0].dtype, (volume_tensor_transformed[0]/1.).mean())
        # save_image(volume_tensor_transformed[0].permute(3, 0, 1, 2), "./check_data/volume_tf.jpg")
        # save_image(volume_tensor_transformed[1].permute(3, 0, 1, 2), "./check_data/volume_tf_.jpg")

        # return_dict["subvolumes"] = volume_tensor_transformed

        return_dict["volume_vis"] = volume_tensor_transformed[0]

        return return_dict
        

    def __len__(self):
        return len(self.datalist)






if __name__ == "__main__":

    mt_transforms = MonaiTransforms(num_samples=2)
    transforms = mt_transforms.load_ssl_transforms()

    transforms = mt.Compose([
        mt.Orientation(axcodes="RAS"),
        mt.ScaleIntensityRange(
            a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
        ),
        mt.Spacing(
            pixdim=(2.0, 2.0, 2.0),
            mode=("bilinear"),
            # min_pixdim=(1.0, 1.0, 1.0), 
            # max_pixdim=None,
            align_corners=False,
        ),
        mt.CropForeground(allow_smaller=False),
        mt.SpatialPad(spatial_size=[96, 96, 96]),
        mt.RandSpatialCropSamples(
            roi_size=[96, 96, 96],
            num_samples=2,
            random_center=True,
            random_size=False,
        ),
    ])

    dataset = TCIACOVID19Dataset(transforms, mode="train")
    print(len(dataset))
    # dataset[10]

    dataloader = DataLoader(dataset, batch_size=4, num_workers=8, collate_fn=collate_fn)
    for batch in tqdm(dataloader):
        # print(batch["subvolumes"].size())
        # break
        pass
