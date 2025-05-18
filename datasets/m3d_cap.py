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
# from utilities.prepocess_m3d.create_csv_for_raw_m3d import get_images_resolution

from datasets import collate_fn
from datasets.text_utils import SentBasedTokenizer, custom_split

class M3DCAPRawDataset(Dataset):

    def __init__(self, data_dir, info_filepath, transforms, mode="train"):
        
        self.data_dir = data_dir
        self.mode = mode
        self.transforms = transforms
        
        self.info_data = pd.read_csv(info_filepath)
        self.info_data = self.info_data[self.info_data["case_id"]=="ct_case_09"]
        # print(len(self.info_data), self.info_data.columns)

        self.images_dir_path_list = self.info_data["images_dir_path"].to_list()
        # print(len(self.images_dir_path_list), self.images_dir_path_list[:2])

        self.study_dir_path_list = self.info_data["study_dir_path"].to_list()
        # print(len(self.study_dir_path_list), self.study_dir_path_list[:2])

        self.depth_list = self.info_data["Depth"].to_list()
        self.height_list = self.info_data["Height"].to_list()
        self.width_list = self.info_data["Width"].to_list()
        # print(len(self.depth_list), self.depth_list[:2])

        self.text_filename = "text.txt"

        ###======== get image-related transforms ========###
        # 1. input dtype torch.uint8
        # 2. make sub-volume consistent

        # mt_transforms = MonaiTransforms()
        # self.transforms = mt_transforms.load_ssl_transforms(mode=mode)


    def __getitem__(self, idx):

        return_dict = {}
        
        ###======== get related path ========###
        study_dir_path = self.study_dir_path_list[idx]
        images_dir_path = self.images_dir_path_list[idx]

        # print(images_dir_path)

        # text_filepath = study_dir_path + self.text_filename
        text_filepath = os.path.join(study_dir_path, self.text_filename)

        ###======== process images into 3d volume ========###
        depth, height, width = self.depth_list[idx], self.height_list[idx], self.width_list[idx]
        volume_tensor = self.process_images(images_dir_path, (depth, height, width)) # [C, H, W, D]
        # print(volume_tensor.shape, volume_tensor.max(), volume_tensor.float().mean(), volume_tensor.dtype)
        # save_image((volume_tensor.permute(3, 0, 1, 2))/255., "./check_data/volume_raw.jpg")

        volume_tensor_transformed = self.transforms(volume_tensor) # a num_samples list of [C, H, W, D]

        return_dict["subvolumes"] = volume_tensor_transformed
        # print(return_dict["subvolumes"][0].size())

        # if isinstance(volume_tensor_transformed, list):
        #     volume_tensor_transformed = volume_tensor_transformed[0]
        # print(volume_tensor_transformed.shape, volume_tensor_transformed.max(), volume_tensor_transformed.mean(), volume_tensor_transformed.dtype)
        # save_image(volume_tensor_transformed.permute(3, 0, 1, 2), "./check_data/volume_tf.jpg")


        return return_dict

        ###======== process text ========###


    def process_images(self, images_dir_path, spatial_size):

        # depth, height, width = spatial_size
        # print(spatial_size, spatial_size[1])

        ###======== read all individual slices ========###
        # for root_path, dirnames, filenames in os.walk(images_dir_path):
        #     # print("root_path", root_path)
        #     # print("dirs", dirs): A list of subdirectory names in the current root_path
        #     # print("filenames", filenames)
            
        #     if dirnames != []:
        #         continue

        #     sorted_filenames = sorted(filenames, key=lambda x: int(x.split(".")[0]))
        #     # print(sorted_filenames)

        #     volume_np = []

        images_dir_path = Path(images_dir_path)
        filepath_list = [filepath for filepath in images_dir_path.rglob("*") if filepath.is_file()]
        sorted_filepath_list = sorted(filepath_list, key=lambda x: int(x.stem))
        # print(sorted_filepath_list)

        volume_np = []
        
        for slice_filepath in sorted_filepath_list:
            # for filename in sorted_filenames:
            #     slice_filepath = os.path.join(root_path, filename)

            try:
                single_slice_np = cv2.imread(slice_filepath, cv2.IMREAD_UNCHANGED)

                if single_slice_np is None:
                    continue

                height, width = single_slice_np.shape[:2]

                if height != spatial_size[1] or width != spatial_size[2]:
                    continue

                if len(single_slice_np.shape) == 2:
                    single_slice_np = single_slice_np[..., np.newaxis]

                elif len(single_slice_np.shape) == 3:

                    if single_slice_np.shape[-1] == 3:
                        single_slice_np = cv2.cvtColor(single_slice_np, cv2.COLOR_BGR2GRAY)
                        single_slice_np = single_slice_np[..., np.newaxis]

                    elif single_slice_np.shape[-1] == 4:
                        single_slice_np = cv2.cvtColor(single_slice_np, cv2.COLOR_BGRA2BGR)
                        single_slice_np = cv2.cvtColor(single_slice_np, cv2.COLOR_BGR2GRAY)
                        single_slice_np = single_slice_np[..., np.newaxis]

                    elif single_slice_np.shape[-1] == 1:
                        pass

                if single_slice_np.shape != (spatial_size[1], spatial_size[2], 1):
                    print(f"Warning: {slice_filepath} final shape={single_slice_np.shape}, "
                        f"expected ({spatial_size[1]}, {spatial_size[2]}, 1). Skipping.")
                    continue

                volume_np.append(single_slice_np)

            except:
                print(f"Failed to read {slice_filepath}")

            
        if volume_np:
            volume_np = np.stack(volume_np) # volume_np [D, H, W, C]
        else:
            volume_np = np.random.randint(0, 256, (128, 512, 512, 1))
            print("Empty volume_np from images_dir_path:", images_dir_path)
        
        # volume_tensor = torch.from_numpy(volume_np).permute(0, 3, 1, 2) # volume_tensor [D, C, H, W] for check
        # save_image(volume_tensor/255., "./check_data/ctcase_volume.jpg")

        volume_tensor = torch.from_numpy(volume_np).permute(3, 1, 2, 0) # volume_tensor [C, H, W, D] for return

        return volume_tensor # volume_tensor [C, H, W, D]


    def __len__(self):
        return len(self.info_data)




class M3DCAPLMDBDataset(Dataset):

    def __init__(self, lmdb_path, info_filepath, transforms=None):

        self.env = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        # self.key_list = []
        # with self.env.begin(write=False) as txn:
        #     for key, _ in tqdm(txn.cursor()):
        #         self.key_list.append(key.decode())

        self.info_data = pd.read_csv(info_filepath)
        self.info_data = self.info_data[self.info_data["case_id"]=="ct_case_00"]
        images_dir_list = self.info_data["images_dir_path"].to_list()
        self.key_list = self.get_key_list(images_dir_list)

        self.transforms = transforms


    def get_key_list(self, images_dir_list):
        key_list = []
        for images_dir in images_dir_list:
            images_dir = Path(images_dir)
            case_id = images_dir.parent.parent.stem
            study_id = images_dir.parent.stem
            imaging_id = images_dir.stem
            key_stem = f"{case_id}-{study_id}-{imaging_id}"
            key_list.append(f"{key_stem}-volume")
        return key_list
                

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):

        return_dict = {}

        with self.env.begin(write=False) as txn:
            key_volume = self.key_list[idx]

            ### get data
            volume_bytes = txn.get(key_volume.encode())
            buffer = io.BytesIO(volume_bytes)
            volume_npz = np.load(buffer)
            volume_np = volume_npz["arr"]
            # print(volume_np.shape, volume_np.dtype, volume_np.mean())

        volume_tensor = torch.from_numpy(volume_np) # (C, H, W, D)
        # save_image((volume_tensor.permute(3, 0, 1, 2))/255., "./check_data/volume_raw.jpg")

        volume_tensor_transformed = self.transforms(volume_tensor)
        # save_image(volume_tensor_transformed[0].permute(3, 0, 1, 2), "./check_data/volume_tf.jpg")

        return_dict["subvolumes"] = volume_tensor_transformed



        return return_dict



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

    data_dir = "/home/olg7848/p32335/M3D_CAP/ct_case"
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



