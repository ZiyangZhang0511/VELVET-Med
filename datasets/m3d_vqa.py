import os
import yaml
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import monai.transforms as mt

from transformers import LlamaTokenizer, AutoTokenizer

from utilities.vision_transforms import MonaiTransforms
from utilities.constants import DataPath, ConfigPath

from datasets import collate_fn
from datasets.text_utils import SentBasedTokenizer, custom_split


class M3DVQADataset(Dataset):

    def __init__(
        self,
        data_dir,
        csv_dir,
        # transforms=None,
        global_transforms=None,
        data_ratio=1.0,
        mode="train",
        task_type="cls_vqa_yn", # choose from "cls_vqa_yn", "close_vqa_yn", "open_vqa_yn", "close_vqa_mc", "open_vqa_mc"
        # text_model_type="sentbert",
        config_path="./config_dt/cls_vqa_yn_StSv.yaml",
    ):
        self.data_dir = data_dir
        self.task_type = task_type

        config_filepath = os.path.abspath(config_path)
        with open(config_filepath, "r") as f:
            config = yaml.safe_load(f)


        if task_type in ["cls_vqa_yn", "close_vqa_yn", "open_vqa_yn"]:
            if mode == "train":
                csv_filepath = os.path.join(csv_dir, "m3d_vqa_yn_train.csv")
            elif mode == "val":
                csv_filepath = os.path.join(csv_dir, "m3d_vqa_yn_val.csv")

            if task_type == "cls_vqa_yn":
                self.label2idx = {
                    "No": 0, # no
                    "Yes": 1, # yes
                }
        elif task_type in ["close_vqa_mc", "open_vqa_mc"]:
            if mode == "train":
                csv_filepath = os.path.join(csv_dir, "m3d_vqa_train.csv")
            elif mode == "val":
                csv_filepath = os.path.join(csv_dir, "m3d_vqa_val.csv")

        if task_type in ["close_vqa_yn", "open_vqa_yn", "close_vqa_mc", "open_vqa_mc", "report_gen"]:
            model_id = config["downstream_task"]["llm"]["model_id"]
            access_token = config["downstream_task"]["llm"]["access_token"]
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
            # print(dir(self.llm_tokenizer))
            # print(self.llm_tokenizer.special_tokens_map)
            # print(self.llm_tokenizer.all_special_ids)
            # print(self.llm_tokenizer.pad_token_type_id)

        df = pd.read_csv(csv_filepath)
        self.df = df.sample(frac=data_ratio, random_state=42)
        # print(set(self.df["question_type"].to_list()))

        self.global_transforms = global_transforms
        self.image_loader = mt.LoadImage(image_only=True, ensure_channel_first=False, dtype=np.uint8)

        self.tokenizer = SentBasedTokenizer(config["tokenizer"])
        self.config = config


    def __getitem__(self, idx):
        return_dict = {}
        current_row = self.df.iloc[idx]

        ### process vision data ###
        nii_filepath = os.path.join(self.data_dir, current_row["image_path"])

        volume_tensor = self.image_loader(nii_filepath)
        # print(volume_tensor.shape, volume_tensor.dtype, (volume_tensor/1.).mean())
        # save_image((volume_tensor.permute(3, 0, 1, 2))/255., "./check_data/volume_raw_vqa.jpg")

        volume_tensor_global = self.global_transforms(volume_tensor)
        # print(volume_tensor_global.shape, volume_tensor_global.dtype, (volume_tensor_global).mean())
        # save_image((volume_tensor_global.permute(3, 0, 1, 2)), "./check_data/volume_global_vqa.jpg")

        return_dict["volume_global"] = volume_tensor_global

        ### process text data ###
        if self.task_type in ["cls_vqa_yn"]:
            question = current_row["question"]
            # print(question)
            answer = current_row["answer"]
            # print(answer)
            sentences_list = custom_split(question)
            # print(sentences_list)

            input_ids, token_type_ids, attention_mask = self.tokenizer.encode_report(sentences_list)

            target = self.label2idx[answer]
            target = torch.full((1, 1), target, dtype=torch.float32)
            return_dict["target"] = target

        elif self.task_type in ["close_vqa_yn", "open_vqa_yn", "close_vqa_mc", "open_vqa_mc"]:
            question = current_row["question"]
            answer = current_row["answer"]
            if self.task_type == "close_vqa_yn":
                question = question + " choice_A: " + current_row["choice_A"] + ", choice_B: " + current_row["choice_B"]
            elif self.task_type == "close_vqa_mc":
                question = question + " choice_A: " + current_row["choice_A"] \
                                    + ", choice_B: " + current_row["choice_B"] \
                                    + ", choice_C: " + current_row["choice_C"] \
                                    + ", choice_D: " + current_row["choice_D"] \

            sentences_list = custom_split(question)
            # print(sentences_list)
            input_ids, token_type_ids, attention_mask = self.tokenizer.encode_report(question)

            # print(answer)
            encoded_llm_input = self.llm_tokenizer(
                answer,
                padding="max_length",
                truncation=True,
                max_length=self.config["downstream_task"]["llm"]["max_length"],
                return_tensors="pt",
            )
            # print(encoded_llm_input["input_ids"])
            # print(encoded_llm_input["attention_mask"])

            return_dict["input_ids_lm"] = encoded_llm_input["input_ids"]
            return_dict["attention_mask_lm"] = encoded_llm_input["attention_mask"]
            return_dict["answer"] = answer
        

        # print(input_ids.shape, input_ids[..., :5])
        # print(token_type_ids.shape, token_type_ids[..., :5])
        # print(attention_mask.shape, attention_mask[..., :5])
        # print(target)

        return_dict["input_ids"] = input_ids
        return_dict["token_type_ids"] = token_type_ids
        return_dict["attention_mask"] = attention_mask

        return return_dict


    def __len__(self):
        return len(self.df)


if __name__ == "__main__":

    data_root = os.path.abspath(DataPath.M3D_CAP)
    data_dir = os.path.join(data_root, "nii_down")
    csv_dir = os.path.abspath(DataPath.M3D_VQA)
    
    mt_transforms = MonaiTransforms(num_samples=1)
    global_transforms = mt_transforms.load_vqa_transforms(mode="global")

    dataset = M3DVQADataset(
        data_dir, csv_dir, global_transforms, 
        task_type="close_vqa_yn", mode="train",
        config_path="./config_dt/close_vqa_yn_StSv.yaml",
    )
    # print(len(dataset))
    # dataset[14680]

    dataloader = DataLoader(dataset, batch_size=8, num_workers=8, collate_fn=collate_fn)
    for batch in tqdm(dataloader):
        # for key in batch.zkeys():
            # print(key, batch[key].size(), batch[key].dtype, batch[key].device)
        print(batch["answer"])
        break