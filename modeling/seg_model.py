import os
import yaml
import math
from tqdm.auto import tqdm
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import monai
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.utils import ensure_tuple_rep
from monai.inferers import sliding_window_inference
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    Dataset as mDataset,
    DataLoader as mDataLoader,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

from utilities.metrics import nlg_metrics
from utilities.vision_transforms import MonaiTransforms
from utilities.utils import print_dict_content, count_params
from utilities.constants import DataPath, ConfigPath, SliceMap
from utilities.seg_utils import load_seg_data_list, visualize_transformed_data

from datasets import collate_fn



def load_segmenter(segmenter_type, ckpt_path=None):

    if segmenter_type == "swinunetr":
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=14,
            feature_size=48,
            use_checkpoint=False,
        )
        if ckpt_path:
            weights = torch.load(ckpt_path, weights_only=True)

            # own pre-trained ckpt
            if "state_dict" not in weights.keys():
                new_weights = {"state_dict": {}}
                for key in weights.keys(): 
                    if "vision_encoder" in key: 
                        new_key = key.replace("vision_encoder", "module")
                        if "linear" in new_key:
                            new_key = new_key.replace("linear", "fc")
                        new_weights["state_dict"][new_key] = weights[key]
                model.load_from(new_weights)

            # official ssl ckpt from swinunetr paper
            else:
                model.load_from(weights)

    return model


class SEGModel(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.device = device


        ### prepare config ###
        vis_enc_name = config["vision_encoder"]["name"]
        input_size = config["vision_encoder"]["input_size"]
        spatial_size = np.array(input_size[1:])
        if vis_enc_name == "swinvit":
            config_vis_enc = config["vision_encoder"]["swinvit"]
            embed_dim = config_vis_enc["embed_dim"]
            vis_feature_shape_list = [
                (embed_dim*2**(i+1), *(spatial_size//(2**(i+2)))) for i in range(len(config_vis_enc["depths"]))
            ]

        config_freeze = config["freeze"]

        if config["dataset"] == "btcv":
            num_classes = 14
        elif config["dataset"] == "abdomenct1k":
            num_classes = 5
        elif config["dataset"] == "ctorg":
            num_classes = 7
        elif config["dataset"] == "totalsegmentator":
            num_classes = 105

        window_size = ensure_tuple_rep(config_vis_enc["window_size"], config_vis_enc["spatial_dims"])
        patch_size = ensure_tuple_rep(config_vis_enc["patch_size"], config_vis_enc["spatial_dims"])
        self.segmenter = SwinUNETR(
            img_size=spatial_size,
            in_channels=config_vis_enc["in_chans"],
            out_channels=num_classes,
            depths=config_vis_enc["depths"],
            num_heads=config_vis_enc["num_heads"],
            # patch_size=patch_size,
            feature_size=config_vis_enc["embed_dim"],
            norm_name="instance",
            drop_rate=config_vis_enc["drop_rate"],
            attn_drop_rate=config_vis_enc["attn_drop_rate"],
            dropout_path_rate=config_vis_enc["drop_path_rate"],
            use_checkpoint=config_vis_enc["use_checkpoint"],
            spatial_dims=config_vis_enc["spatial_dims"],
            downsample=config_vis_enc["downsample"],
            use_v2=config_vis_enc["use_v2"],
        )
        self.segmenter.patch_size = config_vis_enc["patch_size"]
        self.spatial_size = spatial_size

        self.criterion = DiceCELoss(softmax=True, to_onehot_y=True)
        
        self.dice_metric_perclass = DiceMetric(
            include_background=False,
            num_classes=num_classes,
            reduction="mean_batch",
            return_with_label=True,
        )
        self.dice_metric_mean = DiceMetric(
            include_background=False,
            num_classes=num_classes, 
            reduction="mean",
        )
        self.nsd_metric_mean = SurfaceDiceMetric(
            class_thresholds=[2.0]*(num_classes-1),
            include_background=False,
            reduction="mean",
        )
        self.hd_metric_mean = HausdorffDistanceMetric(
            include_background=False,
            distance_metric="euclidean",
            percentile=95,
            reduction="mean",
        )

        ### initialize model ###
        self.initialize_model(config["pretrained_ckpt"])
        self.freeze_params(config_freeze)

    def freeze_params(self, config_freeze):
        if config_freeze["vision_encoder"]:
            for params in self.segmenter.swinViT.parameters():
                params.requires_grad = False
            print("Freeze vision encoder...")


    def initialize_model(self, pretrained_ckpt):
        if not os.path.isfile(pretrained_ckpt):
            # print(self.vision_encoder.layers1[0].blocks[0].attn.qkv.weight.max())
            print("No pre-traned weights loaded......")
            return
        
        state_dict = torch.load(pretrained_ckpt, map_location=self.device, weights_only=True)

        vision_encoder_stata_dict = OrderedDict()

        for name, params in state_dict.items():
            if "vision_ssl.vision_encoder." in name:
                name = name.replace("vision_ssl.vision_encoder.", "")
                vision_encoder_stata_dict[name] = params
            elif "vision_encoder." in name:
                name = name.replace("vision_encoder.", "")
                vision_encoder_stata_dict[name] = params
            
        self.segmenter.swinViT.load_state_dict(vision_encoder_stata_dict, strict=True)


    def forward(self, batch_data):
        images = batch_data["image"]
        labels = batch_data["label"]
        self.device = images.device

        logits_map = self.segmenter(images)
        loss = self.criterion(logits_map, labels)

        return {
            "loss": loss
        }

    @torch.no_grad
    def test_one_step(self, batch_data):
        images = batch_data["image"].to(self.device)
        labels = batch_data["label"].to(self.device)

        outputs = sliding_window_inference(
            images,
            roi_size=self.spatial_size,
            sw_batch_size=4,
            predictor=self.segmenter,
            overlap=0.25,
        )
        num_classes = outputs.shape[1]

        loss = self.criterion(outputs, labels)

        outputs_indices = torch.argmax(outputs, dim=1, keepdim=True)
        outputs_onehot = monai.networks.utils.one_hot(outputs_indices, num_classes=num_classes, dim=1)
        labels_onehot = monai.networks.utils.one_hot(labels, num_classes=num_classes, dim=1)

        self.dice_metric_perclass(y_pred=outputs_onehot, y=labels_onehot)
        self.dice_metric_mean(y_pred=outputs_onehot, y=labels_onehot)
        self.nsd_metric_mean(y_pred=outputs_onehot, y=labels_onehot)
        self.hd_metric_mean(y_pred=outputs_onehot, y=labels_onehot)

        return {
            "loss": loss,
        }, {}

    @torch.no_grad
    def test_on_dataloader(self, dataloader):

        final_dict = OrderedDict()

        epoch_loss_dict = {}

        pred_captions_all = []
        gt_captions_all = []

        for i, batch_data in enumerate(tqdm(dataloader)):
            # if bool((batch_data["label"] == 6.).any().item()):
            #     print(batch_data["label"].size(), batch_data["label"].max())
            loss_dict, output_dict = self.test_one_step(batch_data)

            # process loss items
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_loss_dict.keys():
                    epoch_loss_dict[loss_name] = 0
                epoch_loss_dict[loss_name] += loss_value.item() / len(dataloader)
        final_dict.update(epoch_loss_dict)
        
        dice_mean = self.dice_metric_mean.aggregate().item()
        final_dict["dice_mean"] = dice_mean
        
        dice_metrics = self.dice_metric_perclass.aggregate()
        final_dict.update(dice_metrics)

        nsd_mean = self.nsd_metric_mean.aggregate().item()
        final_dict["nsd_mean"] = nsd_mean

        hd_mean = self.hd_metric_mean.aggregate().item()
        final_dict["hd_mean"] = hd_mean
        

        return final_dict



