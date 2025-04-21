import os
import yaml
import math
from tqdm.auto import tqdm
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import monai.transforms as mt
from monai.losses import ContrastiveLoss
from monai.utils import ensure_tuple_rep
from monai.networks.nets import ViTAutoEnc
from monai.networks.nets.swin_unetr import SwinTransformer

from utilities.utils import get_ssl_dataset
from utilities.constants import ConfigPath, DataPath
from utilities.vision_transforms import MonaiTransforms
from utilities.metrics import multiclass_cls_metrics

from datasets import collate_fn
from datasets.m3d_cap import M3DCAPDataset
from datasets.tcia_covid19 import TCIACOVID19Dataset
from datasets.vision_aug import rot_rand, aug_rand

from .losses import VisionContrastiveLoss
from .networks import DeconvBlock, UNetBlock


class VisionSSL(nn.Module):

    def __init__(self, config, device="cuda"):
        super(VisionSSL, self).__init__()

        self.device = device

        ### prepare configs ###
        self.save_recon_interval = config["save_recon_interval"]

        vis_enc_name = config["vision_encoder"]["name"]
        input_size = config["vision_encoder"]["input_size"]
        spatial_size = np.array(input_size[1:])
        if vis_enc_name == "swinvit":
            config_vis_enc = config["vision_encoder"]["swinvit"]

            embed_dim = config_vis_enc["embed_dim"]

            vis_feature_shape_list = [
                (embed_dim*2**(i+1), *(spatial_size//(2**(i+2)))) for i in range(len(config_vis_enc["depths"]))
            ]

            self.in_chans = config_vis_enc["in_chans"]

        self.config_vis_aug = config["vision_augment"]
        # self.config_vis_encoder = config["vision_encoder"]
        config_vis_ssl_module = config["vision_ssl_module"]
        config_vis_ssl_loss = config["vision_ssl_loss"]
        

        ### build vision encoder ###
        window_size = ensure_tuple_rep(config_vis_enc["window_size"], config_vis_enc["spatial_dims"])
        patch_size = ensure_tuple_rep(config_vis_enc["patch_size"], config_vis_enc["spatial_dims"])
        self.vision_encoder = SwinTransformer(
            in_chans=config_vis_enc["in_chans"],
            embed_dim=config_vis_enc["embed_dim"],
            window_size=window_size,
            patch_size=patch_size,
            depths=config_vis_enc["depths"],
            num_heads=config_vis_enc["num_heads"],
            mlp_ratio=config_vis_enc["mlp_ratio"],
            qkv_bias=config_vis_enc["qkv_bias"],
            drop_rate=config_vis_enc["drop_rate"],
            attn_drop_rate=config_vis_enc["attn_drop_rate"],
            drop_path_rate=config_vis_enc["drop_path_rate"],
            norm_layer=nn.LayerNorm if config_vis_enc["norm_layer"] == "nn.LayerNorm" else "",
            patch_norm=config_vis_enc["patch_norm"],
            use_checkpoint=config_vis_enc["use_checkpoint"],
            spatial_dims=config_vis_enc["spatial_dims"],
            downsample=config_vis_enc["downsample"],
            use_v2=config_vis_enc["use_v2"],
        )
        self.init_vision_encoder(config_vis_enc["pretrained_ckpt"])

        ### configure vision ssl module: contrastive, rotation, reconstruction ###
        self.enable_contras = config_vis_ssl_module["enable_contras"]
        self.enable_rot = config_vis_ssl_module["enable_rot"]
        self.enable_recon = config_vis_ssl_module["enable_recon"]

        self.configure_ssl_module(config_vis_ssl_module, vis_feature_shape_list[-1], spatial_size)

        ### configure loss criteria ###
        self.configure_loss_criteria(config_vis_ssl_loss)

        self.vis_feature_shape_list = vis_feature_shape_list

    @torch.no_grad
    def init_vision_encoder(self, ckpt_path):
        """
        only initialize vision encoder with pre-trained parameters
        """
        if not os.path.isfile(ckpt_path):
            # print(self.vision_encoder.layers1[0].blocks[0].attn.qkv.weight.max())
            print("No pre-traned weights loaded for vision model......")
            return

        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)["state_dict"]
        # print(state_dict.keys())
        # print("--------------------")
        # print(self.vision_encoder.state_dict().keys())

        encoder_stata_dict = {}
        for name, params in state_dict.items():
            if "module." in name: 
                name = name.replace("module.", "")
                if "fc" in name:
                    name = name.replace("fc", "linear")
                encoder_stata_dict[name] = params
        # print("--------------------")
        # print(encoder_stata_dict.keys())

        # print(self.vision_encoder.layers1[0].blocks[0].attn.qkv.weight.max())
        self.vision_encoder.load_state_dict(encoder_stata_dict, strict=False)
        # print(self.vision_encoder.layers1[0].blocks[0].attn.proj.weight.max())

    def configure_ssl_module(self, config_vis_ssl_module, last_vis_feature_shape, spatial_size):
        
        H, W, D = spatial_size
        c, h, w, d = last_vis_feature_shape

        ### contrastive module ###
        if self.enable_contras:
            if config_vis_ssl_module["contrastive_bottleneck"] == "convolution":
                self.contras_bottleneck = nn.Conv3d(c, c, kernel_size=(h, w, d), stride=(h, w, d))
            elif config_vis_ssl_module["contrastive_bottleneck"] == "pooling":
                self.contras_bottleneck = nn.AvgPool3d(kernel_size=(h, w, d), stride=(h, w, d))
            elif config_vis_ssl_module["contrastive_bottleneck"] == "self-attention":
                pass
            self.contras_head = nn.Linear(c, config_vis_ssl_module["contrastive_hidden_dim"])
        
        ### rotation module ###
        if self.enable_rot:
            if config_vis_ssl_module["rotation_bottleneck"] == "convolution":
                self.rot_bottleneck = nn.Conv3d(c, c, kernel_size=(h, w, d), stride=(h, w, d))
            elif config_vis_ssl_module["rotation_bottleneck"] == "pooling":
                self.rot_bottleneck = nn.AvgPool3d(kernel_size=(h, w, d), stride=(h, w, d))
            elif config_vis_ssl_module["rotation_bottleneck"] == "self-attention":
                pass
            self.rot_head = nn.Linear(c, 4)

        ### reconstruction module ###
        if self.enable_recon:
            if config_vis_ssl_module["reconstruction_bottleneck"] == "deconv":
                self.recon_bottleneck = DeconvBlock(c, self.in_chans, num_upsample=int(math.log2(H//h)))
            elif config_vis_ssl_module["reconstruction_bottleneck"] == "unet":
                self.recon_bottleneck = UNetBlock(c, self.in_chans, num_upsample=int(math.log2(H//h)))
            self.recon_head = nn.Identity()

    def configure_loss_criteria(self, config_vis_ssl_loss):

        ### contrative criterion ###
        if self.enable_contras:
            temperature = config_vis_ssl_loss["contras_loss_temperature"]
            learnable_temp = config_vis_ssl_loss["learnable_temp"]
            self.contras_criterion = VisionContrastiveLoss(temperature, learnable_temp)

        ### rotation criterion ###
        if self.enable_rot:
            if config_vis_ssl_loss["rot_loss"] == "cross_entropy":
                self.rot_criterion = nn.CrossEntropyLoss()

         ### reconstruction criterion ###
        if self.enable_recon:
            if config_vis_ssl_loss["recon_loss"] == "l2":
                self.recon_criterion = nn.MSELoss()
            elif config_vis_ssl_loss["recon_loss"] == "l1":
                self.recon_criterion = nn.L1Loss()
            elif config_vis_ssl_loss["recon_loss"] == "smooth_l1":
                self.recon_criterion = nn.SmoothL1Loss()
        
        ### loss weights ###
        self.alpha_contras = config_vis_ssl_loss["alpha_contras"]
        self.alpha_rot = config_vis_ssl_loss["alpha_rot"]
        self.alpha_recon = config_vis_ssl_loss["alpha_recon"]

    def augment_data(self, x):
        """
        - input x shape [bs, in_chans, H, W, D]
        - output x1_augment x2_augment [bs, in_chans, H, W, D]
        -        target_rots [2*bs,], target_recons [2*bs, in_chans, H, W, D], 
        """
        # save_image(x[0].permute(3, 0, 1, 2), "./check_data/check_subvolume.jpg")
        x1, rot1 = rot_rand(x, x.device)
        # save_image(x1[0].permute(3, 0, 1, 2), "./check_data/check_subvolume_rot1.jpg")
        x2, rot2 = rot_rand(x, x.device)
        # save_image(x2[0].permute(3, 0, 1, 2), "./check_data/check_subvolume_rot2.jpg")
        x1_augment = aug_rand(x1, self.config_vis_aug, x.device)
        # save_image(x1_augment[0].permute(3, 0, 1, 2), "./check_data/check_subvolume_aug1.jpg")
        x2_augment = aug_rand(x2, self.config_vis_aug, x.device)
        # save_image(x2_augment[0].permute(3, 0, 1, 2), "./check_data/check_subvolume_aug2.jpg")

        # print(x.shape, x1.shape, x1_augment.shape)
        # print(rot1.dtype, rot1.shape)

        target_rot = torch.cat([rot1, rot2], axis=0)
        target_recon = torch.cat([x1, x2], axis=0)

        return x1_augment, x2_augment, target_rot, target_recon

    def extract_vis_features(self, x):
        """
        - input x shape [bs, in_chans, H, W, D]
        - output list of tensor with hierarchical shape [bs, c, h, w, d]
        """
        # feat_list = self.vision_encoder(x)
        # for feat in feat_list:
        #     print(feat.shape)
        return self.vision_encoder(x)

    def through_ssl_module(self, x1_features, x2_features):
        """
        - input x1_features, x2_features shape [bs, c, h, w, d]
        - output x1_embedding x2_embedding [bs, contras_dim]
        -        logits_rots [2*bs,], logits_recons [2*bs, in_chans, H, W, D], 
        """

        outputs = ()

        if self.enable_contras:
            x1_contras = self.contras_bottleneck(x1_features).squeeze(dim=(2, 3, 4))
            x2_contras = self.contras_bottleneck(x2_features).squeeze(dim=(2, 3, 4))
            x1_embedding, x2_embedding = self.contras_head(x1_contras), self.contras_head(x2_contras)
            # print("contras", x1_embedding.shape, x2_embedding.shape)
            outputs +=  (x1_embedding, x2_embedding)
        else:
            outputs += (None, None)

        if self.enable_rot:
            x1_rot = self.rot_bottleneck(x1_features).squeeze(dim=(2, 3, 4))
            x2_rot = self.rot_bottleneck(x2_features).squeeze(dim=(2, 3, 4))
            logits_x1_rot, logits_x2_rot = self.rot_head(x1_rot), self.rot_head(x2_rot)
            logits_rot = torch.cat([logits_x1_rot, logits_x2_rot], axis=0)
            # print("rot", logits_rot.shape, logits_x1_rot.shape, logits_x2_rot.shape)
            outputs +=  (logits_rot,)
        else:    
            outputs += (None,)

        if self.enable_recon:
            x1_recon = self.recon_bottleneck(x1_features)
            x2_recon = self.recon_bottleneck(x2_features)
            logits_recon = torch.cat([x1_recon, x2_recon], axis=0)
            # print("recon", logits_recon.shape, x1_recon.shape, x2_recon.shape)
            outputs +=  (logits_recon,)
        else:
            outputs += (None,)

        return outputs


    def forward(self, batch_data):
        """
        - batch_data["volume_vis"] [bs, in_chans, H, W, D]
        """
        
        self.device = batch_data["volume_vis"].device

        ### augment volume for ssl ###
        x = batch_data["volume_vis"]
        x1_augment, x2_augment, target_rot, target_recon = self.augment_data(x)

        ### get last visual features through vision encoder ###
        x1_features = self.extract_vis_features(x1_augment)[-1]
        x2_features = self.extract_vis_features(x2_augment)[-1]
        # print(x1_features.shape, x2_features.shape)

        ### get visual features through ssl module ###
        outputs = self.through_ssl_module(x1_features, x2_features)

        ### compute loss items ###
        loss_placeholder = torch.tensor(0, dtype=x1_features.dtype, device=x1_features.device, requires_grad=False)
        loss_contras = self.contras_criterion(outputs[0], outputs[1]) if self.enable_contras else loss_placeholder
        loss_rot = self.rot_criterion(outputs[2], target_rot) if self.enable_rot else loss_placeholder
        loss_recon = self.recon_criterion(outputs[3], target_recon) if self.enable_recon else loss_placeholder

        logits_recon = outputs[3].detach().cpu() if self.enable_recon else None
        target_recon = target_recon.detach().cpu() if self.enable_recon else None

        loss_vis_ssl = self.alpha_contras * loss_contras +\
                       self.alpha_rot * loss_rot +\
                       self.alpha_recon * loss_recon

        return {
            "loss": loss_vis_ssl,
            "loss_contras": loss_contras,
            "loss_rot": loss_rot,
            "loss_recon": loss_recon,
        }

    @torch.no_grad()
    def test_one_step(self, batch_data, requires_recon=False):

        ### augment volume for ssl ###
        x = batch_data["volume_vis"].to(self.device)
        x1_augment, x2_augment, target_rot, target_recon = self.augment_data(x)

        ### get last visual features through vision encoder ###
        x1_features = self.extract_vis_features(x1_augment)[-1]
        x2_features = self.extract_vis_features(x2_augment)[-1]
        # print(x1_features.shape, x2_features.shape)

        ### get visual features through ssl module ###
        outputs = self.through_ssl_module(x1_features, x2_features)

        ### compute loss items ###
        loss_placeholder = torch.tensor(0, dtype=x1_features.dtype, device=x1_features.device, requires_grad=False)
        loss_contras = self.contras_criterion(outputs[0], outputs[1]) if self.enable_contras else loss_placeholder
        loss_rot = self.rot_criterion(outputs[2], target_rot) if self.enable_rot else loss_placeholder
        loss_recon = self.recon_criterion(outputs[3], target_recon) if self.enable_recon else loss_placeholder

        loss_vis_ssl = self.alpha_contras * loss_contras +\
                       self.alpha_rot * loss_rot +\
                       self.alpha_recon * loss_recon


        test_output_dict = OrderedDict()
        if self.enable_rot:
            test_output_dict["logits_rot"] = outputs[2].detach().cpu()
            test_output_dict["target_rot"] = target_rot.detach().cpu()
        if self.enable_recon:
            if requires_recon:
                test_output_dict["volume_recon"] = outputs[3].detach().cpu()[0] # [1, 96, 96, 96]
                test_output_dict["volume_target"] = target_recon.detach().cpu()[0] # [1, 96, 96, 96]
            else:
                test_output_dict["volume_recon"] = None
                test_output_dict["volume_target"] = None

        return {
            "loss": loss_vis_ssl,
            "loss_contras": loss_contras,
            "loss_rot": loss_rot,
            "loss_recon": loss_recon,
        }, test_output_dict


    @torch.no_grad()
    def test_on_dataloader(self, dataloader, training_step, recon_save_dir=None):

        requires_recon = False if recon_save_dir == None else True
        
        test_steps = len(dataloader)

        epoch_loss_dict = {}
        epoch_output_dict = {}

        logits_rot_all = []
        target_rot_all = []
        volume_recon_list = []
        volume_target_list = []

        for i, batch_data in enumerate(tqdm(dataloader)):

            loss_dict, output_dict = self.test_one_step(batch_data, requires_recon)

            # process loss items
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_loss_dict.keys():
                    epoch_loss_dict[loss_name] = 0
                epoch_loss_dict[loss_name] += loss_value.item() / len(dataloader)

            # process output items
            if self.enable_rot:
                logits_rot_all.append(output_dict["logits_rot"])
                target_rot_all.append(output_dict["target_rot"])
            if self.enable_recon:
                volume_recon_list.append(output_dict["volume_recon"])
                volume_target_list.append(output_dict["volume_target"])

        ### structure final dict to return ###
        final_dict = epoch_loss_dict.copy()

        ### rotation prediction metrics ###
        if self.enable_rot:
            logits_rot_all = torch.cat(logits_rot_all, axis=0)
            probs_rot_all = F.softmax(logits_rot_all, dim=-1).numpy()
            target_rot_all = torch.cat(target_rot_all, axis=0).numpy()
            cls_metric_dict = multiclass_cls_metrics(probs_rot_all, target_rot_all)
            final_dict["rot_accuracy"] = cls_metric_dict["accuracy"]
            final_dict["rot_auc"] = cls_metric_dict["auc"]

        ### visualize reconstruction ###  
        if recon_save_dir and self.enable_recon:
            if training_step % self.save_recon_interval == 0:
                for i, (volume_recon, volume_target) in enumerate(zip(volume_recon_list, volume_target_list)):
                    if i % (test_steps//3) == 0:
                        save_image(volume_recon.permute(3, 0, 1, 2).contiguous(), f"{recon_save_dir}/recon-{training_step}-idx{i}.jpg")
                        save_image(volume_target.permute(3, 0, 1, 2).contiguous(), f"{recon_save_dir}/target-{training_step}-idx{i}.jpg")

        return final_dict


class VitAutoEncSSL(nn.Module):

    def __init__(self, config=None, device="cpu"):
        super(VitAutoEncSSL, self).__init__()

        self.device = device

        ### vision encoder combined with vision ssl heads  ###
        self.vit_autoenc = ViTAutoEnc(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            proj_type="conv",
            hidden_size=768,
            mlp_dim=3072,
        ).to(self.device)

        ### loss criterion ###
        self.recon_loss = nn.L1Loss()
        self.contrastive_loss = ContrastiveLoss(temperature=0.05)


    def forward(self, batch_data):
        
        self.device = batch_data["image"].device
        # inputs, inputs_2, gt_input = (
        #     batch_data["image"].to(self.device),
        #     batch_data["image_2"].to(self.device),
        #     batch_data["gt_image"].to(self.device),
        # )
        outputs_v1, hidden_v1 = self.vit_autoenc(inputs)
        outputs_v2, hidden_v2 = self.vit_autoenc(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)
        # print(flat_out_v1.size(), outputs_v1.size())

        r_loss = self.recon_loss(outputs_v1, gt_input)
        cl_loss = self.contrastive_loss(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss + cl_loss * r_loss

        return {
            "loss": total_loss,
            "recon_loss": r_loss,
            "cl_loss": cl_loss,
        }

    @torch.no_grad()
    def test_one_step(self, batch_data):
        inputs, inputs_2, gt_input = (
            batch_data["image"].to(self.device),
            batch_data["image_2"].to(self.device),
            batch_data["gt_image"].to(self.device),
        )
        outputs_v1, hidden_v1 = self.vit_autoenc(inputs)
        outputs_v2, hidden_v2 = self.vit_autoenc(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)
        # print(flat_out_v1.size(), outputs_v1.size())

        r_loss = self.recon_loss(outputs_v1, gt_input)
        cl_loss = self.contrastive_loss(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss + cl_loss * r_loss

        return {
            "loss": total_loss,
            "recon_loss": r_loss,
            "cl_loss": cl_loss,
        }, {}

    @torch.no_grad()
    def test_on_dataloader(self, dataloader, training_step, recon_save_dir=None):

        epoch_loss_dict = {}
        epoch_output_dict = {}

        for i, batch_data in enumerate(dataloader):

            loss_dict, output_dict = self.test_one_step(batch_data)

            # process loss items
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_loss_dict.keys():
                    epoch_loss_dict[loss_name] = 0
                epoch_loss_dict[loss_name] += loss_value.item() / len(dataloader)

        ### structure final dict to return ###
        final_dict = epoch_loss_dict.copy()

        return final_dict

    


if __name__ == "__main__":

    ### prepare data ###
    device = "cuda"

    data_root = os.path.abspath(DataPath.M3D_CAP)
    data_dir = os.path.join(data_root, "nii_down")
    json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")
    mt_transforms = MonaiTransforms(num_samples=2)
    transforms = mt_transforms.load_ssl_transforms()
    global_transforms = mt_transforms.load_ssl_transforms(mode="global")
    dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=0.001, mode="train")


    # transforms = mt.Compose([
    #     mt.Orientation(axcodes="RAS"),
    #     mt.ScaleIntensityRange(
    #         a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
    #     ),
    #     mt.Spacing(
    #         pixdim=(2.0, 2.0, 2.0),
    #         mode=("bilinear"),
    #         # min_pixdim=(1.0, 1.0, 1.0), 
    #         # max_pixdim=None,
    #         align_corners=False,
    #     ),
    #     mt.CropForeground(allow_smaller=False),
    #     mt.SpatialPad(spatial_size=[96, 96, 96]),
    #     mt.RandSpatialCropSamples(
    #         roi_size=[96, 96, 96],
    #         num_samples=2,
    #         random_center=True,
    #         random_size=False,
    #     ),
    # ])
    # dataset = TCIACOVID19Dataset(transforms, mode="train")

    dataloader = DataLoader(dataset, batch_size=2, num_workers=8, shuffle=False, collate_fn=collate_fn)
    first_batch = next(iter(dataloader))
    print(first_batch["volume_vis"].shape)
    first_batch["volume_vis"] = first_batch["volume_vis"].to(device)
    # volume_vis = torch.randn((16, 1, 96, 96, 96)).to(device)
    # save_image(volume_vis[0].permute(3, 0, 1, 2), "./check_data/check_subvolume.jpg")

    ### build model ###
    config_path = os.path.abspath(ConfigPath.VISION_SSL)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # print(type(config))

    model = VisionSSL(config).to(device)
    # model.device = device
    loss_dict = model(first_batch)
    print("===val===")
    final_dict = model.test_on_dataloader(dataloader, training_step=1)
    print(final_dict)


    

    


