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

from monai.utils import ensure_tuple_rep
from monai.networks.nets.swin_unetr import SwinTransformer

from utilities.metrics import binary_cls_metrics
from utilities.vision_transforms import MonaiTransforms
from utilities.constants import DataPath, ConfigPath
from utilities.utils import print_dict_content, count_params

from datasets import collate_fn
from datasets.m3d_vqa import M3DVQADataset
from datasets.text_utils import SentBasedTokenizer

from .networks import SentBertModel


class CLSModel(nn.Module):

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

        config_tokenizer = config["tokenizer"]
        self.tokenizer = SentBasedTokenizer(config_tokenizer)

        config_txt_model = config["text_model"]
        config_txt_model["pad_token_id"] = self.tokenizer.base_tokenizer.pad_token_id
        config_txt_model["vocab_size"] = self.tokenizer.base_tokenizer.vocab_size
        
        config_txt_model["sentence_modeling"] = config_tokenizer["sentence_modeling"]
        if config_txt_model["sentence_modeling"]:
            config_txt_model["type_vocab_size"] = config_tokenizer["max_num_sentences"] + 1
        else:
            config_txt_model["type_vocab_size"] = 2
            
        self.config_txt_model = config_txt_model

        self.config_dt = config["downstream_task"]
        config_freeze = config["freeze"]

        ### create vision encoder ###
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

        ### create text model ###
        self.text_model = SentBertModel(config_txt_model)

        ### create multi-modal adapter ###
        c, h ,w, d = vis_feature_shape_list[-1]
        txt_hidden_size = config_txt_model["hidden_size"]
        self.mm_adpater = nn.Sequential(
            nn.Linear(c, txt_hidden_size),
            nn.LayerNorm(txt_hidden_size),
        )

        ### create classifier ###
        txt_hidden_size = self.config_dt["txt_hidden_size"]
        self.classifier = nn.Sequential(
            nn.Linear(txt_hidden_size, txt_hidden_size),
            nn.GELU(),
            nn.LayerNorm(txt_hidden_size),
            nn.Linear(txt_hidden_size, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss()

        ### initialize model ###
        self.initialize_model(config["pretrained_ckpt"])
        self.freeze_params(config_freeze)


    def initialize_model(self, pretrained_ckpt):
        if not os.path.isfile(pretrained_ckpt):
            # print(self.vision_encoder.layers1[0].blocks[0].attn.qkv.weight.max())
            print("No pre-traned weights loaded......")
            return
        
        state_dict = torch.load(pretrained_ckpt, map_location=self.device, weights_only=True)

        vision_encoder_stata_dict = OrderedDict()
        text_model_stata_dict = OrderedDict()
        mm_adpater_stata_dict = OrderedDict()

        for name, params in state_dict.items():
            if "vision_ssl.vision_encoder." in name:
                name = name.replace("vision_ssl.vision_encoder.", "")
                vision_encoder_stata_dict[name] = params
            elif "text_ssl.text_model." in name:
                name = name.replace("text_ssl.text_model.", "")
                text_model_stata_dict[name] = params
            elif "mm_adpater." in name:
                name = name.replace("mm_adpater.", "")
                mm_adpater_stata_dict[name] = params
            
        self.vision_encoder.load_state_dict(vision_encoder_stata_dict, strict=True)
        self.text_model.load_state_dict(text_model_stata_dict, strict=True)
        if mm_adpater_stata_dict:
            self.mm_adpater.load_state_dict(mm_adpater_stata_dict, strict=True)

    def freeze_params(self, config_freeze):
        if config_freeze["vision_encoder"]:
            for params in self.vision_encoder.parameters():
                params.requires_grad = False
            print("Freeze vision encoder...")
        # if config_freeze["text_model"]:
        #     for params in self.text_model.parameters():
        #         params.requires_grad = False
        #     print("Freeze text model...")
        if config_freeze["mm_adapter"]:
            for params in self.mm_adpater.parameters():
                params.requires_grad = False
            print("Freeze mm adpater...")

        mm_encoder_layers = self.config_txt_model["num_decoder_layers"]
        txt_encoder_layers = self.config_txt_model["num_hidden_layers"] - mm_encoder_layers
        
        txt_encoder_stems = [f"layer.{i}." for i in range(txt_encoder_layers)]
        txt_encoder_stems.append("embeddings")
        if config_freeze["text_encoder"]:
            for name, params in self.text_model.named_parameters():
                if any(stem in name for stem in txt_encoder_stems):
                    params.requires_grad = False
            print("Freeze text encoder...")

        mm_encoder_stems = [f"layer.{i}." for i in range(txt_encoder_layers, txt_encoder_layers+mm_encoder_layers)]
        if config_freeze["mm_encoder"]:
            for name, params in self.text_model.named_parameters():
                if any(stem in name for stem in mm_encoder_stems):
                    params.requires_grad = False
            print("Freeze mm encoder...")

        # print("trainable parmas......")
        # for name, params in self.named_parameters():
        #     if params.requires_grad:
        #         print(name)

    def produce_mm_hidden_state(
        self,
        volume_global,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        
        ### extract visual features ###
        all_vis_features = self.vision_encoder(volume_global)
        encoder_hidden_states = all_vis_features[-1].flatten(start_dim=2, end_dim=-1)
        encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1).contiguous() # (bs, seq_len_enc, hz)
        # print("before", encoder_hidden_states.shape)
        encoder_hidden_states = self.mm_adpater(encoder_hidden_states)
        # print("after", encoder_hidden_states.shape)
        encoder_hidden_shape = encoder_hidden_states.shape[:2]
        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=encoder_hidden_states.device)
        # print(encoder_hidden_states.shape)
        # print(encoder_attention_mask[0, :10])
 
        outputs = self.text_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            mode="mm_mlm",
        )
        # mm_embeddings = outputs.last_hidden_state
        
        return outputs.last_hidden_state


    def calculate_cls_loss(self, mm_embeddings, target):
        logits = self.classifier(mm_embeddings)
        loss = self.criterion(logits, target)
        return loss

    def forward(self, batch_data):
        volume_global = batch_data["volume_global"]
        input_ids = batch_data["input_ids"]
        token_type_ids = batch_data["token_type_ids"]
        attention_mask = batch_data["attention_mask"]
        target = batch_data["target"]
        bs = volume_global.shape[0]
        
        self.device = volume_global.device

        mm_hidden_state = self.produce_mm_hidden_state(
            volume_global,
            input_ids,
            attention_mask,
            token_type_ids,
        )

        logits = self.classifier(mm_hidden_state[:, 0, :])
        loss = self.criterion(logits, target)
        # loss = self.calculate_cls_loss(mm_hidden_state[:, 0, :], target)
        
        return {
            "loss": loss,
        }

    @torch.no_grad
    def test_one_step(self, batch_data):
        volume_global = batch_data["volume_global"].to(self.device)
        input_ids = batch_data["input_ids"].to(self.device)
        token_type_ids = batch_data["token_type_ids"].to(self.device)
        attention_mask = batch_data["attention_mask"].to(self.device)
        target = batch_data["target"].to(self.device)

        test_output_dict = OrderedDict()

        mm_hidden_state = self.produce_mm_hidden_state(
            volume_global,
            input_ids,
            attention_mask,
            token_type_ids,
        )

        logits = self.classifier(mm_hidden_state[:, 0, :])
        loss = self.criterion(logits, target)
        # loss = self.calculate_cls_loss(mm_hidden_state[:, 0, :], target)

        test_output_dict["target"] = target.detach().cpu()
        test_output_dict["logits"] = logits.detach().cpu()
        
        return {
            "loss": loss,
        }, test_output_dict

    @torch.no_grad()
    def test_on_dataloader(self, dataloader):        
        final_dict = OrderedDict()

        epoch_loss_dict = {}
        epoch_output_dict = {}

        logits_all = []
        target_all = []

        for i, batch_data in enumerate(tqdm(dataloader)):

            loss_dict, output_dict = self.test_one_step(batch_data)

            # process loss items
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_loss_dict.keys():
                    epoch_loss_dict[loss_name] = 0
                epoch_loss_dict[loss_name] += loss_value.item() / len(dataloader)

            logits_all.append(output_dict["logits"])
            target_all.append(output_dict["target"])

        final_dict = epoch_loss_dict.copy()

        logits_all = torch.cat(logits_all, axis=0)
        probs_all = F.sigmoid(logits_all).numpy()
        target_all = torch.cat(target_all, axis=0).numpy()
        # print(probs_all.shape, target_all.shape)
        cls_metric_dict = binary_cls_metrics(probs_all, target_all)
        
        final_dict["accuracy"] = cls_metric_dict["accuracy"]
        final_dict["auc"] = cls_metric_dict["auc"]

        return final_dict

