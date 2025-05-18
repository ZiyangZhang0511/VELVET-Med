import os
import yaml
import json
import random
from tqdm.auto import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from monai.networks.nets import ViT
from monai.networks.nets.swin_unetr import SwinTransformer

from transformers import BertModel, BertConfig

from .losses import CLIPLoss

from datasets import collate_fn
from datasets.m3d_cap import M3DCAPDataset

from utilities.constants import DataPath
from utilities.utils import print_dict_content, count_params
from utilities.vision_transforms import MonaiTransforms
from utilities.metrics import compute_score_matrix, compute_RatK


class CLIP3DSSL(nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

        self.vision_encoder = ViT(
            in_channels=1,
            img_size=(256, 256, 32),
            patch_size=(16, 16, 4),
            classification=True,
        )

        # self.vision_encoder = SwinTransformer(
        #     in_chans=1,
        #     embed_dim=48,
        #     window_size=[7, 7, 7],
        #     patch_size=[2, 2, 2],
        #     depths=[2, 2, 2, 2],
        #     num_heads=[3, 6, 12, 24],
        #     use_checkpoint=True,
        #     spatial_dims=3,
        # )

        self.text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
        # self.text_encoder = BertModel.from_pretrained("google-bert/bert-base-uncased")

        # config = BertConfig()
        # self.text_encoder = BertModel(config)
        # print(self.text_encoder.encoder.layer[11].attention.self.query.weight)
        # print(self.text_encoder.embeddings.word_embeddings.weight.data[2][:5])

        self.vis_itc_head = nn.Linear(768, 512)
        self.txt_itc_head = nn.Linear(768, 512)

        self.itc_criterion = CLIPLoss()


    def forward(self, batch_data):

        volume_vl = batch_data["volume_vl"]
        # volume_vis = batch_data["volume_vis"]
        input_ids = batch_data["input_ids"]
        token_type_ids = batch_data["token_type_ids"]
        attention_mask = batch_data["attention_mask"]
        # print(input_ids[0, :])
        # print(token_type_ids[0, :])
        # print(attention_mask[0, :])
        self.device = volume_vl.device

        # print(input_ids[0, :])
        # print(attention_mask[0, :])

        loss_dict = OrderedDict()

        _, hidden_states_list = self.vision_encoder(volume_vl)
        vis_features = hidden_states_list[-1][:, 0, :]
        
        # vis_features = self.vision_encoder(volume_vl)[-1]
        # vis_features = F.avg_pool3d(vis_features, kernel_size=3, stride=3).squeeze(dim=(2, 3, 4))
        # print(vis_features.size())
        
        
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        txt_features = outputs.last_hidden_state[:, 0, :]



        vis_embeds = self.vis_itc_head(vis_features)
        txt_embeds = self.txt_itc_head(txt_features)

        loss = self.itc_criterion(vis_embeds, txt_embeds)
        
        loss_dict["loss"] = loss

        return loss_dict

    @torch.no_grad
    def test_one_step(self, batch_data):

        volume_vl = batch_data["volume_vl"].to(self.device)
        # volume_vis = batch_data["volume_vis"].to(self.device)
        input_ids = batch_data["input_ids"].to(self.device)
        token_type_ids = batch_data["token_type_ids"].to(self.device)
        attention_mask = batch_data["attention_mask"].to(self.device)

        

        loss_dict = OrderedDict()
        loss_dict["loss"] = 0

        output_dict = OrderedDict()

        _, hidden_states_list = self.vision_encoder(volume_vl)
        vis_features = hidden_states_list[-1][:, 0, :]

        # vis_features = self.vision_encoder(volume_vl)[-1]
        # vis_features = F.avg_pool3d(vis_features, kernel_size=3, stride=3).squeeze(dim=(2, 3, 4))

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        txt_features = outputs.last_hidden_state[:, 0, :]

        vis_embeds = self.vis_itc_head(vis_features)
        txt_embeds = self.txt_itc_head(txt_features)

        loss = self.itc_criterion(vis_embeds, txt_embeds)
        loss_dict["loss"] = loss

        output_dict["vis_embeds"] = vis_embeds.detach().cpu()
        output_dict["txt_embeds"] = txt_embeds.detach().cpu()

        return loss_dict, output_dict

    @torch.no_grad
    def test_on_dataloader(self, dataloader, training_step, recon_save_dir=None):
        test_steps = len(dataloader)

        epoch_loss_dict = {}

        vis_embeds_all = []
        txt_embeds_all = []

        final_dict = OrderedDict()

        for i, batch_data in enumerate(tqdm(dataloader)):

            loss_dict, output_dict = self.test_one_step(batch_data)

            # process loss items
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_loss_dict.keys():
                    epoch_loss_dict[loss_name] = 0
                epoch_loss_dict[loss_name] += loss_value.item() / len(dataloader)
            # print(epoch_loss_dict["loss_sentence_itc"])

            # process output items
            vis_embeds_all.append(output_dict["vis_embeds"])
            txt_embeds_all.append(output_dict["txt_embeds"])

        final_dict.update(epoch_loss_dict)

        vis_embeds_all = torch.cat(vis_embeds_all, dim=0)
        txt_embeds_all = torch.cat(txt_embeds_all, dim=0)
        k_test = txt_embeds_all.shape[0] if txt_embeds_all.shape[0] < 256 else 256
        # print(k_test, top_vis_embeds_all.shape, top_txt_embeds_all.shape)
        retrieval_dict = self.compute_retrieval_metrics(
            vis_embeds_all, txt_embeds_all, k_test=k_test,
        )
        final_dict.update(retrieval_dict)

        return final_dict

    def compute_retrieval_metrics(
        self,
        vis_embeds_all,
        txt_embeds_all,
        k_test=256,
    ):
        
        retrieval_dict = OrderedDict()

        score_matrix_v2t_np, score_matrix_t2v_np = compute_score_matrix(
            vis_embeds_all, txt_embeds_all, k_test=k_test,
        )

        R1_v2t = compute_RatK(score_matrix_v2t_np, k=1)
        R5_v2t = compute_RatK(score_matrix_v2t_np, k=5)
        R10_v2t = compute_RatK(score_matrix_v2t_np, k=10)
        retrieval_dict["R1_v2t"] = R1_v2t
        retrieval_dict["R5_v2t"] = R5_v2t
        retrieval_dict["R10_v2t"] = R10_v2t

        R1_t2v = compute_RatK(score_matrix_t2v_np, k=1)
        R5_t2v = compute_RatK(score_matrix_t2v_np, k=5)
        R10_t2v = compute_RatK(score_matrix_t2v_np, k=10)
        retrieval_dict["R1_t2v"] = R1_t2v
        retrieval_dict["R5_t2v"] = R5_t2v
        retrieval_dict["R10_t2v"] = R10_t2v

        return retrieval_dict


if __name__ == "__main__":

    device = "cuda"
    
    data_root = os.path.abspath(DataPath.M3D_CAP)
    data_dir = os.path.join(data_root, "nii_down")
    json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")

    mt_transforms = MonaiTransforms(num_samples=1)
    transforms = mt_transforms.load_ssl_transforms(mode="train")
    global_transforms = mt_transforms.load_ssl_transforms(mode="clip3d")

    dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=0.1, mode="val", pretrained_type="vl_ssl", text_model_type="bert")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, num_workers=8, shuffle=False, collate_fn=collate_fn)
    first_batch = next(iter(dataloader))
    for key in first_batch.keys():
        first_batch[key] = first_batch[key].to(device)

    model = CLIP3DSSL().to(device)
    
    num_trainable_params = count_params(model.vision_encoder)
    print(f"Number of trainable parameters: {num_trainable_params:.2f}M")
    num_trainable_params = count_params(model.text_encoder)
    print(f"Number of trainable parameters of text model: {num_trainable_params:.2f}M")

    ckpt_path = "./checkpoints/clip3d_ssl/clip3d_ssl-vit-m3d_cap-dr0.1-E20-best/pytorch_model.bin"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)

    # re = model(first_batch)
    # print(re)
    
    model.eval()
    re = model.test_on_dataloader(dataloader, 0)
    print_dict_content(re)