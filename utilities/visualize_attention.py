import os
import yaml
import argparse

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import collate_fn
from datasets.m3d_cap import M3DCAPDataset
from datasets.text_utils import SentBasedTokenizer

from modeling.visiontext_model import VisionTextSSL

from utilities.constants import ConfigPath, DataPath
from utilities.vision_transforms import MonaiTransforms

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--save_dir", type=str, default="./check_data")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    return args


def get_dataloader(config_filepath):
    data_root = os.path.abspath(DataPath.M3D_CAP)
    data_dir = os.path.join(data_root, "nii_down")
    json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")
    mt_transforms = MonaiTransforms(num_samples=1)
    transforms = mt_transforms.load_ssl_transforms(mode="local")
    global_transforms = mt_transforms.load_ssl_transforms(mode="global")
    dataset = M3DCAPDataset(data_dir, json_filepath, transforms, global_transforms, data_ratio=0.8, mode="val", pretrained_type="vl_ssl", config_path=config_filepath)
    # print(len(dataset))
    g = torch.Generator()
    g.manual_seed(19) # seed 4, 70 tokens; seed 11, 63 tokens; seed 12, 79 tokens; seed 19, 68 tokens; 
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, collate_fn=collate_fn, generator=g)

    return dataloader


def plot_attention_map(attn_map, save_dir, tokens, attn_type="t2t", model_type="bert"):
    save_filepath = save_dir+"/"+attn_type+"/"
    # print(model_type)
    token_length = attn_map.shape[1]
    
    num_heads = attn_map.shape[0]
    attn_map = attn_map / np.max(attn_map, axis=(1, 2), keepdims=True)
    
    fig, axes = plt.subplots(2, num_heads//2)
    axes = axes.flatten()
    for i in range(num_heads):
        sns.heatmap(
            attn_map[i],
            ax=axes[i],
            cmap="YlOrBr",
            cbar=False,
            xticklabels=False, 
            yticklabels=False,
        )
        # axes[i].set_title(f'Head {i}')
        # axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_filepath+f"mh_map_{model_type}.jpg", format="jpg", bbox_inches="tight")
    plt.close()

    display_length = token_length // 2
    for i in range(num_heads):
        plt.figure(figsize=(4, 4))
        ax = sns.heatmap(
            attn_map[i][:display_length, :display_length],
            cmap="YlOrBr",
            cbar=False,
            xticklabels=tokens[:display_length],
            yticklabels=tokens[:display_length],
            # square=True,
        )
        # plt.title(f"tribert-head-{i}")
        # plt.axis("off")
        ax.tick_params(labelsize=4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_filepath, f"{model_type}-head{i:02d}.jpg"), format="jpg", dpi=300, bbox_inches="tight")
        plt.close()

    for i in range(num_heads):
        plt.figure(figsize=(4, 4))
        ax = sns.heatmap(
            attn_map[i],
            cmap="YlOrBr",
            cbar=False,
            xticklabels=tokens,
            yticklabels=tokens,
            square=True,
        )
        # plt.title(f"tribert-head-{i}")
        plt.axis("off")
        ax.tick_params(labelsize=4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_filepath, f"{model_type}-head{i:02d}-raw.jpg"), format="jpg", dpi=300, bbox_inches="tight")
        plt.close()

    

def main():
    args = get_args()
    
    ### create model ###
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    model = VisionTextSSL(config).to(args.device)

    ### load weights ###
    ckpt = torch.load(args.ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    ### prepare dataloader ###
    dataloader = get_dataloader(args.config_path)
    first_batch = next(iter(dataloader))
    for key in first_batch.keys():
        first_batch[key] = first_batch[key].to(args.device)
        # print(key, first_batch[key].size())
    # print(first_batch["input_ids"][0].tolist())
    tokens = model.text_ssl.tokenizer.base_tokenizer.convert_ids_to_tokens(first_batch["input_ids"][0].tolist())
    



    ### get self-attention map ###
    if "allitc" in args.config_path:
        model_type = "tribert"
    else:
        model_type = "bert"
    # print(model)
    with torch.no_grad():
        outputs = model.text_ssl.text_model(
            input_ids=first_batch["input_ids"],
            token_type_ids=first_batch["token_type_ids"],
            attention_mask=first_batch["attention_mask"],
            mode="mm_mlm",
            output_attentions=True,
            return_dict=True,
        )
    self_attentions = outputs.attentions
    # print(len(self_attentions))
    # print(self_attentions[11].shape, self_attentions[11].max(), self_attentions[11].min())
    attention_map = self_attentions[11][0].cpu().detach().numpy()
    valid_length = (first_batch["attention_mask"][0] != 0).sum().item()
    print(valid_length)
    # valid_length = 15
    attention_map = attention_map[:, :valid_length, :valid_length]
    plot_attention_map(attention_map, save_dir=args.save_dir, tokens=tokens[:valid_length], model_type=model_type)
    print(tokens[:valid_length])
    return

    ### get cross-attention map ###
    encoder_hidden_states, encoder_attention_mask = pepare_visual_inputs(model, first_batch["volume_vl"])
    # print(encoder_hidden_states.shape)
    # print(encoder_attention_mask.shape)
    outputs = model.text_ssl.text_model(
        input_ids=first_batch["input_ids"],
        token_type_ids=first_batch["token_type_ids"],
        attention_mask=first_batch["attention_mask"],
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        mode="mm_mlm",
        output_attentions=True,
        return_dict=True,
    )
    cross_attentions = outputs.cross_attentions
    # print(len(cross_attentions))
    # print(cross_attentions[-1].shape, cross_attentions[-1].max(), cross_attentions[-1].min())
    attention_map = cross_attentions[-1][0].cpu().detach().numpy()
    attention_map = attention_map[:, :valid_length, :]
    plot_attention_map(attention_map, save_dir=args.save_dir, tokens=tokens[:valid_length], attn_type="t2v")


def pepare_visual_inputs(model, volume_global):
    all_vis_features = model.vision_ssl.extract_vis_features(volume_global)
    encoder_hidden_states = all_vis_features[-1].flatten(start_dim=2, end_dim=-1)
    encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1).contiguous() # (bs, seq_len_enc, hz)
    # multi-modal adpater
    # print("before", encoder_hidden_states.shape)
    encoder_hidden_states = model.mm_adpater(encoder_hidden_states)
    # print("after", encoder_hidden_states.shape)
    encoder_hidden_shape = encoder_hidden_states.shape[:2]
    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=encoder_hidden_states.device)
    return encoder_hidden_states, encoder_attention_mask

if __name__ == "__main__":
    # seed_everything(1)
    main()
