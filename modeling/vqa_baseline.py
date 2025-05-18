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
from monai.networks.nets.vit import ViT
import monai

from peft import LoraConfig
from peft import get_peft_model
from transformers import LlamaForCausalLM, AutoTokenizer

from utilities.metrics import nlg_metrics
from utilities.constants import DataPath, ConfigPath
from utilities.vision_transforms import MonaiTransforms
from utilities.utils import print_dict_content, count_params

from datasets import collate_fn
from datasets.m3d_vqa import M3DVQADataset, M3DVQABaselineDataset



class VQABaseline(nn.Module):
    def __init__(self, config, device="cuda:0"):
        super().__init__()

        self.config_dt = config["downstream_task"]

        

        ### build vision encoder
        self.device = device

        self.vision_encoder = ViT(
            in_channels=1,
            img_size=(256, 256, 32),
            patch_size=(16, 16, 4),
            classification=True,
        )

        ### build perceiver
        self.vision_projection = nn.Linear(768, config["downstream_task"]["llm"]["hidden_size"])

        ### build llm
        self.configure_llm(config["downstream_task"]["llm"])
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

         ### initialize model ###
        self.initialize_model(config["pretrained_ckpt"])
        config_freeze = config["freeze"]
        self.freeze_params(config_freeze)

        ### configure generate kwargs ###
        self.generate_kwargs = {
            "max_length": config["downstream_task"]["llm"]["gen_max_length"],
            "min_length": config["downstream_task"]["llm"]["gen_min_length"],
        }

    def initialize_model(self, pretrained_ckpt):
        if not os.path.isfile(pretrained_ckpt):
            # print(self.vision_encoder.layers1[0].blocks[0].attn.qkv.weight.max())
            print("No pre-traned weights loaded......")
            return
        
        state_dict = torch.load(pretrained_ckpt, map_location=self.device, weights_only=True)

        vision_encoder_stata_dict = OrderedDict()

        for name, params in state_dict.items():
            if "vision_encoder." in name:
                name = name.replace("vision_encoder.", "")
                vision_encoder_stata_dict[name] = params
            
        self.vision_encoder.load_state_dict(vision_encoder_stata_dict, strict=False)
        

    def freeze_params(self, config_freeze):
        if config_freeze["vision_encoder"]:
            for params in self.vision_encoder.parameters():
                params.requires_grad = False
            print("Freeze vision encoder...")

    def configure_llm(self, config_llm):
        access_token = config_llm["access_token"]
        model_id = config_llm["model_id"]
        self.llm = LlamaForCausalLM.from_pretrained(model_id, token=access_token)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

        peft_config = LoraConfig(
            r=config_llm["rank"], #Rank
            lora_alpha=32,
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'dense',
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        # print(self.llm)
        self.llm.model.gradient_checkpointing_enable()
        self.llm.model.use_cache = False
        self.llm.model.gradient_checkpointing = True


    def produce_visual_embeddings(self, volume_global):
        _, hidden_states_list = self.vision_encoder(volume_global)
        vis_features = hidden_states_list[-1][:, 1:, :]
        
        vis_features = vis_features.view(-1, 16, 16, 8, 768).permute(0, 4, 1, 2, 3).contiguous()
        vis_features = F.max_pool3d(vis_features, kernel_size=(4, 4, 4), stride=(4, 4, 4))
        vis_features = vis_features.flatten(2, -1).permute(0, 2, 1).contiguous()

        visual_embeddings = self.vision_projection(vis_features)

        return visual_embeddings

    def prepare_llm_input(
        self,
        visual_embeddings,
        input_ids_q,
        attention_mask_q,
        input_ids_a,
        attention_mask_a,
    ):
        bs = visual_embeddings.shape[0]
        # print(visual_embeddings.size())

        inputs_embeds_q = self.llm.get_input_embeddings()(input_ids_q)
        inputs_embeds_a = self.llm.get_input_embeddings()(input_ids_a)
        # print(inputs_embeds_q.size(), attention_mask_q[0])
        # print(inputs_embeds_a.size(), attention_mask_a[0])

        inputs_embeds_lm_list = []
        attention_mask_lm_list = []
        input_ids_lm_list = []
        max_length = self.config_dt["llm"]["max_length"]
        for i in range(bs):
            
            indices_q = torch.where(input_ids_q[i] == 128009)[0]
            cur_input_ids_q = input_ids_q[i][:indices_q[0]]
            cur_input_embeds_q = inputs_embeds_q[i][:indices_q[0]]

            cur_vl_prompt_embeds = torch.cat([visual_embeddings[i], cur_input_embeds_q], dim=0)
            cur_vl_prompt_attention_mask = torch.full((cur_vl_prompt_embeds.shape[0],), 1.).to(visual_embeddings.device)
            cur_vl_prompt_input_ids = torch.full(cur_vl_prompt_attention_mask.size(), -100).to(visual_embeddings.device)
            # print(cur_vl_prompt_embeds.size())

            cur_inputs_embeds_lm = torch.cat([cur_vl_prompt_embeds, inputs_embeds_a[i]], dim=0)
            # print(cur_vl_prompt_attention_mask.size(), attention_mask_a[i].size())
            cur_attention_mask_lm = torch.cat([cur_vl_prompt_attention_mask, attention_mask_a[i]], dim=0)
            cur_input_ids_lm = torch.cat([cur_vl_prompt_input_ids, input_ids_a[i]], dim=0)
            indices = torch.where(cur_input_ids_lm == 128009)[0]
            if indices.numel() > 0:
                cur_input_ids_lm[indices] = -100
                cur_input_ids_lm[indices[0]] = 128009
            cur_input_ids_lm = cur_input_ids_lm[:max_length].unsqueeze(0)
            # print(cur_input_ids_lm.size(), cur_input_ids_lm)
            cur_inputs_embeds_lm = cur_inputs_embeds_lm[:max_length, :].unsqueeze(0)
            cur_attention_mask_lm = cur_attention_mask_lm[:max_length].unsqueeze(0)
            # print(cur_attention_mask_lm.size(), cur_attention_mask_lm)
            inputs_embeds_lm_list.append(cur_inputs_embeds_lm)
            attention_mask_lm_list.append(cur_attention_mask_lm)
            input_ids_lm_list.append(cur_input_ids_lm)
            

        inputs_embeds_lm = torch.cat(inputs_embeds_lm_list, dim=0)
        attention_mask_lm = torch.cat(attention_mask_lm_list, dim=0)
        input_ids_lm =  torch.cat(input_ids_lm_list, dim=0)
            
        return inputs_embeds_lm, attention_mask_lm, input_ids_lm

    def forward(self, batch_data):
        volume_global = batch_data["volume_global"]
        input_ids_q = batch_data["input_ids_q"]
        attention_mask_q = batch_data["attention_mask_q"]
        input_ids_a = batch_data["input_ids_a"]
        attention_mask_a = batch_data["attention_mask_a"]
        bs = volume_global.shape[0]
        self.device = volume_global.device
        # print(input_ids_q.shape, attention_mask_q.shape)
        # print(input_ids_a.shape, attention_mask_a.shape)
        visual_embeddings = self.produce_visual_embeddings(volume_global)

        inputs_embeds_lm, attention_mask_lm, input_ids_lm = self.prepare_llm_input(
            visual_embeddings,
            input_ids_q,
            attention_mask_q,
            input_ids_a,
            attention_mask_a,
        )
        # print(inputs_embeds_lm.size())
        # print(attention_mask_lm.size())
        # print(input_ids_lm.size())
       
        outputs = self.llm(
            inputs_embeds=inputs_embeds_lm,
            attention_mask=attention_mask_lm,
        )
        logits = outputs[0][:, :-1, :].contiguous()
        target = input_ids_lm[:, 1:].contiguous()
        # print(logits.shape)

        loss = self.criterion(logits.view(-1, self.llm.config.vocab_size), target.view(-1))

        return {
            "loss": loss,
        }

    @torch.no_grad
    def test_one_step(self, batch_data, requires_loss=True, requires_gen=True):
        volume_global = batch_data["volume_global"].to(self.device)
        input_ids_q = batch_data["input_ids_q"].to(self.device)
        attention_mask_q = batch_data["attention_mask_q"].to(self.device)
        input_ids_a = batch_data["input_ids_a"].to(self.device)
        attention_mask_a = batch_data["attention_mask_a"].to(self.device)
        bs = volume_global.shape[0]
        
        if requires_loss:
            visual_embeddings = self.produce_visual_embeddings(volume_global)
            

            inputs_embeds_lm, attention_mask_lm, input_ids_lm = self.prepare_llm_input(
                visual_embeddings,
                input_ids_q,
                attention_mask_q,
                input_ids_a,
                attention_mask_a,
            )
            outputs = self.llm(
                inputs_embeds=inputs_embeds_lm,
                attention_mask=attention_mask_lm,
                )
            logits = outputs[0][:, :-1, :].contiguous()
            target = input_ids_lm[:, 1:].contiguous()
            # print(logits.shape)

            loss = self.criterion(logits.view(-1, self.llm.config.vocab_size), target.view(-1))
        else:
            loss = None

        ### generate responces ###
        if requires_gen:
            visual_embeddings = self.produce_visual_embeddings(volume_global)
            inputs_embeds_q = self.llm.get_input_embeddings()(input_ids_q)
            pred_captions_batch = []
            gt_captions_batch = batch_data["answer"]
            if self.config_dt["task_type"] in ["report_gen", "close_vqa_yn", "open_vqa_yn", "close_vqa_mc", "open_vqa_mc"]:
                for i in range(bs):
                    cur_visual_embeddings = visual_embeddings[i]
                    
                    valid_q_length = attention_mask_q[i].sum().item()
                    cur_inputs_embeds_q = inputs_embeds_q[i, :valid_q_length, :]
                    valid_prompt_embeds = torch.cat([cur_visual_embeddings, cur_inputs_embeds_q], dim=0).unsqueeze(0)
                    
                    # print("valid_prompt_embeds", valid_prompt_embeds.size())
                    pred_captions = self.generate(valid_prompt_embeds, **self.generate_kwargs)
                    if len(pred_captions) > 1:
                        pred_captions_batch.append(pred_captions)
                    else:
                        pred_captions_batch.extend(pred_captions)
                    # cp_dict = compute_image_captioning_metric(pred_captions_all, gt_captions_all)
            elif self.config_dt["task_type"] == "":
                valid_length = attention_mask[0].sum().item()
                # print(valid_length)
                # print(attention_mask.sum(dim=1))
                valid_prompt_embeds = prompt_embeddings[:, :valid_length, :]
                # print("valid_prompt_embeds", valid_prompt_embeds.size())
                pred_captions = self.generate(valid_prompt_embeds, **self.generate_kwargs)
                pred_captions_batch.extend(pred_captions)
        else:
            gt_captions_batch = []
            pred_captions_batch = []
            
        return {
            "loss": loss,
        }, {
            "pred_captions_batch": pred_captions_batch,
            "gt_captions_batch": gt_captions_batch,
        }

    @torch.no_grad
    def generate(
        self,
        prompt_embeds,
        **generate_kwargs,
    ):
        batch_size = prompt_embeds.size()[0]
        prompt_attention_mask = torch.ones(
            prompt_embeds.size()[:-1], dtype=torch.long, device=prompt_embeds.device
        )

        start_tokens = [self.llm.config.bos_token_id]
        input_ids = torch.tensor([start_tokens], dtype=torch.long, device=prompt_embeds.device)
        input_ids = input_ids.repeat(batch_size, 1)

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        attention_mask = torch.ones_like(input_ids)
        
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds.to(prompt_embeds.device)], dim=1)
        attention_mask = torch.cat(
            [prompt_attention_mask, attention_mask.to(prompt_attention_mask.device)], dim=1
        )
        # print("inputs_embeds", inputs_embeds.size())
        # print("attention_mask", attention_mask.size())

        inputs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        inputs["input_ids"] = input_ids

        generate_kwargs["max_length"] = (
            generate_kwargs.get("max_length", 20) + prompt_embeds.shape[1] - 1
        )
        generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + prompt_embeds.shape[1]
        # print(generate_kwargs, generate_kwargs["max_length"])

        outputs = self.llm.generate(
            **inputs,
            **generate_kwargs,
            # do_sample=False,            #  Disable sampling → deterministic, faster
            # num_beams=1,                #  No beam search → faster (use greedy)
            
            # num_beams=5,
            # no_repeat_ngram_size=2,
            # early_stopping=True,

            do_sample=True,
            # top_k=50,
            top_p=0.90,
            num_return_sequences=1,

            use_cache=True,
            pad_token_id=self.llm.config.eos_token_id,
        )
        # print(outputs)

        captions = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        captions = [caption.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "") for caption in captions]
        # print(len(captions))
        # print(captions)
        return captions

    @torch.no_grad
    def test_on_dataloader(self, dataloader, gen_interval=20):

        self.llm.model.use_cache = True
        self.llm.model.gradient_checkpointing = False

        final_dict = OrderedDict()

        epoch_loss_dict = {}

        pred_captions_all = []
        gt_captions_all = []

        loss_steps = 0
        for i, batch_data in enumerate(tqdm(dataloader)):
            # continue
            requires_gen = True if i % gen_interval == 0 else False
            requires_loss = True if i % 1 == 0 else False

            loss_dict, output_dict = self.test_one_step(batch_data, requires_loss=requires_loss, requires_gen=requires_gen)

            # process loss items
            if loss_dict["loss"] is not None:
                loss_steps += 1
                for loss_name, loss_value in loss_dict.items():
                    if loss_name not in epoch_loss_dict.keys():
                        epoch_loss_dict[loss_name] = 0
                    epoch_loss_dict[loss_name] += loss_value.item()
            
            if len(output_dict["gt_captions_batch"]) != 0:
                pred_captions_all.extend(output_dict["pred_captions_batch"])
                gt_captions_all.extend(output_dict["gt_captions_batch"])
            # break

        for loss_name, loss_value in epoch_loss_dict.items():
            if loss_value is not None:
                epoch_loss_dict[loss_name] = loss_value / loss_steps
        final_dict.update(epoch_loss_dict)

        if len(gt_captions_all) != 0:
            # print(gt_captions_all)
            # print(len(pred_captions_all))
            # print(len(gt_captions_all))
            nlg_dict = nlg_metrics(
                pred_captions=pred_captions_all,
                gt_captions=gt_captions_all,
            )
            final_dict.update(nlg_dict)

        # print("pred_captions_all", pred_captions_all)
        # print("len gt_captions_all", len(gt_captions_all))
        
        return final_dict

    



if __name__ == "__main__":
    # print(monai.__file__)
    device = "cuda:3"
    # ckpt_path = "./checkpoints/clip3d_ssl/clip3d_ssl-vit-m3d_cap-dr1.0-E39-best/pytorch_model.bin"
    # ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # print(ckpt.keys())

    config_path = "./config_dt/rep_baseline.yaml" 
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = VQABaseline(config).to(device)
    model.device = device
    # print(model)

    data_root = os.path.abspath(DataPath.M3D_CAP)
    data_dir = os.path.join(data_root, "nii_down")
    csv_dir = os.path.abspath(DataPath.M3D_VQA)
    
    mt_transforms = MonaiTransforms(num_samples=1)
    global_transforms = mt_transforms.load_vqa_transforms(mode="clip3d")

    dataset = M3DVQABaselineDataset(data_dir, csv_dir, global_transforms, data_ratio=0.1, task_type="report_gen", mode="val", config_path=config_path)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False, collate_fn=collate_fn)

    first_batch = next(iter(dataloader))
    # print(first_batch["volume_global"].shape)
    for key in first_batch.keys():
        if key not in "answer":
            first_batch[key] = first_batch[key].to(device)
    # re = model(first_batch)
    # print(re)

    model.eval()
    # re = model.test_one_step(first_batch)
    # print(re)

    re = model.test_on_dataloader(dataloader, gen_interval=20)
    print_dict_content(re)