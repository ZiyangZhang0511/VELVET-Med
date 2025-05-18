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

from peft import LoraConfig
from peft import get_peft_model
from transformers import LlamaForCausalLM, AutoTokenizer

from utilities.metrics import nlg_metrics
from utilities.constants import DataPath, ConfigPath
from utilities.vision_transforms import MonaiTransforms
from utilities.utils import print_dict_content, count_params

from datasets import collate_fn
from datasets.text_utils import SentBasedTokenizer
from datasets.m3d_vqa import M3DVQADataset

from .networks import SentBertModel


class VQAModel(nn.Module):

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

        ### configure downstream model such as classifier or language model ###
        self.configure_dt_model()

        ### initialize model ###
        self.config_txt_model = config_txt_model
        self.initialize_model(config["pretrained_ckpt"])
        self.freeze_params(config_freeze)

        ### configure generate kwargs ###
        self.generate_kwargs = {
            "max_length": config["downstream_task"]["llm"]["gen_max_length"],
            "min_length": config["downstream_task"]["llm"]["gen_min_length"],
        }

        
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


    def initialize_model(self, pretrained_ckpt):
        if not os.path.isfile(pretrained_ckpt):
            # print(self.vision_encoder.layers1[0].blocks[0].attn.qkv.weight.max())
            print("No pre-traned weights loaded......")
            return
        
        state_dict = torch.load(pretrained_ckpt, map_location=self.device, weights_only=True)
        # print(state_dict.keys())

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

    def configure_dt_model(self):
        txt_hidden_size = self.config_dt["txt_hidden_size"]
        config_llm = self.config_dt["llm"]
        llm_hidden_size = self.config_dt["llm"]["hidden_size"]
        self.perceiver = nn.Sequential(
            nn.Linear(txt_hidden_size, llm_hidden_size),
            # nn.GELU(),
            # nn.LayerNorm(llm_hidden_size),
            # nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.configure_llm(config_llm)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

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


    def produce_mm_hidden_state(
        self, volume_global, input_ids, attention_mask, token_type_ids
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

        return outputs.last_hidden_state


    def prepare_llm_input(
        self,
        mm_hidden_state,
        attention_mask,
        input_ids_lm,
        attention_mask_lm,
    ):
        bs = mm_hidden_state.shape[0]
        prompt_embeddings = self.perceiver(mm_hidden_state)

        inputs_embeds_lm = self.llm.get_input_embeddings()(input_ids_lm)

        inputs_embeds_lm_list = []
        attention_mask_lm_list = []
        input_ids_lm_list = []
        for i in range(bs):
            cur_attention_mask = attention_mask[i]
            valid_length = cur_attention_mask.sum().item()
            valid_prompt_embeds = prompt_embeddings[i, :valid_length, :]

            cur_inputs_embeds_lm = torch.cat([valid_prompt_embeds, inputs_embeds_lm[i]], dim=0) 
            cur_attention_mask_lm = torch.cat([cur_attention_mask[:valid_length], attention_mask_lm[i]], dim=0) 

            max_length = self.config_dt["llm"]["max_length"]

            cur_inputs_embeds_lm = cur_inputs_embeds_lm[:max_length, :].unsqueeze(0)
            cur_attention_mask_lm = cur_attention_mask_lm[:max_length].unsqueeze(0)

            cur_prompt_ids = torch.full((valid_length,), -100).to(self.device)
            cur_input_ids_lm = torch.cat([cur_prompt_ids, input_ids_lm[i]], dim=0)
            indices = torch.where(cur_input_ids_lm == 128009)[0]
            if indices.numel() > 0:
                cur_input_ids_lm[indices] = -100
                cur_input_ids_lm[indices[0]] = 128009
            cur_input_ids_lm = cur_input_ids_lm[:max_length].unsqueeze(0)
            
            inputs_embeds_lm_list.append(cur_inputs_embeds_lm)
            attention_mask_lm_list.append(cur_attention_mask_lm)
            input_ids_lm_list.append(cur_input_ids_lm)

        inputs_embeds_lm = torch.cat(inputs_embeds_lm_list, dim=0)
        attention_mask_lm = torch.cat(attention_mask_lm_list, dim=0)
        input_ids_lm =  torch.cat(input_ids_lm_list, dim=0)
        # print("attention_mask_lm", attention_mask_lm.size(), attention_mask_lm[0])
        # print("input_ids_lm", input_ids_lm.size(), input_ids_lm[0])

        return inputs_embeds_lm, attention_mask_lm, input_ids_lm

    def forward(self, batch_data):

        volume_global = batch_data["volume_global"]
        input_ids = batch_data["input_ids"]
        token_type_ids = batch_data["token_type_ids"]
        attention_mask = batch_data["attention_mask"]
        input_ids_lm = batch_data["input_ids_lm"]
        attention_mask_lm = batch_data["attention_mask_lm"]
        bs = volume_global.shape[0]

        
        self.device = volume_global.device

        mm_hidden_state = self.produce_mm_hidden_state(
            volume_global,
            input_ids,
            attention_mask,
            token_type_ids,
        )

        inputs_embeds_lm, attention_mask_lm, input_ids_lm = self.prepare_llm_input(
            mm_hidden_state,
            attention_mask,
            input_ids_lm,
            attention_mask_lm,
        )
        
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
        input_ids = batch_data["input_ids"].to(self.device)
        token_type_ids = batch_data["token_type_ids"].to(self.device)
        attention_mask = batch_data["attention_mask"].to(self.device)
        input_ids_lm = batch_data["input_ids_lm"].to(self.device)
        attention_mask_lm = batch_data["attention_mask_lm"].to(self.device)
        bs = volume_global.shape[0]
        
        if requires_loss:
            mm_hidden_state = self.produce_mm_hidden_state(
                volume_global,
                input_ids,
                attention_mask,
                token_type_ids,
            )
            inputs_embeds_lm, attention_mask_lm, input_ids_lm = self.prepare_llm_input(
                mm_hidden_state,
                attention_mask,
                input_ids_lm,
                attention_mask_lm,
            )
            outputs = self.llm(
                inputs_embeds=inputs_embeds_lm,
                attention_mask=attention_mask_lm,
            )
            logits = outputs[0][:, :-1, :].contiguous()
            target = input_ids_lm[:, 1:].contiguous()

            loss = self.criterion(logits.view(-1, self.llm.config.vocab_size), target.view(-1))
        else:
            loss = None

        ### generate responces ###
        if requires_gen:
            mm_hidden_state = self.produce_mm_hidden_state(
                volume_global,
                input_ids,
                attention_mask,
                token_type_ids,
            )
            prompt_embeddings = self.perceiver(mm_hidden_state)
            pred_captions_batch = []
            gt_captions_batch = batch_data["answer"]
            if self.config_dt["task_type"] in ["report_gen", "close_vqa_yn", "open_vqa_yn", "close_vqa_mc", "open_vqa_mc"]:
                for i, cur_attention_mask in enumerate(attention_mask):
                    valid_length = cur_attention_mask.sum().item()
                    valid_prompt_embeds = prompt_embeddings[i, :valid_length, :].unsqueeze(0)
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
                # valid_prompt_embeds = prompt_embeddings[:, :valid_length, :]
                # # print("valid_prompt_embeds", valid_prompt_embeds.size())
                # pred_captions = self.generate(valid_prompt_embeds, **self.generate_kwargs)
                # pred_captions_batch.extend(pred_captions)
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
            
            num_beams=8,
            no_repeat_ngram_size=2,
            early_stopping=True,

            # do_sample=True,
            # top_k=50,
            # top_p=0.90,
            # num_return_sequences=1,

            use_cache=True,
            pad_token_id=self.llm.config.eos_token_id,
        )
        # print(outputs)

        captions = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        captions = [caption.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "") for caption in captions]
        # print(len(captions))
        # print(captions)
        return captions



if __name__ == "__main__":
    # from bert_score import list_models
    # print(list_models())

    device = "cuda:3"

    config_path = "./config_dt/open_vqa_mc_allitc_mmssl_visssl_txtssl_BtSv_r32.yaml" 
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = VQAModel(config).to(device)
    model.device = device
    # print(model)

    ckpt_path = "./checkpoints_dt/open_vqa_mc_allitc_mmssl_visssl_txtssl_BtSv_r32/open_vqa_mc-m3d_vqa-dr0.2-FvisTruetxtTruemmFalse-E9/pytorch_model.bin"
    # ./checkpoints_dt/open_vqa_mc_allitc_mmssl_visssl_txtssl_BtSv_r32/open_vqa_mc-m3d_vqa-dr0.2-FvisTruetxtTruemmFalse-E3-best/pytorch_model.bin
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)

    data_root = os.path.abspath(DataPath.M3D_CAP)
    data_dir = os.path.join(data_root, "nii_down")
    csv_dir = os.path.abspath(DataPath.M3D_VQA)
    
    mt_transforms = MonaiTransforms(num_samples=1)
    global_transforms = mt_transforms.load_vqa_transforms(mode="global")

    dataset = M3DVQADataset(data_dir, csv_dir, global_transforms, data_ratio=0.1, task_type="open_vqa_mc", mode="val", config_path=config_path)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=8, collate_fn=collate_fn)

    # first_batch = next(iter(dataloader))
    # # print(first_batch["volume_global"].shape)
    # for key in first_batch.keys():
    #     if key not in "answer":
    #         first_batch[key] = first_batch[key].to(device)
    # re = model(first_batch)
    # print(re)

    model.eval()
    # re = model.test_one_step(first_batch)
    # print(re)

    re = model.test_on_dataloader(dataloader, gen_interval=20)
    print_dict_content(re)