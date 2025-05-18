import os
import yaml
import json
import random
from tqdm.auto import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from monai.networks.nets.swin_unetr import SwinTransformer

from transformers import BertModel, BertConfig

from datasets import collate_fn
from datasets.m3d_cap import M3DCAPDataset
from datasets.text_utils import SentBasedTokenizer

from .text_model import TextSSL
from .vision_model import VisionSSL
from .networks import MLMHead, SentBertModel
from .losses import CLIPLoss, HierarchicalCLIPLoss

from utilities.utils import print_dict_content, count_params
from utilities.constants import ConfigPath, DataPath
from utilities.vision_transforms import MonaiTransforms
from utilities.metrics import multiclass_cls_metrics, compute_score_matrix, compute_RatK


class VisionTextSSL(nn.Module):
    """
    - first, implement cross-modal functions: hierarchical clip
    - second, consider multi-modal functions: match, mlm
    """
    def __init__(self, config, device="cuda"):
        super().__init__()

        self.device = device

        ### prepare configs ###
        config_vis_ssl = config["vision_ssl"]
        config_txt_ssl = config["text_ssl"]
        config_itc_module = config["itc_module"]
        config_mm_module = config["mm_module"]
        config_loss_weights = config["loss_weights"]

        self.config_itc_module = config_itc_module

        ### initialize its enable_modules and VisionSSL enable_modules ###
        self.config_enable = config["enable_modules"]
        config_vis_ssl["vision_ssl_module"]["enable_contras"] = self.config_enable["vision_contras"]
        config_vis_ssl["vision_ssl_module"]["enable_rot"] = self.config_enable["vision_rot"]
        config_vis_ssl["vision_ssl_module"]["enable_recon"] = self.config_enable["vision_recon"]

        config_txt_ssl["enable_txt_mlm"] = self.config_enable["text_ssl"]

        ### build vision ssl and text ssl ###
        self.vision_ssl = VisionSSL(config_vis_ssl)
        self.text_ssl = TextSSL(config_txt_ssl)


        ### config itc module ###
        self.configure_itc_module(config_itc_module)

        ### config mm module ###
        self.configure_mm_module(config_txt_ssl["text_model"])

        ### loss weights ###
        self.alpha_report_itc = config_itc_module["alpha_report_itc"]
        self.alpha_sentence_itc = config_itc_module["alpha_sentence_itc"]
        self.alpha_word_itc = config_itc_module["alpha_word_itc"]

        self.alpha_mm_mlm = config_mm_module["alpha_mm_mlm"]
        self.alpha_mm_match = config_mm_module["alpha_mm_match"]
        self.alpha_itc = config_loss_weights["alpha_itc"]

        self.alpha_mm_ssl = config_loss_weights["alpha_mm_ssl"]
        self.alpha_vis_ssl = config_loss_weights["alpha_vis_ssl"]
        self.alpha_txt_ssl = config_loss_weights["alpha_txt_ssl"]

        self.requires_match_score = config_mm_module["requires_match_score"] if config_mm_module.get("requires_match_score", None) is not None else False
   

    def configure_itc_module(self, config_itc_module):
        
        vis_feature_shape_list = self.vision_ssl.vis_feature_shape_list
        last_vis_feature_shape = vis_feature_shape_list[-1]
        middle_vis_feature_shape = vis_feature_shape_list[-2]
        bottom_vis_feature_shape = vis_feature_shape_list[-3]

        txt_hidden_size = self.text_ssl.config_txt_model["hidden_size"]
        # txt_hidden_size = 768

        ### report itc module ###
        if self.config_enable["report_itc"]:
            c, h, w, d = last_vis_feature_shape
            if config_itc_module["contrastive_bottleneck"] == "convolution":
                self.top_itc_bottleneck = nn.Conv3d(c, c, kernel_size=(h, w, d), stride=(h, w, d))
            elif config_itc_module["contrastive_bottleneck"] == "pooling":
                self.top_itc_bottleneck = nn.AvgPool3d(kernel_size=(h, w, d), stride=(h, w, d))
            elif config_itc_module["contrastive_bottleneck"] == "self-attention":
                pass
            self.top_itc_vis_head = nn.Linear(c, config_itc_module["contrastive_hidden_dim"])
            self.top_itc_txt_head = nn.Linear(txt_hidden_size, config_itc_module["contrastive_hidden_dim"])

            self.top_itc_criterion = CLIPLoss(
                config_itc_module["itc_temperature"], 
                config_itc_module["learnable_temp"],
                config_itc_module["alpha_i2t"],
            )

        ### sentence itc module ###
        if self.config_enable["sentence_itc"]:
            c, h, w, d = middle_vis_feature_shape
            self.middle_itc_vis_head = nn.Linear(c, config_itc_module["contrastive_hidden_dim"])
            self.middle_itc_txt_head = nn.Linear(txt_hidden_size, config_itc_module["contrastive_hidden_dim"])
            
            self.middle_itc_criterion = HierarchicalCLIPLoss(temp2=5.0, temp3=10.0, agg="sum")
        

        ### word itc module ###
        if self.config_enable["word_itc"]:
            c, h, w, d = bottom_vis_feature_shape
            self.bottom_itc_vis_head = nn.Linear(c, config_itc_module["contrastive_hidden_dim"])
            self.bottom_itc_txt_head = nn.Linear(txt_hidden_size, config_itc_module["contrastive_hidden_dim"])
            
            self.bottom_itc_criterion = HierarchicalCLIPLoss()

    def configure_mm_module(self, config_txt_model):

        if self.config_enable["mm_mlm"]:
            self.mm_mlm_head = MLMHead(config_txt_model)
            self.mm_mlm_criterion = nn.CrossEntropyLoss()

        if self.config_enable["mm_match"]:
            self.mm_match_head = nn.Linear(config_txt_model["hidden_size"], 2)
            self.mm_match_criterion = nn.CrossEntropyLoss()

        if self.config_enable["mm_match"] or self.config_enable["mm_mlm"]:
            vis_feature_shape_list = self.vision_ssl.vis_feature_shape_list
            c, h, w, d = vis_feature_shape_list[-1]
            txt_hidden_size = self.text_ssl.config_txt_model["hidden_size"]
            self.mm_adpater = nn.Sequential(
                nn.Linear(c, txt_hidden_size),
                nn.LayerNorm(txt_hidden_size),
            )

    def through_mm_mlm_module(self, mm_mlm_hidden_states):

        ### multimodal masked language modeling ###
        mm_mlm_logits = self.mm_mlm_head(mm_mlm_hidden_states)
        mm_mlm_logits = mm_mlm_logits.view(-1, self.text_ssl.config_txt_model["vocab_size"]).contiguous()

        return mm_mlm_logits
        

    def select_out_txt_embeddings(self, aggregated_embeddings, processed_sequence):
        """ 
        aggregated_embeddings of shape [bs, seq_len, hz]
        processed_sequence: a list (len bs) of lists of aggregated tokens
        """
        batch_size, seq_len, hz = aggregated_embeddings.size()

        report_embed_batch_list = []
        sentence_embed_batch_list = []
        word_embed_batch_list = []

        ### loop over batch ###
        for i in range(batch_size):
            cur_embeds = aggregated_embeddings[i] # [seq_len, hz]
            cur_sequence = processed_sequence[i] # a list with length of seq_len
            # print(cur_sequence, cur_embeds.size())

            report_indices = []
            sentence_indices = []
            word_indices = []

            for idx, (token, embed) in enumerate(zip(cur_sequence, cur_embeds)):
                # print(token, embed.size())

                if token != "[PAD]":
                    if token == "[CLS]":
                        report_indices.append(idx)
                    elif token.startswith("[SEN"):
                        sentence_indices.append(idx)
                    else:
                        word_indices.append(idx)
                else:
                    final_index = idx
                    break

            # print(report_indices)
            # print(sentence_indices)
            # print("word_indices", word_indices[-1], final_index)

            ### process report_embed ###
            report_embed = cur_embeds[report_indices]
            # report_embed = cur_embeds[:final_index].mean(dim=0, keepdim=True)
            # print(report_embed.size())

            ### process sentence_embed ###
            # sentence_embed = cur_embeds[sentence_indices]
            sentence_embed = []
            for i in range(len(sentence_indices)):
                if i == len(sentence_indices)-1:
                    one_sent_embed = cur_embeds[sentence_indices[i]:final_index].mean(dim=0, keepdim=True)
                else:
                    one_sent_embed = cur_embeds[sentence_indices[i]:sentence_indices[i+1]].mean(dim=0, keepdim=True)

                if torch.isnan(one_sent_embed).any():
                    pass
                    print("one_sent_embed", one_sent_embed)
                    print(sentence_indices, word_indices)
                    print(sentence_indices[i], word_indices[-1])
                    
                sentence_embed.append(one_sent_embed)

            if len(sentence_embed) < 1:
                sentence_embed = []
            else:
                sentence_embed = torch.cat(sentence_embed, dim=0)
            # print(sentence_embed.size())
            
            ### process word_embed ###
            word_embed = cur_embeds[word_indices]
            # print(word_embed.size())

            report_embed_batch_list.append(report_embed)
            sentence_embed_batch_list.append(sentence_embed)
            word_embed_batch_list.append(word_embed)
            
        return report_embed_batch_list, sentence_embed_batch_list, word_embed_batch_list


    def through_itc_module(self, all_vis_features, aggregated_embeddings, processed_sequence):
        
        ### first get report, sentence, word embeds
        report_embed_batch_list, sentence_embed_batch_list, word_embed_batch_list = self.select_out_txt_embeddings(aggregated_embeddings, processed_sequence)

        multiple_vis_embeddings = []
        multiple_txt_embeddings = []

        ### compute top itc embeddings ###
        if self.config_enable["report_itc"]:
            last_vis_features = all_vis_features[-1]
            report_embeddings = torch.cat(report_embed_batch_list, dim=0)
            
            top_vis_embeddings = self.top_itc_bottleneck(last_vis_features).squeeze(dim=(2, 3, 4))
            top_vis_embeddings = self.top_itc_vis_head(top_vis_embeddings)
            top_txt_embeddings = self.top_itc_txt_head(report_embeddings)
            
            multiple_vis_embeddings.append(top_vis_embeddings)
            multiple_txt_embeddings.append(top_txt_embeddings)
        else:
            multiple_vis_embeddings.append(None)
            multiple_txt_embeddings.append(None)


        ### compute middle itc embeddings ###
        if self.config_enable["sentence_itc"]:
            # sentence_embed_batch_list
            middle_vis_features = all_vis_features[-2]
            middle_vis_features = middle_vis_features.permute(0, 2, 3, 4, 1).contiguous() # [bs, h, w, d, c]
            # [bs, h, w, d, c]
            middle_vis_embeddings = self.middle_itc_vis_head(middle_vis_features)
            
            middle_txt_embeddings_list = []
            for sentence_embeds in sentence_embed_batch_list:
                middle_txt_embeddings_list.append(self.middle_itc_txt_head(sentence_embeds))
            # print(len(middle_txt_embeddings_list), middle_txt_embeddings_list[1].size(), middle_vis_embeddings.shape)

            multiple_vis_embeddings.append(middle_vis_embeddings)
            multiple_txt_embeddings.append(middle_txt_embeddings_list)
        else:
            multiple_vis_embeddings.append(None)
            multiple_txt_embeddings.append(None)

        ### compute bottom itc embeddings ###
        if self.config_enable["word_itc"]:

            bottom_vis_features = all_vis_features[-3]
            bottom_vis_features = bottom_vis_features.permute(0, 2, 3, 4, 1).contiguous() # [bs, h, w, d, c]
            # [bs, h, w, d, c]
            bottom_vis_embeddings = self.bottom_itc_vis_head(bottom_vis_features)
            
            bottom_txt_embeddings_list = []
            for word_embeds in word_embed_batch_list:
                bottom_txt_embeddings_list.append(self.bottom_itc_txt_head(word_embeds))
            # print(len(bottom_txt_embeddings_list), bottom_txt_embeddings_list[1].size(), bottom_vis_embeddings.shape)

            multiple_vis_embeddings.append(bottom_vis_embeddings)
            multiple_txt_embeddings.append(bottom_txt_embeddings_list)
        else:
            multiple_vis_embeddings.append(None)
            multiple_txt_embeddings.append(None)
            

        return multiple_vis_embeddings, multiple_txt_embeddings


    def through_mm_match_module(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        txt_hidden_state,
        encoder_hidden_states,
        encoder_attention_mask,
        top_vis_embeddings,
        top_txt_embeddings,
    ):
        # positive_mm_embeddings of shape (bs, hz)
        positive_mm_embeddings = self.text_ssl.extract_features(
            None,
            token_type_ids,
            attention_mask,
            inputs_embeds=txt_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            mode="mm_match",
        )
        # print(positive_mm_embeddings.size())

        top_vis_embeds_normalized = F.normalize(top_vis_embeddings, dim=-1)
        top_txt_embeds_normalized = F.normalize(top_txt_embeddings, dim=-1)
        # print(self.top_itc_criterion.temperature)
        sim_v2t = (top_vis_embeds_normalized @ top_txt_embeds_normalized.t()) / self.top_itc_criterion.temperature
        sim_t2v = (top_txt_embeds_normalized @ top_vis_embeds_normalized.t()) / self.top_itc_criterion.temperature
        # print(sim_v2t.size(), sim_t2v.size())
        with torch.no_grad():
            batch_size = top_vis_embeddings.size(0)          
            weights_v2t = F.softmax(sim_v2t[:,:batch_size],dim=1)
            weights_t2v = F.softmax(sim_t2v[:,:batch_size],dim=1)
            weights_v2t.fill_diagonal_(0)
            weights_t2v.fill_diagonal_(0)

        # pick hard negative sample for each text
        vis_hidden_state_neg = []    
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_t2v[b], 1).item()
            vis_hidden_state_neg.append(encoder_hidden_states[neg_idx])
        vis_hidden_state_neg = torch.stack(vis_hidden_state_neg,dim=0)  
        # print(vis_hidden_state_neg.size()) 

        # pick hard negative sample for each volume
        txt_hidden_state_neg = []
        txt_token_type_ids_neg = []
        txt_atts_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_v2t[b], 1).item()
            txt_hidden_state_neg.append(txt_hidden_state[neg_idx])
            txt_token_type_ids_neg.append(token_type_ids[neg_idx])
            txt_atts_neg.append(attention_mask[neg_idx])
        txt_hidden_state_neg = torch.stack(txt_hidden_state_neg, dim=0)   
        txt_token_type_ids_neg = torch.stack(txt_token_type_ids_neg, dim=0)
        txt_atts_neg = torch.stack(txt_atts_neg, dim=0)  
        # print(txt_hidden_state_neg.size())

        txt_hidden_state_all = torch.cat([txt_hidden_state, txt_hidden_state_neg], dim=0)
        txt_token_type_ids_all = torch.cat([token_type_ids, txt_token_type_ids_neg], dim=0) 
        txt_atts_all = torch.cat([attention_mask, txt_atts_neg], dim=0)     

        vis_hidden_state_all = torch.cat([vis_hidden_state_neg, encoder_hidden_states], dim=0)
        vis_atts_all = torch.cat([encoder_attention_mask, encoder_attention_mask], dim=0)

        negative_mm_embeddings = self.text_ssl.extract_features(
            None,
            txt_token_type_ids_all,
            txt_atts_all,
            inputs_embeds=txt_hidden_state_all,
            encoder_hidden_states=vis_hidden_state_all,
            encoder_attention_mask=vis_atts_all,
            mode="mm_match",
        )
        # print(negative_mm_embeddings.size())

        mm_match_embeddings = torch.cat([positive_mm_embeddings, negative_mm_embeddings], dim=0)
        mm_match_logits = self.mm_match_head(mm_match_embeddings)            

        mm_match_target = torch.cat(
            [
                torch.ones(batch_size, dtype=torch.long), 
                torch.zeros(2*batch_size, dtype=torch.long),
            ], dim=0
        ).to(mm_match_embeddings.device)

        return mm_match_logits, mm_match_target


    def forward(self, batch_data):
        """
        - batch_data["volume_vl"] [bs, in_chans, H, W, D]
        - batch_data["volume_vis"] [bs, in_chans, H, W, D]
        - batch_data["input_ids"] [bs, seq_len]
        - batch_data["token_type_ids"] [bs, seq_len]
        - batch_data["attention_mask"] [bs, seq_len]
        """

        volume_vl = batch_data["volume_vl"]
        volume_vis = batch_data["volume_vis"]
        input_ids = batch_data["input_ids"]
        token_type_ids = batch_data["token_type_ids"]
        attention_mask = batch_data["attention_mask"]
        self.device = volume_vl.device

        # print(input_ids[0, :])
        # print(attention_mask[0, :])
        # print("input_ids", input_ids[0, :5])
        # print("attention_mask", attention_mask[0, :5])
        # print("token_type_ids", token_type_ids[0, :5])

        loss_dict = OrderedDict()
        loss_dict["loss"] = 0
        loss_placeholder = torch.tensor(0, dtype=volume_vl.dtype, device=volume_vl.device, requires_grad=False)

        ###======== do cross-modal ssl (essential part) ========###
        all_vis_features = self.vision_ssl.extract_vis_features(volume_vl)
        
        if self.config_enable["sentence_itc"] or self.config_enable["word_itc"]:
            # aggregated_embeddings of shape [bs, seq_len, hz]
            aggregated_embeddings, processed_sequence, txt_hidden_state = self.text_ssl.extract_features(
                input_ids,
                token_type_ids,
                attention_mask,
                mode="itc_all",
                last_n_layers_for_itc=self.config_itc_module["last_n_layers_for_itc"],
            )
            multiple_vis_embeddings, multiple_txt_embeddings = self.through_itc_module(
                all_vis_features, aggregated_embeddings, processed_sequence
            )

        else:
            report_embeddings, txt_hidden_state = self.text_ssl.extract_features(
                input_ids,
                token_type_ids,
                attention_mask,
                mode="itc",
                last_n_layers_for_itc=self.config_itc_module["last_n_layers_for_itc"],
            )
            last_vis_features = all_vis_features[-1]
            
            top_vis_embeddings = self.top_itc_bottleneck(last_vis_features).squeeze(dim=(2, 3, 4))
            top_vis_embeddings = self.top_itc_vis_head(top_vis_embeddings)
            top_txt_embeddings = self.top_itc_txt_head(report_embeddings)

            multiple_vis_embeddings = [top_vis_embeddings, None, None]
            multiple_txt_embeddings = [top_txt_embeddings, None, None]


        top_vis_embeddings, top_txt_embeddings = list(zip(multiple_vis_embeddings, multiple_txt_embeddings))[0]
        loss_report_itc = self.top_itc_criterion(top_vis_embeddings, top_txt_embeddings) if self.config_enable["report_itc"] else loss_placeholder
        
        middle_vis_embeddings, middle_txt_embeddings_list = list(zip(multiple_vis_embeddings, multiple_txt_embeddings))[1]
        loss_sentence_itc, _ = self.middle_itc_criterion(middle_vis_embeddings, middle_txt_embeddings_list) if self.config_enable["sentence_itc"] else (loss_placeholder, None)

        bottome_vis_embeddings, bottom_txt_embeddings_list = list(zip(multiple_vis_embeddings, multiple_txt_embeddings))[2]
        loss_word_itc, _ = self.bottom_itc_criterion(bottome_vis_embeddings, bottom_txt_embeddings_list) if self.config_enable["word_itc"] else (loss_placeholder, None)

        loss_itc = self.alpha_report_itc*loss_report_itc +\
                   self.alpha_sentence_itc*loss_sentence_itc +\
                   self.alpha_word_itc*loss_word_itc

        loss_dict["loss_itc"] = loss_itc
        loss_dict["loss_report_itc"] = loss_report_itc
        loss_dict["loss_sentence_itc"] = loss_sentence_itc
        loss_dict["loss_word_itc"] = loss_word_itc

        
        ###======== do multi-modal ssl ========###
        if self.config_enable["mm_ssl"]:
            encoder_hidden_states = all_vis_features[-1].flatten(start_dim=2, end_dim=-1)
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1).contiguous() # (bs, seq_len_enc, hz)
            # multi-modal adpater
            # print("before", encoder_hidden_states.shape)
            encoder_hidden_states = self.mm_adpater(encoder_hidden_states)
            # print("after", encoder_hidden_states.shape)
            encoder_hidden_shape = encoder_hidden_states.shape[:2]
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=encoder_hidden_states.device)

        if self.config_enable["mm_match"]:
            mm_match_logits, mm_match_target = self.through_mm_match_module(
                input_ids,
                token_type_ids,
                attention_mask,
                txt_hidden_state,
                encoder_hidden_states,
                encoder_attention_mask,
                top_vis_embeddings,
                top_txt_embeddings,
            )
            loss_mm_match = self.mm_match_criterion(mm_match_logits, mm_match_target)
        else: 
            loss_mm_match = loss_placeholder
        

        # mm_mlm_hidden_states is for mm mlm
        if self.config_enable["mm_mlm"]:
            masked_input_ids, masked_target_ids = self.text_ssl.mask_input_ids(
                input_ids.clone(), self.text_ssl.config_txt_model["mlm_probability"]
            )
            mm_mlm_hidden_states = self.text_ssl.extract_features(
                masked_input_ids,
                token_type_ids,
                attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                mode="mm_mlm",
            )

            mm_mlm_logits = self.through_mm_mlm_module(mm_mlm_hidden_states)
            masked_target_ids = masked_target_ids.view(-1).contiguous()
            loss_mm_mlm = self.mm_mlm_criterion(mm_mlm_logits, masked_target_ids)
        else:
           loss_mm_mlm = loss_placeholder


        loss_mm_ssl = self.alpha_mm_mlm*loss_mm_mlm +\
                      self.alpha_mm_match*loss_mm_match

        loss_dict["loss_mm_ssl"] = loss_mm_ssl
        loss_dict["loss_mm_mlm"] = loss_mm_mlm
        loss_dict["loss_mm_match"] = loss_mm_match

        ###======== do vision uni-modal ssl ========###
        if self.config_enable["vision_ssl"]:
            vis_ssl_loss_dict = self.vision_ssl(batch_data)
            loss_vis_ssl = vis_ssl_loss_dict["loss"]
            loss_contras = vis_ssl_loss_dict["loss_contras"]
            loss_rot = vis_ssl_loss_dict["loss_rot"]
            loss_recon = vis_ssl_loss_dict["loss_recon"]

        else:
            loss_vis_ssl = loss_placeholder
            loss_contras = loss_placeholder
            loss_rot = loss_placeholder
            loss_recon = loss_placeholder
        
        loss_dict["loss_vis_ssl"] = loss_vis_ssl
        loss_dict["loss_contras"] = loss_contras
        loss_dict["loss_rot"] = loss_rot
        loss_dict["loss_recon"] = loss_recon

        ###======== do text uni-modal ssl ========###
        if self.config_enable["text_ssl"]:
            txt_ssl_loss_dict = self.text_ssl(batch_data)
            loss_txt_ssl = txt_ssl_loss_dict["loss"]
        else:
            loss_txt_ssl = loss_placeholder
        loss_dict["loss_txt_ssl"] = loss_txt_ssl


        loss = self.alpha_itc*loss_itc +\
               self.alpha_mm_ssl*loss_mm_ssl +\
               self.alpha_vis_ssl*loss_vis_ssl +\
               self.alpha_txt_ssl*loss_txt_ssl

        loss_dict["loss"] = loss
        # loss_dict["loss"] = loss_report_itc
        return loss_dict


    @torch.no_grad
    def test_one_step(self, batch_data):

        volume_vl = batch_data["volume_vl"].to(self.device)
        volume_vis = batch_data["volume_vis"].to(self.device)
        input_ids = batch_data["input_ids"].to(self.device)
        token_type_ids = batch_data["token_type_ids"].to(self.device)
        attention_mask = batch_data["attention_mask"].to(self.device)

        loss_dict = OrderedDict()
        loss_dict["loss"] = 0
        loss_placeholder = torch.tensor(0, dtype=volume_vl.dtype, device=volume_vl.device, requires_grad=False)

        output_dict = OrderedDict()

        ###======== do cross-modal ssl (essential part) ========###
        all_vis_features = self.vision_ssl.extract_vis_features(volume_vl)

        if self.config_enable["sentence_itc"] or self.config_enable["word_itc"]:
            # aggregated_embeddings of shape [bs, seq_len, hz]
            aggregated_embeddings, processed_sequence, txt_hidden_state = self.text_ssl.extract_features(
                input_ids,
                token_type_ids,
                attention_mask,
                mode="itc_all",
                last_n_layers_for_itc=self.config_itc_module["last_n_layers_for_itc"],
            )
            multiple_vis_embeddings, multiple_txt_embeddings = self.through_itc_module(
                all_vis_features, aggregated_embeddings, processed_sequence
            )

        else:
            report_embeddings, txt_hidden_state = self.text_ssl.extract_features(
                input_ids,
                token_type_ids,
                attention_mask,
                mode="itc",
                last_n_layers_for_itc=self.config_itc_module["last_n_layers_for_itc"],
            )
            last_vis_features = all_vis_features[-1]
            
            top_vis_embeddings = self.top_itc_bottleneck(last_vis_features).squeeze(dim=(2, 3, 4))
            top_vis_embeddings = self.top_itc_vis_head(top_vis_embeddings)
            top_txt_embeddings = self.top_itc_txt_head(report_embeddings)

            multiple_vis_embeddings = [top_vis_embeddings, None, None]
            multiple_txt_embeddings = [top_txt_embeddings, None, None]


        top_vis_embeddings, top_txt_embeddings = list(zip(multiple_vis_embeddings, multiple_txt_embeddings))[0]
        loss_report_itc = self.top_itc_criterion(top_vis_embeddings, top_txt_embeddings) if self.config_enable["report_itc"] else loss_placeholder
        
        middle_vis_embeddings, middle_txt_embeddings_list = list(zip(multiple_vis_embeddings, multiple_txt_embeddings))[1]
        loss_sentence_itc, _ = self.middle_itc_criterion(middle_vis_embeddings, middle_txt_embeddings_list) if self.config_enable["sentence_itc"] else (loss_placeholder, None)

        bottom_vis_embeddings, bottom_txt_embeddings_list = list(zip(multiple_vis_embeddings, multiple_txt_embeddings))[2]
        loss_word_itc, _ = self.bottom_itc_criterion(bottom_vis_embeddings, bottom_txt_embeddings_list) if self.config_enable["word_itc"] else (loss_placeholder, None)

        loss_itc = self.alpha_report_itc*loss_report_itc +\
                   self.alpha_sentence_itc*loss_sentence_itc +\
                   self.alpha_word_itc*loss_word_itc

        loss_dict["loss_itc"] = loss_itc
        loss_dict["loss_report_itc"] = loss_report_itc
        loss_dict["loss_sentence_itc"] = loss_sentence_itc
        loss_dict["loss_word_itc"] = loss_word_itc

        if self.config_enable["report_itc"]:
            output_dict["top_vis_embeds"] = top_vis_embeddings.detach().cpu()
            output_dict["top_txt_embeds"] = top_txt_embeddings.detach().cpu()
        if self.config_enable["sentence_itc"]:
            output_dict["middle_vis_embeds"] = middle_vis_embeddings.detach().cpu()
            output_dict["middle_txt_embeds_list"] = [item.detach().cpu() for item in middle_txt_embeddings_list]
        if self.config_enable["word_itc"]:
            output_dict["bottom_vis_embeds"] = bottom_vis_embeddings.detach().cpu()
            output_dict["bottom_txt_embeds_list"] = [item.detach().cpu() for item in bottom_txt_embeddings_list]
        
        ###======== do multi-modal ssl ========###
        if self.config_enable["mm_ssl"]:
            encoder_hidden_states = all_vis_features[-1].flatten(start_dim=2, end_dim=-1)
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1).contiguous()
            # multi-modal adpater
            encoder_hidden_states = self.mm_adpater(encoder_hidden_states)
            encoder_hidden_shape = encoder_hidden_states.shape[:2]
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=encoder_hidden_states.device)
        
        if self.config_enable["mm_match"]:
            mm_match_logits, mm_match_target = self.through_mm_match_module(
                None,
                token_type_ids,
                attention_mask,
                txt_hidden_state,
                encoder_hidden_states,
                encoder_attention_mask,
                top_vis_embeddings,
                top_txt_embeddings,
            )
            loss_mm_match = self.mm_match_criterion(mm_match_logits, mm_match_target)

            output_dict["txt_hidden_states"] = txt_hidden_state
            output_dict["token_type_ids"] = token_type_ids
            output_dict["attention_mask"] = attention_mask
            output_dict["vis_features"] = encoder_hidden_states
            output_dict["vis_attention_mask"] = encoder_attention_mask
        else: 
            loss_mm_match = loss_placeholder
        

        # mm_mlm_hidden_states is for mm mlm
        if self.config_enable["mm_mlm"]:
            masked_input_ids, masked_target_ids = self.text_ssl.mask_input_ids(
                input_ids.clone(), self.text_ssl.config_txt_model["mlm_probability"]
            )
            mm_mlm_hidden_states = self.text_ssl.extract_features(
                masked_input_ids,
                token_type_ids,
                attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                mode="mm_mlm",
            )

            mm_mlm_logits = self.through_mm_mlm_module(mm_mlm_hidden_states)
            masked_target_ids = masked_target_ids.view(-1).contiguous()
            loss_mm_mlm = self.mm_mlm_criterion(mm_mlm_logits, masked_target_ids)
        else:
           loss_mm_mlm = loss_placeholder

        loss_mm_ssl = self.alpha_mm_mlm*loss_mm_mlm +\
                      self.alpha_mm_match*loss_mm_match

        loss_dict["loss_mm_ssl"] = loss_mm_ssl
        loss_dict["loss_mm_mlm"] = loss_mm_mlm
        loss_dict["loss_mm_match"] = loss_mm_match

        ###======== do vision uni-modal ssl ========###
        if self.config_enable["vision_ssl"]:
            vis_ssl_loss_dict, vis_ssl_output_dict = self.vision_ssl.test_one_step(batch_data)
            loss_vis_ssl = vis_ssl_loss_dict["loss"]
            loss_contras = vis_ssl_loss_dict["loss_contras"]
            loss_rot = vis_ssl_loss_dict["loss_rot"]
            loss_recon = vis_ssl_loss_dict["loss_recon"]
        else:
            loss_vis_ssl = loss_placeholder
            loss_contras = loss_placeholder
            loss_rot = loss_placeholder
            loss_recon = loss_placeholder
            vis_ssl_output_dict = OrderedDict()

        loss_dict["loss_vis_ssl"] = loss_vis_ssl
        loss_dict["loss_contras"] = loss_contras
        loss_dict["loss_rot"] = loss_rot
        loss_dict["loss_recon"] = loss_recon
        output_dict.update(vis_ssl_output_dict)

        ###======== do text uni-modal ssl ========###
        if self.config_enable["text_ssl"]:
            txt_ssl_loss_dict, _ = self.text_ssl.test_one_step(batch_data)
            loss_txt_ssl = txt_ssl_loss_dict["loss"]
        else:
            loss_txt_ssl = loss_placeholder

        loss_dict["loss_txt_ssl"] = loss_txt_ssl

        loss = self.alpha_itc*loss_itc +\
               self.alpha_mm_ssl*loss_mm_ssl +\
               self.alpha_txt_ssl*loss_txt_ssl +\
               self.alpha_vis_ssl*loss_vis_ssl
        loss_dict["loss"] = loss

        # loss_dict["loss"] = loss_report_itc
        return loss_dict, output_dict


    @torch.no_grad
    def test_on_dataloader(self, dataloader, training_step, recon_save_dir=None, requires_match_score=False, similarity_type=None):
        
        if self.requires_match_score is not None and requires_match_score == False:
            requires_match_score = self.requires_match_score
        # print("requires_match_score", requires_match_score)

        test_steps = len(dataloader)

        epoch_loss_dict = {}

        top_vis_embeds_all = []
        top_txt_embeds_all = []
        middle_vis_embeds_all = []
        middle_txt_embeds_list_all = []
        bottom_vis_embeds_all = []
        bottom_txt_embeds_list_all = []

        txt_hidden_state_all = []
        token_type_ids_all = []
        attention_mask_all = []
        vis_features_all = []
        vis_attention_mask_all= []

        logits_rot_all = []
        target_rot_all = []
        volume_recon_list = []
        volume_target_list = []

        ### structure final dict to return ###
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
            if self.vision_ssl.enable_rot:
                logits_rot_all.append(output_dict["logits_rot"])
                target_rot_all.append(output_dict["target_rot"])
            if self.vision_ssl.enable_recon:
                volume_recon_list.append(output_dict["volume_recon"])
                volume_target_list.append(output_dict["volume_target"])
            if self.config_enable["report_itc"]:
                top_vis_embeds_all.append(output_dict["top_vis_embeds"])
                top_txt_embeds_all.append(output_dict["top_txt_embeds"])
            if self.config_enable["sentence_itc"]:
                middle_vis_embeds_all.append(output_dict["middle_vis_embeds"])
                middle_txt_embeds_list_all.extend(output_dict["middle_txt_embeds_list"])
            if self.config_enable["word_itc"]:
                bottom_vis_embeds_all.append(output_dict["bottom_vis_embeds"])
                bottom_txt_embeds_list_all.extend(output_dict["bottom_txt_embeds_list"])
            if self.config_enable["mm_match"]:
                txt_hidden_state_all.append(output_dict["txt_hidden_states"])
                token_type_ids_all.append(output_dict["token_type_ids"])
                attention_mask_all.append(output_dict["attention_mask"])
                vis_features_all.append(output_dict["vis_features"])
                vis_attention_mask_all.append(output_dict["vis_attention_mask"])
                
        final_dict.update(epoch_loss_dict)

        ### cross-modal retrival ###
        if self.config_enable["report_itc"]:

            similarity_type = self.config_itc_module["similarity_type"] if similarity_type is None else similarity_type
            # print("similarity_type", similarity_type)

            top_vis_embeds_all = torch.cat(top_vis_embeds_all, dim=0)
            top_txt_embeds_all = torch.cat(top_txt_embeds_all, dim=0)
            top_similarity_v2t = self.compute_global_similarity(top_vis_embeds_all, top_txt_embeds_all)
            
            k_test = top_txt_embeds_all.shape[0] if top_txt_embeds_all.shape[0] < 256 else 256
            # print(k_test, top_vis_embeds_all.shape, top_txt_embeds_all.shape)

            if self.config_enable["sentence_itc"] and similarity_type in ["m", "tm", "tmb"]:
                middle_vis_embeds_all = torch.cat(middle_vis_embeds_all, dim=0)
                middle_similarity_v2t = self.compute_local_similarity(middle_vis_embeds_all, middle_txt_embeds_list_all, level="middle")
            if self.config_enable["word_itc"] and similarity_type in ["b", "tb", "tmb"]:
                bottom_vis_embeds_all = torch.cat(bottom_vis_embeds_all, dim=0)
                bottom_similarity_v2t = self.compute_local_similarity(bottom_vis_embeds_all, bottom_txt_embeds_list_all, level="bottom")

            if requires_match_score:
                txt_hidden_state_all = torch.cat(txt_hidden_state_all, dim=0)
                token_type_ids_all = torch.cat(token_type_ids_all, dim=0)
                attention_mask_all= torch.cat(attention_mask_all, dim=0)
                vis_features_all = torch.cat(vis_features_all, dim=0)
                vis_attention_mask_all = torch.cat(vis_attention_mask_all, dim=0)
            else:
                txt_hidden_state_all = None
                token_type_ids_all = None
                attention_mask_all= None
                vis_features_all = None
                vis_attention_mask_all = None

            
            if similarity_type == "t":
                similarity_v2t = top_similarity_v2t
            elif similarity_type == "m":
                similarity_v2t = middle_similarity_v2t
            elif similarity_type == "b":
                similarity_v2t = bottom_similarity_v2t
            elif similarity_type == "tb":
                similarity_v2t = (top_similarity_v2t+bottom_similarity_v2t) / 2
            elif similarity_type == "tm":
                similarity_v2t = (top_similarity_v2t+middle_similarity_v2t) / 2
            elif similarity_type == "tmb":
                similarity_v2t = (top_similarity_v2t+middle_similarity_v2t+bottom_similarity_v2t) / 3

            retrieval_dict = self.compute_retrieval_metrics(
                vis_embeds=None,
                txt_embeds=None,
                similarity_matrix_v2t=similarity_v2t,
                k_test=k_test,
                requires_match_score=requires_match_score,
                txt_hidden_state_all=txt_hidden_state_all,
                token_type_ids_all=token_type_ids_all,
                attention_mask_all=attention_mask_all,
                vis_features_all=vis_features_all,
                vis_attention_mask_all=vis_attention_mask_all,
            )

            final_dict.update(retrieval_dict)

        ### rotation prediction metrics ###
        if self.vision_ssl.enable_rot:
            logits_rot_all = torch.cat(logits_rot_all, axis=0)
            probs_rot_all = F.softmax(logits_rot_all, dim=-1).numpy()
            target_rot_all = torch.cat(target_rot_all, axis=0).numpy()
            cls_metric_dict = multiclass_cls_metrics(probs_rot_all, target_rot_all)
            final_dict["rot_accuracy"] = cls_metric_dict["accuracy"]
            final_dict["rot_auc"] = cls_metric_dict["auc"]

        ### visualize reconstruction ###
        if recon_save_dir and self.vision_ssl.enable_recon:
            if training_step % self.vision_ssl.save_recon_interval == 0:
                for i, (volume_recon, volume_target) in enumerate(zip(volume_recon_list, volume_target_list)):
                    if i % (test_steps//3) == 0:
                        save_image(volume_recon.permute(3, 0, 1, 2).contiguous(), f"{recon_save_dir}/recon-{training_step}-idx{i}.jpg")
                        save_image(volume_target.permute(3, 0, 1, 2).contiguous(), f"{recon_save_dir}/target-{training_step}-idx{i}.jpg")

        return final_dict


    @staticmethod
    def compute_global_similarity(vis_embeds, txt_embeds):
        vis_embeds_normalized = F.normalize(vis_embeds, dim=-1)
        txt_embeds_normalized = F.normalize(txt_embeds, dim=-1)
        sims_matrix_v2t = vis_embeds_normalized @ txt_embeds_normalized.t()
        return sims_matrix_v2t

    @torch.no_grad
    def compute_local_similarity(self, vis_embeds, txt_embeds_list, level="word"):
        """
        vis_embeds [bs, h, w, d, embed_size]
        txt_embeds_list: a list of bs tensors [token_len (word or sentence), embed_size]
        """

        if level == "middle":
            criterion = self.middle_itc_criterion
        elif level == "bottom":
            criterion = self.bottom_itc_criterion
        else:
            raise ValueError("No designated level !!!")

        att_maps = []
        similarities = []

        batch_size = vis_embeds.shape[0]
        # [bs, h, w, d, embed_size]
        vis_embeds = F.normalize(vis_embeds, dim=-1)
        # [bs, embed_size, h, w, d]
        vis_embeds = vis_embeds.permute(0, 4, 1, 2, 3).contiguous()

        for i in range(batch_size):
            # [bs, num_tokens, embed_size]
            txt_embeds = txt_embeds_list[i].unsqueeze(0).repeat(batch_size, 1, 1).contiguous()

            txt_embeds = F.normalize(txt_embeds, dim=-1)
            txt_embeds = txt_embeds.permute(0, 2, 1).contiguous() # [bs, embed_size, num_tokens]
            # print(vis_embeds.size(), txt_embeds.size())

            num_tokens = txt_embeds.shape[-1]

            weiContext, attn = criterion.attention_fn(
                txt_embeds, vis_embeds, criterion.temp1
            )  # [48, 512, 25], [48, 25, 6, 6, 6]
            # print(weiContext.size(), attn.size())
            # return

            att_maps.append(
                attn[i].unsqueeze(0).contiguous()
            )  # add attention for curr index  [25, 6, 6, 6]
            txt_embeds = txt_embeds.transpose(1, 2).contiguous()  # [48, 25, 512]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 512]

            txt_embeds = txt_embeds.view(batch_size * num_tokens, -1)  # [1200, 512]
            weiContext = weiContext.view(batch_size * num_tokens, -1)  # [1200, 512]

            row_sim = criterion.custom_cosine_similarity(txt_embeds, weiContext)
            row_sim = row_sim.view(batch_size, num_tokens)  # [48, 25]

            row_sim.mul_(criterion.temp2).exp_()
            if criterion.agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)  #
        similarities = similarities * criterion.temp3

        return similarities

    def compute_retrieval_metrics(
        self,
        vis_embeds=None,
        txt_embeds=None,
        similarity_matrix_v2t=None,
        k_test=256,
        requires_match_score=False,
        txt_hidden_state_all=None,
        token_type_ids_all=None,
        attention_mask_all=None,
        vis_features_all=None,
        vis_attention_mask_all=None,
    ):
        
        retrieval_dict = OrderedDict()
        # print("computing retrieval metrics")

        score_matrix_v2t_np, score_matrix_t2v_np = self.compute_score_matrix(
            vis_embeds=vis_embeds,
            txt_embeds=txt_embeds,
            similarity_matrix_v2t=similarity_matrix_v2t,
            k_test=k_test,
            requires_match_score=requires_match_score,
            txt_hidden_state=txt_hidden_state_all,
            token_type_ids=token_type_ids_all,
            attention_mask=attention_mask_all,
            vis_features=vis_features_all,
            vis_attention_mask=vis_attention_mask_all,
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

    @torch.no_grad
    def compute_score_matrix(
        self,
        vis_embeds=None,
        txt_embeds=None,
        similarity_matrix_v2t=None,
        k_test=256,
        requires_match_score=False,
        txt_hidden_state=None,
        token_type_ids=None,
        attention_mask=None,
        vis_features=None,
        vis_attention_mask=None,
    ):
        if similarity_matrix_v2t is None and vis_embeds is not None:
            num_vis_embeds, num_txt_embeds = vis_embeds.shape[0], txt_embeds.shape[0]
            vis_embeds_normalized = F.normalize(vis_embeds, dim=-1)
            txt_embeds_normalized = F.normalize(txt_embeds, dim=-1)
            sims_matrix_v2t = vis_embeds_normalized @ txt_embeds_normalized.t()
        elif similarity_matrix_v2t is not None:
            num_vis_embeds, num_txt_embeds = similarity_matrix_v2t.shape[0], similarity_matrix_v2t.shape[1]
            sims_matrix_v2t = similarity_matrix_v2t
        else:
            raise ValueError("Type error of similarity matrix and embeddings when computing score matrix")

        ### vision to text score matrix ###
        score_matrix_v2t = torch.full((num_vis_embeds, num_txt_embeds), -100.0).to(sims_matrix_v2t.device)
        for i, sims in enumerate(sims_matrix_v2t):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

            if requires_match_score:
                match_score = self.compute_match_score(
                    txt_hidden_state=txt_hidden_state[topk_idx],
                    attention_mask=attention_mask[topk_idx],
                    token_type_ids=token_type_ids[topk_idx],
                    vis_features=vis_features[i].repeat(k_test, 1, 1),
                    vis_attention_mask=vis_attention_mask[i].repeat(k_test, 1),
                ).to(sims_matrix_v2t.device)
            else:
                match_score = 0.

            score_matrix_v2t[i, topk_idx] = topk_sim + match_score

        ### text to vision score matrix ###
        sims_matrix_t2v = sims_matrix_v2t.t()
        score_matrix_t2v = torch.full((num_txt_embeds, num_vis_embeds), -100.0).to(sims_matrix_v2t.device)
        for i, sims in enumerate(sims_matrix_t2v):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            
            if requires_match_score:
                match_score = self.compute_match_score(
                    txt_hidden_state=txt_hidden_state[i].repeat(k_test, 1, 1),
                    attention_mask=attention_mask[i].repeat(k_test, 1),
                    token_type_ids=token_type_ids[i].repeat(k_test, 1),
                    vis_features=vis_features[topk_idx],
                    vis_attention_mask=vis_attention_mask[topk_idx],
                ).to(sims_matrix_t2v.device)
            else:
                match_score = 0.

            score_matrix_t2v[i, topk_idx] = topk_sim + match_score

        return score_matrix_v2t.numpy(), score_matrix_t2v.numpy()

    @torch.no_grad
    def compute_match_score(
        self,
        txt_hidden_state,
        attention_mask,
        token_type_ids,
        vis_features,
        vis_attention_mask,
    ):
        mm_embeddings = self.text_ssl.extract_features(
            None,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=txt_hidden_state,
            encoder_hidden_states=vis_features,
            encoder_attention_mask=vis_attention_mask,
            mode="mm_match",
        )

        logits = self.mm_match_head(mm_embeddings)
        # scores = F.softmax(logits, dim=-1)
        positive_score = logits[:, 1]
        # print("positive score", positive_score, positive_score.shape)

        return positive_score

