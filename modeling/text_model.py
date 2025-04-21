import os
import yaml
import json
import random
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utilities.constants import ConfigPath, DataPath

from datasets import collate_fn
from datasets.m3d_cap import M3DCAPDataset
from datasets.text_utils import SentBasedTokenizer

from .networks import SentBertModel, MLMHead


class TextSSL(nn.Module):

    def __init__(self, config, device="cuda"):
        super(TextSSL, self).__init__()

        self.device = device

        ### prepare configs ###
        config_tokenizer = config["tokenizer"]
        # print(config_tokenizer)
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


        ### build text model ###
        self.text_model = SentBertModel(config_txt_model)
        self.init_text_model(config_txt_model["pretrained_ckpt"])
        # print(self.text_model)

        ### config txt ssl head and criterion ###
        if config["enable_txt_mlm"]:
            self.text_mlm_head = MLMHead(config_txt_model)
            self.text_mlm_criterion = nn.CrossEntropyLoss()

        self.enable_txt_mlm = config["enable_txt_mlm"]

    @torch.no_grad
    def init_text_model(self, ckpt_path):

        if not os.path.isfile(ckpt_path):
            # print(self.text_model.encoder.layer[0].attention.self.query.weight.max())
            print("No pre-traned weights loaded for text model......")
            return

        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        ### initialize encoder layers ###
        encoder_stata_dict = {}
        for name, params in state_dict.items():
            if "encoder" in name:
                if name.startswith("bert."):
                    name = name.replace("bert.", "")
                encoder_stata_dict[name] = params
        # print(self.text_model.encoder.layer[0].attention.self.query.weight.max())
        self.text_model.load_state_dict(encoder_stata_dict, strict=False)
        # print(self.text_model.encoder.layer[0].attention.self.query.weight.max())

        ### initialize embedding layers ###
        for name, params in state_dict.items():
            if "word_embeddings" in name:
                if self.config_txt_model["sentence_modeling"]:
                    self.text_model.embeddings.word_embeddings.weight.data[:30522].copy_(params.data)
                    self.text_model.embeddings.word_embeddings.weight.data[30522:].copy_(params.data[2].repeat(self.config_txt_model["type_vocab_size"]-1, 1))
                else:
                    self.text_model.embeddings.word_embeddings.weight.data[:30522].copy_(params.data)
            
            elif "position_embeddings" in name:
                self.text_model.embeddings.position_embeddings.weight.data.copy_(params.data)
            
            elif "token_type_embeddings" in name:
                if self.config_txt_model["sentence_modeling"]:
                    self.text_model.embeddings.token_type_embeddings.weight.data[1:].copy_(params.data[0].repeat(self.config_txt_model["type_vocab_size"]-1, 1))
                    # print(params.data[0].repeat(self.config_txt_model["type_vocab_size"]-1 ,1).size())
                else:
                    self.text_model.embeddings.token_type_embeddings.weight.data[1].copy_(params.data[0])
                
            elif "embeddings.LayerNorm.weight" in name:
                self.text_model.embeddings.LayerNorm.weight.data.copy_(params.data.clone())
            
            elif "embeddings.LayerNorm.bias" in name:
                self.text_model.embeddings.LayerNorm.bias.data.copy_(params.data.clone())
        
        print("Use pre-trained weights for text model......")


    def aggregate_token_embeddings(self, embeddings, input_ids):
        """
        Function: aggregate token embeddings for subword ("##")
        Input:
            embeddings of shape [bs, num_layers, seq_len, hz]
            input_ids of shape [bs, seq_len]
        """

        batch_size, num_layers, seq_len, hz = embeddings.size()
        embeddings = embeddings.permute(0, 2, 1, 3).contiguous() # [bs, seq_len, num_layers, hz]
        aggregated_embeddings = [] # a list (len: bz) 
        processed_sequence = [] # a list (len: bz) of lists of real words

        ### loop over batch ###
        for seq_emb, input_id in zip(embeddings, input_ids):
            # seq_emb [seq_len, num_layers, hz]
            # input_id [seq_len]

            tokens = []
            agg_embs = []

            token_bank = []
            token_embs_bank = []

            ### loop over current input_id ###
            for token_emb, current_id in zip(seq_emb, input_id):
                # token_emb [num_layers, hz]
                # current_id 
                
                token = self.tokenizer.base_tokenizer.ids_to_tokens[current_id.item()]
                # print(token)

                if token == "[PAD]":
                    # current input_id finished
                    new_emb = torch.stack(token_embs_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    tokens.append("".join(token_bank))
                    break


                if not token.startswith("##"):

                    if len(token_bank) == 0:
                        token_embs_bank.append(token_emb)
                        token_bank.append(token)
                    else:
                        new_emb = torch.stack(token_embs_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        tokens.append("".join(token_bank))

                        token_embs_bank = [token_emb]
                        token_bank = [token]
                else:
                    if token.startswith("##"):
                        token_embs_bank.append(token_emb)
                        token_bank.append(token[2:])
            # print(len(agg_embs))
            # print(len(tokens))

            agg_embs = torch.stack(agg_embs)
            padding_size = seq_len - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, hz).to(agg_embs.device)
            tokens = tokens + ["[PAD]"] * padding_size

            aggregated_embeddings.append(torch.cat([agg_embs, paddings]))
            processed_sequence.append(tokens)

        aggregated_embeddings = torch.stack(aggregated_embeddings)
        aggregated_embeddings = aggregated_embeddings.permute(0, 2, 1, 3).contiguous()
        # print(aggregated_embeddings.size(), len(processed_sequence))
        return aggregated_embeddings, processed_sequence

    def extract_features(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        mode="txt_mlm", # "itc"
        last_n_layers_for_itc=3,
    ):
        
        if mode in ["txt_mlm", "itc_all", "itc"]:
            outputs = self.text_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                mode="txt_mlm",
            )

            if mode == "txt_mlm":
                # print((outputs.last_hidden_state).shape)
                return outputs.last_hidden_state

            elif mode == "itc":
                return outputs.last_hidden_state[:, 0, :], outputs.last_hidden_state

            elif mode == "itc_all":
                all_hidden_states = outputs.hidden_states
                # print(len(all_hidden_states))

                # embeddings of shape [bs, num_layers, seq_len, hz]
                embeddings = torch.stack(
                    all_hidden_states[self.config_txt_model["layer_for_txt_ssl"]-last_n_layers_for_itc+1 : self.config_txt_model["layer_for_txt_ssl"]+1]
                ).permute(1, 0, 2, 3).contiguous()
                # print(embeddings.size())

                # [bs, num_layers, seq_len, hz]
                aggregated_embeddings, processed_sequence = self.aggregate_token_embeddings(embeddings, input_ids)

                if self.config_txt_model["layer_aggregate_fn"] == "sum":
                    aggregated_embeddings = aggregated_embeddings.sum(dim=1)
                elif self.config_txt_model["layer_aggregate_fn"] == "mean":
                    # [bs, seq_len, hz]
                    aggregated_embeddings = aggregated_embeddings.mean(dim=1)

                # aggregated_embeddings of shape [bs, seq_len, hz]
                # processed_sequence a list of lists of aggregated tokens
                # print("itc_all", (outputs.last_hidden_state).size())
                return aggregated_embeddings, processed_sequence, outputs.last_hidden_state

           
        elif mode == "mm_mlm":
            outputs = self.text_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                mode="mm_mlm",
            )
            # print("mm_mlm", len(outputs.hidden_states), (outputs.last_hidden_state).size())
            return outputs.last_hidden_state
        
        elif mode == "mm_match":
            outputs = self.text_model(
                input_ids=None,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                mode="mm_match",
            )
            # print("mm_match", len(outputs.hidden_states), (outputs.last_hidden_state).size())
            return outputs.last_hidden_state[:, 0, :]


    def forward(self, batch_data):
        """
        batch_data["input_ids"] shape [bs, seq_len]
        batch_data["token_type_ids"]
        batch_data["attention_mask"]
        """
        
        input_ids = batch_data["input_ids"]
        token_type_ids = batch_data["token_type_ids"]
        attention_mask = batch_data["attention_mask"]
        self.device = input_ids.device
        loss_placeholder = torch.tensor(0, dtype=input_ids.dtype, device=input_ids.device, requires_grad=False)

        if self.enable_txt_mlm:
            ### mask data for mlm ssl ###
            masked_input_ids, masked_target_ids = self.mask_input_ids(input_ids.clone(), self.config_txt_model["mlm_probability"])

            ### extract txt features ###
            features = self.extract_features(
                input_ids=masked_input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                mode="txt_mlm",
            )

            ### calculate loss ###
            mlm_logits = self.text_mlm_head(features)
            mlm_logits = mlm_logits.view(-1, self.config_txt_model["vocab_size"]).contiguous()
            masked_target_ids = masked_target_ids.view(-1).contiguous()
            loss_txt_mlm = self.text_mlm_criterion(mlm_logits, masked_target_ids)
        else:
            loss_txt_mlm = loss_placeholder

        return {
            "loss": loss_txt_mlm,
            "loss_txt_mlm": loss_txt_mlm,
        }


    @torch.no_grad()
    def test_one_step(self, batch_data):
        input_ids = batch_data["input_ids"].to(self.device)
        token_type_ids = batch_data["token_type_ids"].to(self.device)
        attention_mask = batch_data["attention_mask"].to(self.device)
        loss_placeholder = torch.tensor(0, dtype=input_ids.dtype, device=input_ids.device, requires_grad=False)

        if self.enable_txt_mlm:
            ### mask data for mlm ssl ###
            masked_input_ids, masked_target_ids = self.mask_input_ids(input_ids.clone(), self.config_txt_model["mlm_probability"])

            ### extract txt features ###
            features = self.extract_features(
                input_ids=masked_input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                mode="txt_mlm",
            )

            ### calculate loss ###
            mlm_logits = self.text_mlm_head(features)
            mlm_logits = mlm_logits.view(-1, self.config_txt_model["vocab_size"]).contiguous()
            masked_target_ids = masked_target_ids.view(-1).contiguous()
            loss_txt_mlm = self.text_mlm_criterion(mlm_logits, masked_target_ids)

        else:
            loss_txt_mlm = loss_placeholder
    
        return {
            "loss": loss_txt_mlm,
            "loss_txt_mlm": loss_txt_mlm,
        }, {}

    @torch.no_grad()
    def test_on_dataloader(self, dataloader, training_step, recon_save_dir=None):

        epoch_loss_dict = {}

        for i, batch_data in enumerate(tqdm(dataloader)):

            loss_dict, _ = self.test_one_step(batch_data)

            # process loss items
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_loss_dict.keys():
                    epoch_loss_dict[loss_name] = 0
                epoch_loss_dict[loss_name] += loss_value.item() / len(dataloader)

        ### structure final dict to return ###
        final_dict = epoch_loss_dict.copy()

        return final_dict


    def mask_input_ids(self, input_ids, mlm_probability=0.15):

        device = input_ids.device
        target_ids = input_ids.clone()

        probability_matrix = torch.full(input_ids.shape, mlm_probability)                        
        masked_indices = torch.bernoulli(probability_matrix).bool() #(bs, seq_len)

        # keep pad_token, cls_token and sent_token unmasked
        # print(self.tokenizer.base_tokenizer.additional_special_tokens_ids)
        # print(self.tokenizer.base_tokenizer.mask_token_id, self.tokenizer.base_tokenizer.pad_token_id, self.tokenizer.base_tokenizer.cls_token_id)
        for sent_token_id in self.tokenizer.base_tokenizer.additional_special_tokens_ids:
            masked_indices[input_ids == sent_token_id] = False    
        masked_indices[input_ids == self.tokenizer.base_tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.base_tokenizer.cls_token_id] = False
        
        # only compute loss on masked tokens
        target_ids[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.base_tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.base_tokenizer.vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        return input_ids, target_ids


if __name__ == "__main__":

    device = "cuda"

    data_root = os.path.abspath(DataPath.M3D_CAP)
    data_dir = os.path.join(data_root, "nii_down")
    json_filepath = os.path.join(data_root, "m3d_cap_split_thr48.json")
    transforms = None
    dataset = M3DCAPDataset(data_dir, json_filepath, transforms, data_ratio=0.1, mode="val", pretrained_type="txt_ssl")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False, collate_fn=collate_fn)
    # first_batch = next(iter(dataloader))
    # for key in first_batch.keys():
    #     first_batch[key] = first_batch[key].to(device)
    # print(first_batch["input_ids"][0])
    # print(first_batch["token_type_ids"][0])
    # print(first_batch["attention_mask"][0])

    txt_config_filepath = os.path.abspath(ConfigPath.TEXT_SSL)
    with open(txt_config_filepath, "r") as f:
        txt_config = yaml.safe_load(f)
    model = TextSSL(txt_config).to(device)
    print(model)

    # model(first_batch)

    # model.extract_features(
    #     first_batch["input_ids"],
    #     first_batch["token_type_ids"],
    #     first_batch["attention_mask"],
    #     mode="itc_all",
    # )

    re = model.test_on_dataloader(dataloader, 0)
    print(re)


