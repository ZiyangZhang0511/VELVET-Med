import os
import re
import yaml
import json
import random
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, AutoTokenizer

from utilities.constants import ConfigPath


def remove_m_elements(old_list, m):
    """
    Randomly removes m elements from a list of length n,
    preserving the order of the remaining elements.
    """
    n = len(old_list)
    removal_count = m  # how many elements to remove
    all_indices = list(range(n))
    
    # randomly pick which indices to remove
    indices_to_remove = set(random.sample(all_indices, k=removal_count))
    
    # build the new list, skipping removed indices
    new_list = [old_list[i] for i in range(n) if i not in indices_to_remove]
    return new_list


def custom_split(text):

    unwanted_words_list = [
        "Title:",
        "presentation:",
        "patient:",
        "Age:",
        "Gender:",
        "study_findings:",
        "discussion:",
        # "\n",
    ]

    blocks = re.split(r"\n{3,}", text) # block that are split by two or more consecutive newlines

    results = [] # all sentences
    for block in blocks:
        pieces = re.split(r"([.!?])", block)

        combined = [] # all sentences in a block
        for i in range(0, len(pieces), 2):
            # `pieces[i]` is text (might be empty/whitespace if there's punctuation in a row)
            chunk = pieces[i].strip() # single sentence
            
            # If there's a next element (the punctuation), attach it
            if i + 1 < len(pieces):
                chunk += pieces[i + 1]
            chunk = chunk.strip()

            for word in unwanted_words_list:
                    chunk = chunk.replace(word, "")
            chunk = chunk.strip()
            

            if chunk: # process one sentence, chunk is a string

                word_counts = len(re.split(r"[ \n]+", chunk))

                if word_counts > 30 and "\n" in chunk:
                    subchunks = re.split(r"\n", chunk)
                    for subchunk in subchunks:
                        if subchunk:
                            subchunk = subchunk.replace("\xa0", " ")
                            combined.append(subchunk)
                else:
                    chunk = chunk.replace("\xa0", " ")
                    chunk = chunk.replace("\n", " ")
                    combined.append(chunk)


        if combined:
            results.extend(combined)

    return results



class SentBasedTokenizer():

    def __init__(self, config):

        ### bulid base tokenizer and some initial attributes ###
        # config = config["tokenizer"]
        self.base_tokenizer = BertTokenizer.from_pretrained(config["base_type"])
        # print(dir(self.base_tokenizer))
        self.sentence_modeling = config["sentence_modeling"]
        self.max_len_report = config["max_len_report"]
        self.max_len_sentence = config["max_len_sentence"]
        self.max_num_sentences = config["max_num_sentences"]

        if self.sentence_modeling:
            print("enable sentence modeling in tokenizer...")
            ### add [SEN] tokens ###
            special_tokens = [f"[SEN_{idx+1}]" for idx in range(config["max_num_sentences"])]
            new_tokens_dict = {"additional_special_tokens": special_tokens}
            num_added_tokens = self.base_tokenizer.add_special_tokens(new_tokens_dict)
            # 'add_special_tokens', 'add_tokens', 'added_tokens_decoder', 'added_tokens_encoder', 'additional_special_tokens', 'additional_special_tokens_ids', 'all_special_ids', 'all_special_tokens', 'all_special_tokens_extended'
            self.update_vocab()
        

    def update_vocab(self):
        """
        include additional special tokens ("[SEN]") into vocab and ids_to_tokens
        """
        for new_token, new_id in zip(self.base_tokenizer.additional_special_tokens, self.base_tokenizer.additional_special_tokens_ids):
            self.base_tokenizer.vocab[new_token] = new_id
            self.base_tokenizer.ids_to_tokens[new_id] = new_token
    
    def encode_single_sentence(self, sentence:str, sent_idx=1):
        """
        sentence: a string of sentence
        return: a dict of "input_ids", "token_type_ids", "attention_mask" all with Tensor shape [len_sentence,]
        """
        sent_token = f"[SEN_{sent_idx}]"

        # print(sentence)
        encoded_dict = self.base_tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="do_not_pad",
            max_length=self.max_len_sentence,
        )
        # for key, value in encoded_dict.items():
        #     print(key, value)

        ### remove [SEP] ###
        input_ids = encoded_dict["input_ids"][..., :-1]
        token_type_ids = encoded_dict["token_type_ids"][..., :-1]
        attention_mask = encoded_dict["attention_mask"][..., :-1]

        ### replace [CLS] with [SEN_{idx}] ###
        input_ids[..., 0] = self.base_tokenizer.vocab[sent_token]
        token_type_ids = torch.ones_like(token_type_ids) * sent_idx
        attention_mask = torch.ones_like(token_type_ids)
        # print(input_ids, token_type_ids, attention_mask)

        return input_ids, token_type_ids, attention_mask

    def encode_report(self, sent_list:list):

        if not self.sentence_modeling:
            whole_report = " ".join(sent_list)

            encoded_dict = self.base_tokenizer(
                whole_report,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_len_report,
            )

            # for key in encoded_dict.keys():
            #     new = encoded_dict[key][..., 1:-1].clone()
            #     encoded_dict[key][..., 2:] = new
            
            input_ids = encoded_dict["input_ids"]
            token_type_ids = encoded_dict["token_type_ids"]
            attention_mask = encoded_dict["attention_mask"]

            # sent_token = f"[SEN_1]"
            # input_ids[..., 1] = self.base_tokenizer.vocab[sent_token]
            # token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[input_ids != 0] = 1
            # attention_mask = torch.zeros_like(input_ids)
            # attention_mask[input_ids != 0] = 1
            # print(input_ids)

            return input_ids, token_type_ids, attention_mask


        ### report length
        num_sent = len(sent_list)

        if num_sent > self.max_num_sentences:
            old_sent_indices = list(range(num_sent))
            sent_indices = remove_m_elements(old_sent_indices, num_sent-self.max_num_sentences)
        else:
            sent_indices = list(range(num_sent))
        # print(sent_indices)

        ### 
        all_input_ids = []
        all_token_type_ids = []
        all_attention_mask = []

        for i, sent_idx in enumerate(sent_indices):
            input_ids, token_type_ids, attention_mask = self.encode_single_sentence(sent_list[sent_idx], i+1)
            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)
            all_attention_mask.append(attention_mask)
        all_input_ids = torch.cat(all_input_ids, dim=-1)
        all_token_type_ids = torch.cat(all_token_type_ids, dim=-1)
        all_attention_mask = torch.cat(all_attention_mask, dim=-1)

        cls_token_id = self.base_tokenizer.vocab["[CLS]"]
        cls_token_id = torch.tensor([[cls_token_id]], device=all_input_ids.device, dtype=all_input_ids.dtype)
        all_input_ids = torch.cat([cls_token_id, all_input_ids], dim=-1)
        cls_token_type_ids = torch.ones_like(cls_token_id)
        cls_attention_mask = torch.ones_like(cls_token_id)
        all_token_type_ids = torch.cat([cls_token_type_ids, all_token_type_ids], dim=-1)
        all_attention_mask = torch.cat([cls_attention_mask, all_attention_mask], dim=-1)

        # print(all_input_ids.shape, all_input_ids[..., :20])
        # print(all_token_type_ids.shape, all_token_type_ids[..., :20])
        # print(all_attention_mask.shape, all_attention_mask[..., :20])

        if all_input_ids.shape[-1] < self.max_len_report:

            pad_len = self.max_len_report - all_input_ids.shape[-1]
            all_input_ids = F.pad(all_input_ids, (0, pad_len), mode='constant', value=0)
            all_token_type_ids = F.pad(all_token_type_ids, (0, pad_len), mode='constant', value=0)
            all_attention_mask = F.pad(all_attention_mask, (0, pad_len), mode='constant', value=0)

            # if torch.unique(all_token_type_ids).numel() == self.max_num_sentences:
            #     # print("need to be padded")
            #     pad_len = self.max_len_report - all_input_ids.shape[-1]
            #     all_input_ids = F.pad(all_input_ids, (0, pad_len), mode='constant', value=0)
            #     all_token_type_ids = F.pad(all_token_type_ids, (0, pad_len), mode='constant', value=0)
            #     all_attention_mask = F.pad(all_attention_mask, (0, pad_len), mode='constant', value=0)

            # elif torch.unique(all_token_type_ids).numel() < self.max_num_sentences:
            #     # print("need to be sampled")
            #     # while num_words < max_len_report and num_sent < max_num_sent:
            #         # sample new sentence and add into input_ids
            #     while all_input_ids.shape[-1] < self.max_len_report and torch.unique(all_token_type_ids).numel() < self.max_num_sentences:
            #         cur_sent_token_idx = torch.unique(all_token_type_ids).numel() + 1
            #         cur_sent = random.sample(sent_list, k=1)
            #         input_ids, token_type_ids, attention_mask = self.encode_single_sentence(cur_sent, cur_sent_token_idx)
                    
            #         all_input_ids = torch.cat([all_input_ids, input_ids], dim=-1)
            #         all_token_type_ids = torch.cat([all_token_type_ids, token_type_ids], dim=-1)
            #         all_attention_mask = torch.cat([all_attention_mask, attention_mask], dim=-1)
                
            #     if all_input_ids.shape[-1] > self.max_len_report:
            #         all_input_ids = all_input_ids[..., :self.max_len_report]
            #         all_token_type_ids = all_token_type_ids[..., :self.max_len_report]
            #         all_attention_mask = all_attention_mask[..., :self.max_len_report]
                
            #     else:
            #         pad_len = self.max_len_report - all_input_ids.shape[-1]
            #         all_input_ids = F.pad(all_input_ids, (0, pad_len), mode='constant', value=0)
            #         all_token_type_ids = F.pad(all_token_type_ids, (0, pad_len), mode='constant', value=0)
            #         all_attention_mask = F.pad(all_attention_mask, (0, pad_len), mode='constant', value=0)

            # else:
            #     print("other situation, words_count:", all_input_ids.shape[-1])
            #     print("sent_count:", torch.unique(all_token_type_ids).numel())

        elif all_input_ids.shape[-1] > self.max_len_report:
            # print("need to be truncated")
            all_input_ids = all_input_ids[..., :self.max_len_report]
            all_token_type_ids = all_token_type_ids[..., :self.max_len_report]
            all_attention_mask = all_attention_mask[..., :self.max_len_report]

        
        # print(all_input_ids.shape, all_input_ids[..., :20])
        # print(all_token_type_ids.shape, all_token_type_ids[..., :20])
        # print(all_attention_mask.shape, all_attention_mask[..., :20])

        return all_input_ids, all_token_type_ids, all_attention_mask





if __name__ == "__main__":

    config_path = os.path.abspath(ConfigPath.TEXT_SSL)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tokenizer = SentBasedTokenizer(config["tokenizer"])


    ### prepare a text report to be split into multiple sentences ###
    # text_path = "./data/m3d_cap/nii_down/ct_case_12-004876-Axial_non_contrast.txt"
    # text_path = os.path.abspath(text_path)
    # # print(text_path)
    # with open(text_path, "r") as f:
    #     text = f.read()
    # splitted_list = custom_split(text)
    # print(text)
    # print(splitted_list)
    # tokenizer.encode_report(splitted_list)


    json_path = "./data/m3d_cap/m3d_cap_split_thr48.json"
    data_dir = "./data/m3d_cap/nii_down"
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    txt_filename_list = []
    for key, value in json_data.items():
        for item in value:
            txt_filename_list.append(item["text"])

    for txt_filename in tqdm(txt_filename_list):
        txt_filepath = os.path.join(data_dir, txt_filename)

        with open(txt_filepath, "r") as f:
           text = f.read()
        splitted_list = custom_split(text)

        tokenizer.encode_report(splitted_list)