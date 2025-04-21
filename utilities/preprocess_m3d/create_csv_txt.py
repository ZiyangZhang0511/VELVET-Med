import os
import re
import json
import nltk
import argparse
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize


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



def word_count_stats(list_of_strings):

    # 1. Split each string by whitespace and count the words
    counts = [len(s.split()) for s in list_of_strings]

    # 2. Compute max, min, and mean of the counts
    max_count = max(counts)
    min_count = min(counts)
    mean_count = sum(counts) / len(counts) if len(counts) > 0 else 0
    total_count = sum(counts)

    median_count = sorted(counts)[len(counts)//2]

    return (max_count, min_count, mean_count, median_count, total_count)


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", default="/home/olg7848/p32335/my_research/vlp_ct/data/m3d_cap/m3d_cap_split_thr48.json", type=str)
    parser.add_argument("--output_filepath", type=str, required=True)
    parser.add_argument("--data_dir", default="/home/olg7848/p32335/my_research/vlp_ct/data/m3d_cap/nii_down", type=str)


    args = parser.parse_args()

    return args


def main():

    args = get_args()

    with open(args.json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    txt_filename_list = []
    for key, value in json_data.items():
        for item in value:
            txt_filename_list.append(item["text"])
    # print(txt_filename_list[:2], len(txt_filename_list))



    if os.path.exists(args.output_filepath):
        df = pd.read_csv(args.output_filepath)
        # print(df.columns.to_list())
    else:
        columns = ['txt_filename', 'num_sents', 'max_len_sent', 'min_len_sent', 'mean_len_sent','median_len_sent', 'total_word_count']
        df = pd.DataFrame(columns=columns) 

    for i, txt_filename in enumerate(tqdm(txt_filename_list)):

        txt_path = os.path.join(args.data_dir, txt_filename)

        with open(txt_path, "r") as f:
            text = f.read()

        splitted_list = custom_split(text)

        word_stats = word_count_stats(splitted_list)

        new_row = {
            'txt_filename':txt_filename, 
            'num_sents': len(splitted_list),
            'max_len_sent': word_stats[0], 
            'min_len_sent': word_stats[1], 
            'mean_len_sent': word_stats[2],
            'median_len_sent': word_stats[3],
            'total_word_count':word_stats[4], 
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # if i % 1000 == 0:
        df.to_csv(args.output_filepath, index=False)



if __name__ == "__main__":

    main()

