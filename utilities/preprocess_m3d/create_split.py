import json
import random
import argparse
from pathlib import Path

import pandas as pd


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_info_path", default="/home/olg7848/p32335/my_research/vlp_ct/data/m3d_cap/ct_case_info.csv")
    parser.add_argument("--output_filepath", required=True, type=str)
    parser.add_argument("--depth_threshold", type=int, default=48)
    parser.add_argument("--data_dir", default="/home/olg7848/p32335/my_research/vlp_ct/data/m3d_cap/nii_down", type=str)

    args = parser.parse_args()

    return args


def main():

    args = get_args()

    data_info = pd.read_csv(args.data_info_path)
    data_info = data_info[data_info["Depth"] >= args.depth_threshold]
    print(len(data_info))

    csv_nii_filename_list = data_info["nii_path"].to_list()
    csv_txt_filename_list = data_info["txt_path"].to_list()
    # print(nii_filename_list[104], txt_filename_list[104])
    # return

    
    data_dir = Path(args.data_dir)
    nii_filepath_list = [filepath for filepath in data_dir.rglob("*.nii.gz") if filepath.is_file()]
    txt_filepath_list = [filepath for filepath in data_dir.rglob("*.txt") if filepath.is_file()]
    nii_filepath_list = sorted(nii_filepath_list)
    txt_filepath_list = sorted(txt_filepath_list)
    # # print(nii_filepath_list[0].name, txt_filepath_list[0].name)
    # # print(len(nii_filepath_list), len(txt_filepath_list))

    nii_filename_list = [filepath.name for filepath in nii_filepath_list]
    txt_filename_list = [filepath.name for filepath in txt_filepath_list]
    # # print(nii_filename_list[1004], txt_filename_list[1004])
    # # print(len(nii_filename_list), len(txt_filename_list))
    data_pair_filename_list = list(zip(nii_filename_list, txt_filename_list))
    # print(pair_filename_list[104], len(pair_filename_list))


    pair_filename_list = []
    for csv_nii_filename, csv_txt_filename in list(zip(csv_nii_filename_list, csv_txt_filename_list)):
        if (csv_nii_filename, csv_txt_filename) in data_pair_filename_list:
            pair_filename_list.append((csv_nii_filename, csv_txt_filename))
    print(len(pair_filename_list))


    random.shuffle(pair_filename_list)
    # print(pair_filename_list[1004], len(pair_filename_list))
    split_idx = int(0.95*(len(pair_filename_list)))
    train_datalist = pair_filename_list[:split_idx]
    val_datalist = pair_filename_list[split_idx:]
    print(len(train_datalist), len(val_datalist))

    json_dict = {
        "train":[],
        "val":[],
    }

    for nii_filename, txt_filename in train_datalist:
        dict_item = {
            "image": nii_filename,
            "text": txt_filename,
        }
        json_dict["train"].append(dict_item)


    for nii_filename, txt_filename in val_datalist:
        dict_item = {
            "image": nii_filename,
            "text": txt_filename,
        }
        json_dict["val"].append(dict_item)

    with open(args.output_filepath, "w") as f:
        json.dump(json_dict, f, indent=4)





if __name__ == "__main__":
    
    random.seed(1)
    main()