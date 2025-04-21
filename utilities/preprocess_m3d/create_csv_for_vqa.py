import os
import json
import argparse
import pandas as pd
from tqdm.auto import tqdm



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_csv_path", default="./data/m3d_vqa/raw_csv/M3D_VQA_val.csv", type=str)
    parser.add_argument("--raw_json_path", default="./data/m3d_cap/m3d_cap_split_thr48.json", type=str)
    parser.add_argument("--output_csv_path", type=str, required=True)
    # parser.add_argument("--data_dir", default="./data/m3d_cap/nii_down", type=str)

    args = parser.parse_args()

    return args


def main():

    args = get_args()
    mode = "train"

    ### acqurie formatted volume filenames from raw_json_path
    with open(args.raw_json_path, "r") as f:
        raw_json_data = json.load(f)
    raw_data_list = raw_json_data[mode]
    
    data_dict = {}
    for entry in raw_data_list:
        raw_image_path = entry["image"]
        parts = raw_image_path.split('-')
        study_id = parts[1]
        images_id = parts[2].replace(".nii.gz", "")
        # print(study_id, images_id)
        data_dict[f"{study_id}-{images_id}"] = raw_image_path
    # print(len(data_dict))
    # print(data_dict)
    data_keys_list = list(data_dict.keys())



    ### pick related entry from raw_csv_path
    raw_vqa_df = pd.read_csv(args.raw_csv_path)
    # npy_image_path_list =raw_vqa_df["Image Path"].to_list()
    # npy_image_path_case_list = [] 
    # for npy_image_path in npy_image_path_list:
    #     if "ct_case" in npy_image_path:
    #         npy_image_path_case_list.append(npy_image_path)
    # print(len(npy_image_path_case_list), npy_image_path_case_list[0])

    new_rows = []
    for idx, row in tqdm(raw_vqa_df.iterrows()):
        npy_image_path = row["Image Path"]
        if "ct_case" not in npy_image_path:
            continue

        parts = npy_image_path.split("/")
        study_id = parts[2]
        images_id = parts[3].replace(".npy", "")
        temp_key = f"{study_id}-{images_id}"

        if temp_key in data_keys_list:
            image_path = data_dict[temp_key]
            # print(image_path)

            new_row = {
                "image_path": image_path,
                "text_path": image_path.replace("nii.gz", "txt"),
                "question_type": row["Question Type"],
                "question": row["Question"],
                "choice_A": row["Choice A"],
                "choice_B": row["Choice B"],
                "choice_C": row["Choice C"],
                "choice_D": row["Choice D"],
                "answer": row["Answer"],
                "answer_choice": row["Answer Choice"],
            }
            # print(new_row)

            new_rows.append(new_row)

    
    df = pd.DataFrame(new_rows)
    print(len(df))
    print(len(list(set(df["image_path"].to_list()))))
    df.to_csv(args.output_csv_path, index=False)


if __name__ == "__main__":

    main()