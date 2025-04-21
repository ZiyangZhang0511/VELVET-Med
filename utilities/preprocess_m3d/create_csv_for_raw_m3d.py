import os
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
import multiprocessing
from functools import partial

import pandas as pd
import numpy as np
import cv2


def get_images_resolution(images_filepath):

    images_size = []
    for image_filepath in images_filepath:
        try:
            image_np = cv2.imread(image_filepath, cv2.IMREAD_UNCHANGED)
        except:
            print(f"Failed to read {image_filepath}")

        if image_np is not None:
            images_size.append(image_np.shape[:2])
        else:
            print(f"NoneType at {image_filepath}")

    if images_size:
        most_common_size, count = Counter(images_size).most_common(1)[0]
    else:
        most_common_size, count = (0, 0), 0

    return most_common_size, count


def process_case_dir(
    case_dir,
    images_dir_path_list,
    df,
    save_path,
):

    print(f"Processing {case_dir.stem}......")
    study_dirs = [d for d in case_dir.iterdir() if d.is_dir()]

    for study_dir in tqdm(study_dirs):
        # print(f"study_id {study_dir.stem}")
        image_dirs = [d for d in study_dir.iterdir() if d.is_dir()]

        for image_dir in image_dirs:
            if image_dir.resolve() in images_dir_path_list:
                continue

            # image_files = [f for f in image_dir.iterdir() if f.is_file()]
            image_files = [f for f in image_dir.rglob("*") if f.is_file()]

            # print(image_dir.resolve(), image_dir.name, len(image_files))
            common_image_size, count = get_images_resolution(image_files)
            # print(common_image_size, (count, *common_image_size))

            new_row = {
                "case_id": case_dir.name,
                "study_id": study_dir.name,
                "images_id": image_dir.name,
                "num_images": len(image_files),
                "study_dir_path": study_dir.resolve(),
                "images_dir_path": image_dir.resolve(),
                # "D, H, W": (count, *common_image_size),
                "Depth": count,
                "Height": common_image_size[0],
                "Width": common_image_size[1],
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(save_path, index=False)
            # images_dir_path_list.append(image_dir.resolve())

    return df


def process_study_dir(study_dir, case_dir, images_dir_path_list):
    image_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
    rows = []
    for image_dir in image_dirs:

        if str(image_dir.resolve()) in images_dir_path_list:
            print(0, end=" ")
            continue

        # image_files = [f for f in image_dir.iterdir() if f.is_file()]
        image_files = [f for f in image_dir.rglob("*") if f.is_file()]

        common_image_size, count = get_images_resolution(image_files)

        new_row = {
            "case_id": case_dir.name,
            "study_id": study_dir.name,
            "images_id": image_dir.name,
            "num_images": len(image_files),
            "study_dir_path": study_dir.resolve(),
            "images_dir_path": image_dir.resolve(),
            # "D, H, W": (count, *common_image_size),
            "Depth": count,
            "Height": common_image_size[0],
            "Width": common_image_size[1],
        }
        rows.append(new_row)
    # print(f"Getting {len(rows)} new entries.")
    return rows 



def main(base_dir):
    # Define the base directory
    base_dir = Path(base_dir)
    save_path = "./data/ct_case_info_1.csv"

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        # print(df.columns.to_list())
    else:
        columns = ['case_id', 'study_id', 'images_id', 'num_images', 'study_dir_path', 'images_dir_path','Depth', 'Height', 'Width']
        df = pd.DataFrame(columns=columns)

    study_dir_path_list = df["study_dir_path"].to_list()
    images_dir_path_list = df["images_dir_path"].to_list()

    case_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    # print(case_dirs, len(case_dirs))
    # for case_dir in case_dirs:
    #     df = process_case_dir(case_dir, images_dir_path_list, save_path, df)
    #     # df.to_csv(save_path, index=False)


    # process_case_dir_partial = partial(
    #     process_case_dir,
    #     images_dir_path_list=images_dir_path_list,
    #     # df=df,
    #     # save_path=save_path,
    # )
    # with multiprocessing.Pool(8) as pool:
    #     list(tqdm(pool.map(process_case_dir_partial, case_dirs), total=len(case_dirs)))
                
    for case_dir in case_dirs:
        print(f"Processing {case_dir.stem}......")
        study_dirs = [d for d in case_dir.iterdir() if d.is_dir()]
        process_study_dir_partial = partial(
            process_study_dir,
            case_dir=case_dir,
            images_dir_path_list=images_dir_path_list,
            # df=df,
            # save_path=save_path,
        )
        with multiprocessing.Pool(8) as pool:
            results = tqdm(pool.map(process_study_dir_partial, study_dirs), total=len(study_dirs))
            print(f"Completed {case_dir.name} with results {len(results)}")

        if results is not None:
            for result in results:
                if result:
                    df = pd.concat([df, pd.DataFrame(result)], ignore_index=True)
                    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    
    base_dir = "/projects/p32335/M3D_CAP/ct_case/"
    main(base_dir)

