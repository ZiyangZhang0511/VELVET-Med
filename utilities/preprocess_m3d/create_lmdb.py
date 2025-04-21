import os
import lmdb
import argparse
import pickle, io
from pathlib import Path
from tqdm.auto import tqdm

import cv2
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt



def process_images(images_dir, spatial_size=None):

    ###======== read all individual slices ========###
    filepath_list = [filepath for filepath in images_dir.rglob("*") if filepath.is_file()]
    sorted_filepath_list = sorted(filepath_list, key=lambda x: int(x.stem))
    # print(sorted_filepath_list)

    volume_np = []
    
    for slice_filepath in sorted_filepath_list:

        try:
            single_slice_np = cv2.imread(slice_filepath, cv2.IMREAD_UNCHANGED)

            if single_slice_np is None:
                continue

            height, width = single_slice_np.shape[:2]

            if height != spatial_size[1] or width != spatial_size[2]:
                continue

            if len(single_slice_np.shape) == 2:
                single_slice_np = single_slice_np[..., np.newaxis]

            elif len(single_slice_np.shape) == 3:

                if single_slice_np.shape[-1] == 3:
                    single_slice_np = cv2.cvtColor(single_slice_np, cv2.COLOR_BGR2GRAY)
                    single_slice_np = single_slice_np[..., np.newaxis]

                elif single_slice_np.shape[-1] == 4:
                    single_slice_np = cv2.cvtColor(single_slice_np, cv2.COLOR_BGRA2BGR)
                    single_slice_np = cv2.cvtColor(single_slice_np, cv2.COLOR_BGR2GRAY)
                    single_slice_np = single_slice_np[..., np.newaxis]

                elif single_slice_np.shape[-1] == 1:
                    pass

            if single_slice_np.shape != (spatial_size[1], spatial_size[2], 1):
                print(f"Warning: {slice_filepath} final shape={single_slice_np.shape}, "
                    f"expected ({spatial_size[1]}, {spatial_size[2]}, 1). Skipping.")
                continue

            volume_np.append(single_slice_np)

        except:
            print(f"Failed to read {slice_filepath}")

    if volume_np:
        volume_np = np.stack(volume_np) # volume_np [D, H, W, C]
        return volume_np.transpose(3, 1, 2, 0)
    else:
        # volume_np = np.random.randint(0, 256, (128, 512, 512, 1))
        print("Empty volume_np from images_dir:", images_dir)
        return None

    # volume_np = np.stack(volume_np) # volume_np [D, H, W, C]
    # return volume_np.transpose(3, 1, 2, 0)

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--info_data_path", 
        default="/home/olg7848/p32335/my_research/vlp_ct/data/m3d_cap/ct_case_info_1.csv",
        type=str,
    )
    parser.add_argument("--target_case_id", required=True, type=str)
    parser.add_argument("--lmdb_dir", required=True, type=str)

    args = parser.parse_args()

    return args



def main():

    args = get_args()

    ### read info data
    info_data = pd.read_csv(args.info_data_path)
    info_data = info_data[info_data["case_id"]==args.target_case_id]

    # study_dir_list = info_data["study_dir_path"].to_list()
    images_dir_list = info_data["images_dir_path"].to_list()
    depth_list = info_data["Depth"].to_list()
    height_list = info_data["Height"].to_list()
    width_list = info_data["Width"].to_list()

    ### open lmdb environment
    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(args.lmdb_dir, map_size=map_size)

    ### iterate all images_dir
    # images_dir_list = images_dir_list[:5]
    with env.begin(write=True) as txn:
        for idx, images_dir in enumerate(tqdm(images_dir_list)):

            spatial_size = (depth_list[idx], height_list[idx], width_list[idx])
            images_dir = Path(images_dir)

            ### perpare key name
            case_id = images_dir.parent.parent.stem
            study_id = images_dir.parent.stem
            imaging_id = images_dir.stem
            key_stem = f"{case_id}-{study_id}-{imaging_id}"

            ### get volumn data

            volume_np = process_images(images_dir, spatial_size=spatial_size)
            if not volume_np:
                continue
            # print(volume_np.shape, volume_np.dtype, volume_np.mean())

            # meta = f"{volume_np.shape}|{volume_np.dtype.str}"
            # plt.imshow(volume_np[0], cmap="gray")
            # volume_bytes = pickle.dumps(volume_np)

            buffer = io.BytesIO()
            np.savez_compressed(buffer, arr=volume_np)
            volume_bytes = buffer.getvalue()

            # raw_bytes = volume_np.tobytes()
            # volume_bytes = lz4.frame.compress(raw_bytes, compression_level=16)
            # key_meta = key_stem + "-meta"
            # txn.put(key_meta.encode(), meta.encode())

            ### get text data

            
            ### put in lmdb
            key_volume = key_stem + "-volume"
            txn.put(key_volume.encode(), volume_bytes)
        

    env.close()

if __name__ in "__main__":

    main()