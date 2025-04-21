import os
import shutil
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt



def downsample_array(volume_np):

    volume_tensor = torch.from_numpy(volume_np) # (D, H, W, C)
    volume_tensor = volume_tensor.permute(0, 3, 1, 2).contiguous()
    # print(volume_tensor.shape, volume_tensor.dtype, (volume_tensor*1.0).mean())

    # save_image(volume_tensor/255., "./check_data/volume_nii_raw.jpg")
    volume_tensor_downsampled = F.interpolate(
        volume_tensor,
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    ) # (D, C, h, w)
    # print(volume_tensor_downsampled.shape, volume_tensor_downsampled.dtype, (volume_tensor_downsampled*1.0).mean())
    # save_image(volume_tensor_downsampled/255., "./check_data/volume_nii_down.jpg")

    volume_tensor_downsampled = volume_tensor_downsampled.permute(0, 2, 3, 1).contiguous()

    volume_np_downsampled = volume_tensor_downsampled.numpy()
    # print(volume_np_downsampled.shape, volume_np_downsampled.dtype, (volume_np_downsampled*1.0).mean())

    return volume_np_downsampled


def generate_nii(images_dir, spatial_size, save_dir):

    images_dir = Path(images_dir)
    case_id = images_dir.parent.parent.stem
    study_id = images_dir.parent.stem
    imaging_id = images_dir.stem
    key_stem = f"{case_id}-{study_id}-{imaging_id}"

    nii_filename = f"{key_stem}.nii.gz"# + f"{images_dir_name}.nii.gz"
    nii_filepath = Path(save_dir)/nii_filename

    filepath_list = [filepath for filepath in images_dir.rglob("*") if filepath.is_file()]
    sorted_filepath_list = sorted(filepath_list, key=lambda x: int(x.stem))

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
        volume_np = downsample_array(volume_np) # volume_np [D, H, W, C]

        affine = np.eye(4, dtype=np.float16)
        if volume_np.shape[0] >= 96:
            affine[2, 2] = 0.5
        
        # print(volume_np.transpose(3, 1, 2, 0).shape, volume_np.dtype, affine.dtype)
        nii_img = nib.Nifti1Image(volume_np.transpose(3, 1, 2, 0), affine)
        nib.save(nii_img, nii_filepath)

        txt_filename = f"{key_stem}.txt"
        txt_filepath = Path(save_dir)/txt_filename
        source_txt_filepath = images_dir.parent/"text.txt"
        shutil.copyfile(source_txt_filepath, txt_filepath)

        return
    else:
        # volume_np = np.random.randint(0, 256, (128, 512, 512, 1))
        print("Empty volume_np from images_dir:", str(images_dir))
        return


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--info_data_path", 
        default="./data/m3d_cap/ct_case_info_1.csv",
        type=str,
    )
    parser.add_argument("--target_case_id", default=None, type=str)
    parser.add_argument("--nii_dir", required=True, type=str)

    args = parser.parse_args()

    return args

def main():

    args = get_args()

    ### read info data ###
    info_data = pd.read_csv(args.info_data_path)
    case_id_list = sorted(list(set(info_data["case_id"].to_list())))
    # print(case_id_list)

    if args.target_case_id:
        info_data = info_data[info_data["case_id"]==args.target_case_id]
    # print(len(info_data))

    images_dir_list = info_data["images_dir_path"].to_list()
    depth_list = info_data["Depth"].to_list()
    height_list = info_data["Height"].to_list()
    width_list = info_data["Width"].to_list()

    images_dir_list = images_dir_list[:1]
    ### generate each nii.gz for each images_dir ###
    for idx, images_dir in enumerate(tqdm(images_dir_list)):

        spatial_size = (depth_list[idx], height_list[idx], width_list[idx])
        images_dir = Path(images_dir)

        generate_nii(images_dir, spatial_size, args.nii_dir)





if __name__ == "__main__":
    main()