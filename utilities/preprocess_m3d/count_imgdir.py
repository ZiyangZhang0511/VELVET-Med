import os
from pathlib import Path


def main(base_dir):
    base_dir = Path(base_dir)

    total_img_dir = 0

    case_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    for case_dir in case_dirs:
        case_img_dir = 0
        study_dirs = [d for d in case_dir.iterdir() if d.is_dir()]
        for study_dir in study_dirs:
            image_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
            # print(f"{case_dir.name}, {study_dir.name}: num_images_dir {len(image_dirs)}")
            total_img_dir += len(image_dirs)
            case_img_dir += len(image_dirs)

        print(f"{case_dir.name}: num_images_dir {case_img_dir}")
    print("total img dirs:", total_img_dir)

if __name__ == "__main__":
    base_dir = "/home/olg7848/p32335/M3D_CAP/ct_case/"
    main(base_dir)
