from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    Dataset as mDataset,
    DataLoader as mDataLoader,
    load_decathlon_datalist
)


from utilities.constants import DataPath, SliceMap
from utilities.vision_transforms import MonaiTransforms
from utilities.seg_utils import load_seg_data_list, visualize_transformed_data


if __name__ == "__main__":

    train_list, val_list, test_list = load_seg_data_list("abdomenct1k")

    transforms = MonaiTransforms(num_samples=2)
    train_mt = transforms.load_supervised_seg_transforms("train")
    val_mt = transforms.load_supervised_seg_transforms("val")

    train_dataset = CacheDataset(
        data=train_list,
        transform=train_mt,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    # val_dataset = CacheDataset(
    #     data=val_list,
    #     transform=val_mt,
    #     cache_num=6,
    #     cache_rate=1.0,
    #     num_workers=4,
    # )
    print(train_dataset[0][0]["image"].meta)
    # print(train_dataset[0][0]["label"].size(), train_dataset[0][0]["label"].max())
    # visualize_transformed_data(val_dataset, 1, SliceMap.abdomenct1k["val"], save=True)