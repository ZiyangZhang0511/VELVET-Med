import os
from tqdm.auto import tqdm

import monai
from monai.data import load_decathlon_datalist
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

import torch

import matplotlib.pyplot as plt

from utilities.constants import DataPath

def load_seg_data_list(dataset_name, data_ratio=1.0):

    if dataset_name == "btcv":
        data_root = DataPath.BTCV
        json_path = os.path.abspath(data_root + "/data_split.json")
        train_list = load_decathlon_datalist(json_path, True, "training")
        # print(len(train_list), train_list[0])
        val_list = load_decathlon_datalist(json_path, True, "validation")
        # print(len(val_list), val_list[0])
        test_list = load_decathlon_datalist(json_path, True, "test")
        # print(len(test_list), test_list[0])
    
    elif dataset_name == "abdomenct1k":
        data_root = DataPath.ABDOMENCT1K
        json_path = os.path.abspath(data_root + "/data_split.json")

        train_list = load_decathlon_datalist(json_path, True, "training")
        # print(len(train_list), train_list[0])
        val_list = load_decathlon_datalist(json_path, True, "validation")
        # print(len(val_list), val_list[0])
        # test_list = load_decathlon_datalist(json_path, True, "test")
        test_list = None

    elif dataset_name == "ctorg":
        data_root = DataPath.CTORG
        json_path = os.path.abspath(data_root + "/data_split.json")

        train_list = load_decathlon_datalist(json_path, True, "training")
        # print(len(train_list), train_list[0])
        val_list = load_decathlon_datalist(json_path, True, "validation")
        # print(len(val_list), val_list[0])
        # test_list = load_decathlon_datalist(json_path, True, "test")
        test_list = None

    elif dataset_name == "totalsegmentator":
        data_root = DataPath.TOTALSEGMENTATOR
        json_path = os.path.abspath(data_root + "/data_split.json")

        train_list = load_decathlon_datalist(json_path, True, "training")
        # print(len(train_list), train_list[0])
        val_list = load_decathlon_datalist(json_path, True, "validation")
        # print(len(val_list), val_list[0])
        test_list = load_decathlon_datalist(json_path, True, "test")
        
    return train_list, val_list, test_list

def visualize_transformed_data(dataset, case_num, slice_map, save=False):

    if isinstance(dataset[case_num], dict):
        data_sample = dataset[case_num]
    elif isinstance(dataset[case_num], list):
        data_sample = dataset[case_num][0]
    # print(data_sample["image"].meta)

    if isinstance(data_sample["image"], torch.Tensor):
        img_name = list(slice_map.keys())[case_num]
    elif not data_sample["image"].meta.get("filename_or_obj", None):
        img_name = list(slice_map.keys())[case_num]
    else:
        img_name = os.path.split(data_sample["image"].meta["filename_or_obj"])[1]

    img = data_sample["image"]
    label = data_sample["label"]
    print(f"image shape: {img.shape}, label shape: {label.shape}")
    print(f"image max: {img.max()}, image min: {img.min()}, image mean: {img.mean()}")
    print(f"label max: {label.max()}, label min: {label.min()}, unique label: {torch.unique(label)}")

    for i in range(label.shape[-1]):
        if torch.unique(label[:, :, :, i]).numel() > 2:
            # print(torch.unique(label[:, :, :, i]).numel())
            slice_map[img_name] = i
            break

    plt.figure("image", (12, 6))

    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())

    if save:
        plt.savefig("./check_data/image_label.jpg", format="jpg", bbox_inches="tight", dpi=300)
    else:
        plt.show()

def visualize_predicition(model, dataset, slice_map, case_num, save=True, device="cuda"):
    model.to(device)

    if isinstance(dataset[case_num], dict):
        data_sample = dataset[case_num]
    elif isinstance(dataset[case_num], list):
        data_sample = dataset[case_num][0]
    # print(data_sample["image"].meta)

    if isinstance(data_sample["image"], torch.Tensor):
        img_name = list(slice_map.keys())[case_num]
    elif not data_sample["image"].meta.get("filename_or_obj", None):
        img_name = list(slice_map.keys())[case_num]
    else:
        img_name = os.path.split(data_sample["image"].meta["filename_or_obj"])[1]

    image = data_sample["image"].unsqueeze(dim=0).to(device)
    label = data_sample["label"].unsqueeze(dim=0).to(device)
    # print(f"image shape: {image.shape}, label shape: {label.shape}")

    with torch.no_grad():
        output = sliding_window_inference(
            image,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.20,
        )
    # print(f"prediciton shape: {output.shape}")


    plt.figure("comparison", (18, 6))

    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(image.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("ground truth")
    plt.imshow(label.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
    plt.subplot(1, 3, 3)
    plt.title("prediciton")
    plt.imshow(torch.argmax(output, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
    
    if save:
        plt.savefig("./check_data/image_gt_pred.jpg", bbox_inches="tight", format="jpg", dpi=300)
    else:
        plt.show()



@torch.no_grad()
def test_model(model, dataloader, criterion, per_class=False, device="cuda", mode="validating"):

    metric_dict = dict()

    # dice_metric_perclass = DiceMetric(reduction="mean_batch", get_not_nans=False, return_with_label=True)
    dice_metric_mean = DiceMetric(reduction="mean")

    epoch_loss = 0
    epoch_dice = 0
    epoch_iterator = tqdm(dataloader, desc=f"{mode} loss=X.X", dynamic_ncols=True)
    for i, batch in enumerate(epoch_iterator):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = sliding_window_inference(
            images,
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            predictor=model,
            overlap=0.25,
        )
        num_classes = outputs.shape[1]

        loss = criterion(outputs, labels)

        outputs_indices = torch.argmax(outputs, dim=1, keepdim=True)
        outputs_onehot = monai.networks.utils.one_hot(outputs_indices, num_classes=num_classes, dim=1)
        labels_onehot = monai.networks.utils.one_hot(labels, num_classes=num_classes, dim=1)

        # dice_metric_perclass(y_pred=outputs_onehot, y=labels_onehot)
        dice_metric_mean(y_pred=outputs_onehot, y=labels_onehot)

        # val_labels_list = decollate_batch(labels)
        # val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list] 
        # val_outputs_list = decollate_batch(outputs)
        # val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]        
        # dice_metric_detail(y_pred=val_output_convert, y=val_labels_convert)
        # dice_metric_mean(y_pred=val_output_convert, y=val_labels_convert)

        epoch_loss += loss.item() / len(epoch_iterator)
        epoch_iterator.set_description(
            f"{mode} loss={loss:2.5f}"
        )
    
    dice_mean = dice_metric_mean.aggregate().item()

    metric_dict["loss"] = epoch_loss
    metric_dict["dice_mean"] = dice_mean

    if per_class:
        # dice_perclass = dice_metric_perclass.aggregate()
        metric_dict["dice_perclass"] = dice_perclass

    return metric_dict