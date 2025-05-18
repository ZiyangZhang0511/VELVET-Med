import torch

import monai.transforms as mt

def sample_and_resize(img):
    """
    img shape: [C, H, W, D] = [1, 512, 512, 100]
    1) Sample 96 slices along dimension D=100 -> D=96
    2) Resize from (H=512, W=512, D=96) to (96, 96, 96).
    """

    # 1) Uniformly sample 96 indices in the last dimension
    current_d = img.shape[-1]  # 100
    target_d = 96
    indices = torch.linspace(0, current_d-1, steps=target_d)  # 0..99 => 96 steps
    indices = torch.round(indices).long()  # make integer indices

    # shape becomes [C, H, W, 96]
    sampled = img[..., indices]

    # 2) Resize from (H=512, W=512, D=96) to (96, 96, 96)
    # We can use MONAI's 3D Resize transform:
    resize_transform = mt.Resize(spatial_size=(96, 96, 96), mode="area")
    out = resize_transform(sampled)

    return out

class MonaiTransforms:

    def __init__(self, num_samples=1, device="cuda"):
        self.num_samples = num_samples
        self.device = device

    def load_vqa_transforms(self, mode="global"):
        vqa_transforms_dict = {
            "global": mt.Compose([
                mt.Orientation(axcodes="RAS"),
                mt.ScaleIntensityRange(
                    a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True,
                ),
                mt.Spacing(
                    pixdim=(1.0, 1.0, 0.5),
                    mode=("bilinear"),
                    min_pixdim=(1.0, 1.0, 1.0),
                    max_pixdim=None,
                    align_corners=False,
                ),
                mt.CropForeground(allow_smaller=False),
                # mt.SpatialPad(spatial_size=[96, 96, 96]),
                mt.Lambda(func=sample_and_resize),
            ]),
            "clip3d": mt.Compose([
                mt.Orientation(axcodes="RAS"),
                mt.ScaleIntensityRange(
                    a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True,
                ),
                mt.CropForeground(allow_smaller=False),
                mt.Resize(spatial_size=(256, 256, 32), mode="area"),
            ]),
        }
        
        return vqa_transforms_dict[mode]

    def load_ssl_transforms(self, args=None, mode="local"):

        if args is None or args.dataset_name in ["m3d_cap"]:
            ssl_transforms_dict = {
                "clip3d": mt.Compose([
                    mt.Orientation(axcodes="RAS"),
                    mt.ScaleIntensityRange(
                        a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    mt.CropForeground(allow_smaller=False),
                    mt.Resize(spatial_size=(256, 256, 32), mode="area"),
                ]),

                "global": mt.Compose([
                    mt.Orientation(axcodes="RAS"),
                    mt.ScaleIntensityRange(
                        a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    mt.Spacing(
                        pixdim=(1.0, 1.0, 0.5),
                        mode=("bilinear"),
                        min_pixdim=(1.0, 1.0, 1.0),
                        max_pixdim=None,
                        align_corners=False,
                    ),
                    mt.CropForeground(allow_smaller=False),
                    mt.SpatialPad(spatial_size=[96, 96, 96]),
                    mt.Lambda(func=sample_and_resize), 
                ]),

                "local": mt.Compose([
                    mt.Orientation(axcodes="RAS"),
                    mt.ScaleIntensityRange(
                        a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    mt.Spacing(
                        pixdim=(1.0, 1.0, 0.5),
                        mode=("bilinear"),
                        min_pixdim=(1.0, 1.0, 1.0),
                        max_pixdim=None,
                        align_corners=False,
                    ),
                    mt.CropForeground(allow_smaller=False),
                    mt.SpatialPad(spatial_size=[96, 96, 96]),
                    mt.RandSpatialCropSamples(
                        roi_size=[96, 96, 96],
                        num_samples=self.num_samples,
                        random_center=True,
                        random_size=False,
                    ),
                ]),
                "val": mt.Compose([
                    mt.Orientation(axcodes="RAS"),
                    mt.ScaleIntensityRange(
                        a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    mt.Spacing(
                        pixdim=(1.0, 1.0, 0.5),
                        mode=("bilinear"),
                        min_pixdim=(1.0, 1.0, 1.0),
                        max_pixdim=None,
                        align_corners=False,
                    ),
                    mt.CropForeground(allow_smaller=False),
                    mt.SpatialPad(spatial_size=[96, 96, 96]),
                    mt.RandSpatialCropSamples(
                        roi_size=[96, 96, 96],
                        num_samples=self.num_samples,
                        random_center=False,
                        random_size=False,
                    ),
                ])
            }

        elif args.pretrain_type in ["vitautoenc_ssl"] and args.dataset_name in ["tcia_covid19"]:
            ssl_transforms_dict = {
                "train": mt.Compose([
                    mt.LoadImaged(keys=["image"]),
                    mt.EnsureChannelFirstd(keys=["image"]),
                    mt.Orientationd(keys=["image"], axcodes="RAS"),
                    mt.Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
                    mt.ScaleIntensityRanged(
                        keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    mt.CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
                    mt.SpatialPadd(keys=["image"], spatial_size=[96, 96, 96]),
                    mt.RandSpatialCropSamplesd(
                        keys=["image"],
                        roi_size=[96, 96, 96],
                        num_samples=self.num_samples,
                        random_center=True,
                        random_size=False,
                    ),
                    mt.CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
                    mt.OneOf(
                        transforms=[
                            mt.RandCoarseDropoutd(
                                keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                            ),
                            mt.RandCoarseDropoutd(
                                keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                            ),
                        ]
                    ),
                    mt.RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
                    # Please note that that if image, image_2 are called via the same transform call because of the determinism
                    # they will get augmented the exact same way which is not the required case here, hence two calls are made
                    mt.OneOf(
                        transforms=[
                            mt.RandCoarseDropoutd(
                                keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                            ),
                            mt.RandCoarseDropoutd(
                                keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                            ),
                        ]
                    ),
                    mt.RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
                ])
            }

        elif args.pretrain_type in ["vis_ssl"] and args.dataset_name in ["tcia_covid19"]:
            ssl_transforms_dict = {
                "train": mt.Compose([
                    mt.Orientation(axcodes="RAS"),
                    mt.ScaleIntensityRange(
                        a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    ),
                    mt.Spacing(
                        pixdim=(2.0, 2.0, 2.0),
                        mode=("bilinear"),
                        # min_pixdim=(1.0, 1.0, 1.0), 
                        # max_pixdim=None,
                        align_corners=False,
                    ),
                    mt.CropForeground(allow_smaller=False),
                    mt.SpatialPad(spatial_size=[96, 96, 96]),
                    mt.RandSpatialCropSamples(
                        roi_size=[96, 96, 96],
                        num_samples=self.num_samples,
                        random_center=True,
                        random_size=False,
                    ),
                ])
            }

        return ssl_transforms_dict[mode]

    def load_supervised_seg_transforms(self, mode="train"):
        ###
        # used for btcv
        ###

        supervised_seg_transforms_dict = {
            "train": mt.Compose([
                mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                mt.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                mt.CropForegroundd(keys=["image", "label"], source_key="image"),
                mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
                mt.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"),),
                mt.SpatialPadd(keys=["image", "label"], spatial_size=[96, 96, 96]),
                mt.EnsureTyped(keys=["image", "label"], track_meta=False),
                mt.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=self.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                mt.RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10,),
                mt.RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10,),
                mt.RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10,),
                mt.RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3,),
                mt.RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50,),
            ]),
            "val": mt.Compose([
                mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                mt.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                mt.CropForegroundd(keys=["image", "label"], source_key="image"),
                mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
                mt.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"),),
                mt.EnsureTyped(keys=["image", "label"], track_meta=True),
            ])
        }

        return supervised_seg_transforms_dict[mode]