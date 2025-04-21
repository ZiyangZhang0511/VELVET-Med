class DataPath:
    TCIA_COVID19 = "./data/tcia_covid19"
    BTCV = "./data/btcv/RawData"
    ABDOMENCT1K = "./data/abdomenct1k"
    CTORG = "./data/ctorg"
    TOTALSEGMENTATOR = "./data/totalsegmentator"
    M3D_CAP = "./data/m3d_cap"
    M3D_VQA = "./data/m3d_vqa/all_case"

class ConfigPath:
    VISION_SSL = "./config_ssl/vision_ssl.yaml"
    TEXT_SSL = "./config_ssl/text_ssl.yaml"
    VISIONTEXT_SSL = "./config_ssl/visiontext_ssl.yaml"

class PretainSaveDir:
    VISIONTEXT_SSL = "./checkpoints/visiontext_ssl"
    VISION_SSL = "./checkpoints/vision_ssl"
    VITAUTOENC_SSL = "./checkpoints/vitautoenc_ssl"
    TEXT_SSL = "./checkpoints/text_ssl"
    CLIP3D_SSL = "./checkpoints/clip3d_ssl"


# totalsegmentator: (16724.0, -10248.0)

class SliceMap:
    btcv = {
        "train": {
            "img0001.nii.gz": 17,
            "img0002.nii.gz": 23,
            "img0003.nii.gz": 20,
            "img0004.nii.gz": 20,
            "img0005.nii.gz": 17,
            "img0006.nii.gz": 23,
            "img0007.nii.gz": 20,
            "img0008.nii.gz": 20,
            "img0009.nii.gz": 20,
            "img0010.nii.gz": 18,
        },

        "val": {
            "img0035.nii.gz": 170,
            "img0036.nii.gz": 230,
            "img0037.nii.gz": 204,
            "img0038.nii.gz": 204,
            "img0039.nii.gz": 204,
            "img0040.nii.gz": 180,
        }
    }

    totalsegmentator = {
        "train": {
            "s0001/ct.nii.gz": 39,
            "s0002/ct.nii.gz": 23,
            "s0003/ct.nii.gz": 20,
            "s0004/ct.nii.gz": 20,
        },

        "val": {
            "s0000/ct.nii.gz": 70,
            "s0021/ct.nii.gz": 23,
            "s0032/ct.nii.gz": 24,
            "s0045/ct.nii.gz": 24,
            "s0095/ct.nii.gz": 20,
        }
    }

    abdomenct1k = {
        "train": {
            "Case_00001_0000.nii.gz": 39,
            "Case_00002_0000.nii.gz": 23,
            "Case_00003_0000.nii.gz": 20,
            "Case_00004_0000.nii.gz": 20,
        },

        "val": {
            "Case_00514_0000.nii.gz": 70,
            "Case_00342_0000.nii.gz": 23,
            "Case_00987_0000.nii.gz": 24,
            "Case_00689_0000.nii.gz": 24,
            "Case_00181_0000.nii.gz": 20,
        }
    }

    ctorg = {
        "train": {
            "volumes-21.nii.gz": 39,
            "volumes-22.nii.gz": 23,
            "volumes-23.nii.gz": 20,
            "volumes-24.nii.gz": 20,
        },

        "val": {
            "volumes-0.nii.gz": 17,
            "volumes-1.nii.gz": 23,
            "volumes-2.nii.gz": 20,
            "volumes-3.nii.gz": 24,
            "volumes-4.nii.gz": 24,
        }
    }

    tcia_covid19 = {
        "train": {
            "volume-covid19-A-0269.nii.gz": 17,
            "volume-covid19-A-0012.nii.gz": 23,
            "volume-covid19-A-0721_day008.nii.gz": 20,
            "volume-covid19-A-0154.nii.gz": 20,
            "volume-covid19-A-0358.nii.gz": 17,
            "volume-covid19-A-0734_day002.nii.gz": 23,
            "volume-covid19-A-0304.nii.gz": 20,
            "volume-covid19-A-0640.nii.gz": 20,
            "volume-covid19-A-0263.nii.gz": 20,
            "volume-covid19-A-0737_day000.nii.gz": 18,
        },

        "val": {
            "volume-covid19-A-0656.nii.gz": 70,
            "volume-covid19-A-0495.nii.gz": 30,
            "volume-covid19-A-0705_day043.nii.gz": 24,
            "volume-covid19-A-0264.nii.gz": 24,
            "volume-covid19-A-0300.nii.gz": 20,
            "volume-covid19-A-0722_day047.nii.gz": 18,
        }
    }
