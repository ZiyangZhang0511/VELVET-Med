import os
import yaml
import pickle
import argparse
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import seed_everything

from modeling.text_model import TextSSL
from modeling.vision_model import VitAutoEncSSL, VisionSSL
from modeling.visiontext_model import VisionTextSSL
from modeling.m3dcap_model import CLIP3DSSL

from utilities import utils
from utilities.constants import SliceMap, ConfigPath, PretainSaveDir


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16)."
             "Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )

    parser.add_argument("--pretrain_type", type=str, choices=["vis_ssl", "vl_ssl", "vitautoenc_ssl", "txt_ssl", "clip3d_ssl"])
    parser.add_argument("--vision_encoder", type=str, default="none", choices=["swinvit", "vit", "none"])
    parser.add_argument("--dataset_name", type=str, choices=["m3d_cap", "tcia_covid19"])

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--num_samples_per_volume", type=int, default=1)
    parser.add_argument("--initial_lr", type=float, default=2e-5)
    parser.add_argument("--monitored_loss", type=str, default="loss")

    parser.add_argument("--config_path", default=None, type=str)
    parser.add_argument("--ckpt_save_dir", default=None, type=str)
    parser.add_argument("--recon_save_dir", default=None, type=str) # default="./check_data/vis_ssl_recon",

    parser.add_argument("--restart_optimizer", action="store_true")
    parser.add_argument("--cur_epoch", default=-1, type=int)
    parser.add_argument("--resume_pretraining", action="store_true")

    args = parser.parse_args()

    return args


def train_function(model, train_dataloader, val_dataloader, args):

    if args.ckpt_save_dir:
        ckpt_save_dir = args.ckpt_save_dir
    else:
        if args.pretrain_type == "vis_ssl":
            ckpt_save_dir = PretainSaveDir.VISION_SSL
        elif args.pretrain_type == "vitautoenc_ssl":
            ckpt_save_dir = PretainSaveDir.VITAUTOENC_SSL
        elif args.pretrain_type == "txt_ssl":
            ckpt_save_dir = PretainSaveDir.TEXT_SSL
        elif args.pretrain_type == "vl_ssl":
            ckpt_save_dir = PretainSaveDir.VISIONTEXT_SSL
        elif args.pretrain_type == "clip3d_ssl":
            ckpt_save_dir = PretainSaveDir.CLIP3D_SSL
        
    ckpt_stem = f"{args.pretrain_type}-{args.vision_encoder}-{args.dataset_name}-dr{args.data_ratio}"

    os.makedirs(ckpt_save_dir, exist_ok=True)
    if args.recon_save_dir:
        os.makedirs(args.recon_save_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.98), weight_decay=0.02)
    scheduler = CosineAnnealingLR(optimizer, len(train_dataloader)*args.num_epochs, eta_min=1e-8)

    model, train_dataloader, optimizer, scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, scheduler
    )
    # accelerator.print(model)

    best_loss_filepath = os.path.join(ckpt_save_dir, f"best_loss-{ckpt_stem}.pkl")
    
    
    if not args.resume_pretraining:
        cur_epoch = -1
        best_val_loss = 1e5
        accelerator.save_state(ckpt_save_dir + f"/{ckpt_stem}-E{cur_epoch}", safe_serialization=False)
    else:
        cur_epoch = args.cur_epoch
        with open(best_loss_filepath, 'rb') as file:
            best_val_loss = pickle.load(file)
        accelerator.print(f"current best loss: {best_val_loss:2.4f}")
        accelerator.load_state(ckpt_save_dir + f"/{ckpt_stem}-E{cur_epoch}")


    if args.restart_optimizer:
        optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.98), weight_decay=0.02)
        scheduler = CosineAnnealingLR(optimizer, len(train_dataloader)*args.num_epochs, eta_min=1e-8)
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=len(train_dataloader)*10,
        #     max_epochs=len(train_dataloader)*args.num_epochs,
        #     warmup_start_lr=1e-5, 
        #     eta_min=1e-8,
        # )
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    training_step = 0
    accelerator.print("Start training......")
    for epoch in range(cur_epoch+1, args.num_epochs):
        epoch_loss_dict = {}
        model.train()

        cur_lr = scheduler.get_last_lr()
        accelerator.print(f"==== Training current lr: {cur_lr[-1]} ====")
        for i, batch_data in enumerate(tqdm(train_dataloader)):
            # break
        
            # with accelerator.accumulate(model):
            for key in batch_data.keys():
                # if "volume" in key:
                batch_data[key] = batch_data[key].to(accelerator.device)
            
            with accelerator.autocast():
                loss_dict = model(batch_data)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_loss_dict.keys():
                    epoch_loss_dict[loss_name] = 0
                epoch_loss_dict[loss_name] += loss_value / len(train_dataloader)
            
            training_step += 1 * accelerator.num_processes

        utils.print_dict_content(epoch_loss_dict, f"Epoch-{epoch} training results:", accelerator)

        ###======== validating model ========###
        accelerator.print("====== Validating model ======")

        model_unwraped = accelerator.unwrap_model(model)
        model_unwraped.eval()

        val_dict = model_unwraped.test_on_dataloader(val_dataloader, training_step, args.recon_save_dir)
        utils.print_dict_content(val_dict, f"Epoch-{epoch} validation results:", accelerator)
        accelerator.wait_for_everyone()
        
        val_loss = val_dict[args.monitored_loss]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(best_loss_filepath, 'wb') as file:
                pickle.dump(best_val_loss, file)
            # torch.save({
            #     "model_state_dict": model.state_dict(),
            #     "val_loss": best_val_loss,
            #     "val_metric_dict": val_metric_dict,
            # }, ckpt_save_path)
            accelerator.save_state(ckpt_save_dir + f"/{ckpt_stem}-E{epoch}-best", safe_serialization=False)
            accelerator.print(f"saved better checkpoint at epoch {epoch}.")
        
        elif accelerator.is_main_process: 
            accelerator.save_state(ckpt_save_dir + f"/{ckpt_stem}-E{epoch}", safe_serialization=False)
            accelerator.print(f"saved at epoch {epoch}")
        accelerator.wait_for_everyone()

        if epoch > 0 and accelerator.is_main_process:
            dir_path = ckpt_save_dir + f"/{ckpt_stem}-E{epoch-2}"
            if os.path.exists(dir_path):
                utils.remove_directory(dir_path)
        accelerator.wait_for_everyone()

        
def main():

    args = get_args()

    ###======== prepare dataset and dataloader ========###
    train_dataset, val_dataset = utils.get_ssl_dataset(args)
    print(len(train_dataset), len(val_dataset))

    # if args.dataset_name == "tcia_covid19":
        # slice_map = SliceMap.tcia_covid19_slice_map
    # utils.visualize_transformed_data(train_dataset, 0, slice_map["train"], save=False)
    train_dataloader, val_dataloader = utils.get_dataloader(train_dataset, val_dataset, args)


    ###======== build model ========###
    if args.pretrain_type == "vis_ssl":
        config_path = os.path.abspath(ConfigPath.VISION_SSL)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = VisionSSL(config)

    elif args.pretrain_type == "vitautoenc_ssl":
        model = VitAutoEncSSL()

    elif args.pretrain_type == "clip3d_ssl":
        model = CLIP3DSSL()

    elif args.pretrain_type == "txt_ssl":
        config_path = os.path.abspath(ConfigPath.TEXT_SSL)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = TextSSL(config)
    
    elif args.pretrain_type == "vl_ssl":
        if args.config_path and os.path.isfile(args.config_path):
            config_path = args.config_path
        elif args.config_path:
            raise RuntimeError("User provided configuration file doesn't exist!!!")
        else:
            config_path = os.path.abspath(ConfigPath.VISIONTEXT_SSL)
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = VisionTextSSL(config)
    

    ###======== training function and save ckpt ========###
    train_function(model, train_dataloader, val_dataloader, args)

    


if __name__ == "__main__":
    seed_everything(1)
    main()