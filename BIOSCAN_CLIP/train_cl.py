import copy
import datetime
import json
import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf

from epoch.train_epoch import train_epoch
from inference_and_eval import get_features_and_label, inference_and_print_result
from model.loss_func import ContrastiveLoss, ClipLoss
from model.simple_clip import load_clip_model
from util.dataset import load_bioscan_dataloader, load_bioscan_6M_dataloader


def save_prediction(pred_list, gt_list, json_path):
    data = {
        "gt_labels": gt_list,
        "pred_labels": pred_list
    }

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)

def ddp_setup(rank: int, world_size: int, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def eval_phase(model, device, all_keys_dataloader, seen_val_dataloader, unseen_val_dataloader, k_list,
               species_to_drop=None, rank=None):
    keys_dict = get_features_and_label(
        all_keys_dataloader, model, device, for_key_set=True)

    seen_val_dict = get_features_and_label(
        seen_val_dataloader, model, device)

    unseen_val_dict = get_features_and_label(
        unseen_val_dataloader, model, device)

    acc_dict, _, pred_dict = inference_and_print_result(keys_dict, seen_val_dict, unseen_val_dict,
                                                     small_species_list=None, k_list=k_list)
    return acc_dict, pred_dict

def convert_acc_dict_to_wandb_dict(acc_dict):
    dict_for_wandb = {}
    acc_dict = acc_dict['encoded_image_feature']['encoded_dna_feature']

    # For now, we just put query: image and key:image acc to wandb

    for split, split_dict in acc_dict.items():
        for type_of_acc, type_of_acc_dict in split_dict.items():
            for k, k_dict in type_of_acc_dict.items():
                for level, curr_acc in split_dict.items():
                    dict_for_wandb[f"{split} {type_of_acc} top-{k} {level} level"] = curr_acc

    return dict_for_wandb



def broadcast_model(model, rank):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def main_process(rank: int, world_size: int, args):

    if args.debug_flag or rank != 0:
        args.activate_wandb = False
        args.save_inference = False
        args.save_ckpt = False



    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    args = copy.deepcopy(args)

    ddp_setup(rank, world_size, str(args.model_config.port))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print("Construct dataloader...")
    if hasattr(args.model_config, 'dataset') and args.model_config.dataset == 'BIOSCAN_6M':
        pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_bioscan_6M_dataloader(
            args, world_size=world_size, rank=rank)
    else:
        pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_bioscan_dataloader(
            args, world_size=world_size, rank=rank)

    if rank == 0:
        print("Initialize model...")
    model = load_clip_model(args)
    model = model.to(device)
    broadcast_model(model, rank)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    open_clip_ver = False
    if hasattr(args.model_config, 'open_clip_ver') and args.model_config.open_clip_ver:
        open_clip_ver = True
        criterion = ClipLoss(local_loss=args.model_config.loss_setup.local_loss, gather_with_grad=args.model_config.loss_setup.gather_with_grad, rank=rank, world_size=world_size, use_horovod=args.model_config.loss_setup.use_horovod, criterion=nn.CrossEntropyLoss())

    else:
        criterion = ContrastiveLoss(criterion=nn.CrossEntropyLoss(), logit_scale=1 / 0.07)


    if args.activate_wandb:
        wandb.init(project=args.model_config.wandb_project_name, name=args.model_config.model_output_name)
    if rank == 0:
        print("training...")

    k_list = [1, 3, 5]

    best_epoch = None
    best_overall_acc = None
    folder_path = os.path.join(args.project_root_path, args.model_output_dir,
                               args.model_config.model_output_name, formatted_datetime)
    os.makedirs(folder_path, exist_ok=True)

    OmegaConf.save(args, os.path.join(folder_path, 'config.yaml'))


    for epoch in range(args.model_config.epochs):
        train_epoch(args.activate_wandb, args.model_config.epochs, epoch, pre_train_dataloader, model, optimizer,
                    criterion, device, open_clip_ver=open_clip_ver, rank=rank)
        if epoch % args.model_config.evaluation_period == 0 or epoch == args.model_config.epochs - 1 and rank == 0:
            acc_dict, pred_dict = eval_phase(model, device, all_keys_dataloader, seen_val_dataloader, unseen_val_dataloader, k_list, rank=rank)
            # TODO: Simplify the result that will forward to wandb.
            dict_for_wandb = convert_acc_dict_to_wandb_dict(acc_dict)
            dict_for_wandb['epoch'] = epoch
            # Find a way to calculate overall acc.
            # OR just use seen and unseen micro top-1 species acc. to determine the best ckpt.
            overall_acc = (acc_dict['encoded_image_feature']['encoded_image_feature']['seen_val']['micro_acc'][1]['species'] + acc_dict['encoded_image_feature']['encoded_image_feature']['unseen_val']['micro_acc'][1]['species'])/2
            if best_overall_acc is None or best_overall_acc < overall_acc:
                best_epoch = epoch
                best_overall_acc = overall_acc
                if args.save_ckpt:
                    best_ckpt_path = os.path.join(folder_path, f'best.pth')
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f'Best ckpt: {best_ckpt_path}')
            dict_for_wandb["overall_acc"] = overall_acc
            dict_for_wandb["best_epoch"] = best_epoch
            if args.activate_wandb:
                wandb.log(dict_for_wandb,
                          commit=True)

            if args.save_ckpt:
                last_ckpt_path = os.path.join(folder_path, f'last.pth')
                torch.save(model.state_dict(), last_ckpt_path)
                print(f'Last ckpt: {last_ckpt_path}')


@hydra.main(config_path="config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    world_size = torch.cuda.device_count()
    print(f'world_size： {world_size}')
    mp.spawn(main_process, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()
