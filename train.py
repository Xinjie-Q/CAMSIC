import argparse
import math
import random
import sys
import time

from collections import defaultdict
from typing import List


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models.camsic import *
from models.elic import *
from models.loss import *
from lib.utils import get_output_folder, AverageMeter, save_checkpoint, StereoImageDataset, pad, crop
import numpy as np
import yaml
import os
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from compressai.zoo.image import model_architectures, model_urls, cfgs
from compressai.zoo.pretrained import load_pretrained
from torch.hub import load_state_dict_from_url

# torch.backends.cudnn.benchmark = True
# torch.set_default_dtype(torch.float64)

# import wandb
# os.environ["WANDB_API_KEY"] = "56d69e32e2234c64a18fff3729afecc6c5eeef27"

def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
        if backward is True:
            aux_loss.backward()

    return aux_loss_sum

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(p for p in net.named_parameters() if p[1].requires_grad)
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    if len(aux_parameters) == 0:
        aux_optimizer = None
    else:
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.learning_rate,
        )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, args):
    model.train()
    device = next(model.parameters()).device
    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = args.metric+"0", args.metric+"1"

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')

    train_dataloader = tqdm(train_dataloader)
    print('Train epoch:', epoch)
    for i, batch in enumerate(train_dataloader):
        d = [frame.to(device) for frame in batch]
        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()
        #aux_optimizer.zero_grad()
        
        out_net = model(d)

        out_criterion = criterion(out_net, d, args.lmbda, epoch)

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if aux_optimizer is not None:
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
            aux_optimizer.step()
            aux_loss.update(out_aux_loss.item())
        # else:
        #     out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)
        #out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
        #aux_optimizer.step()

        loss.update(out_criterion["loss"].item())
        bpp_loss.update((out_criterion["bpp_loss"]).item())

        metric_loss.update(out_criterion[metric_name].item())
        
        left_bpp.update(out_criterion["bpp0"].item())
        right_bpp.update(out_criterion["bpp1"].item())

        if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
            left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
            right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
            left_db.update(left_metric)
            right_db.update(right_metric)
            metric_dB.update((left_metric+right_metric)/2)

        train_dataloader.set_description('[{}/{}]'.format(i, len(train_dataloader)))
        train_dataloader.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
            metric_dB_name:metric_dB.avg})

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg, "right_bpp": right_bpp.avg,
            left_db_name:left_db.avg, right_db_name: right_db.avg,}

    return out

def test_epoch(epoch, val_dataloader, model, criterion, aux_optimizer, args):
    model.eval()
    device = next(model.parameters()).device

    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = args.metric+"0", args.metric+"1"

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')
    loop = tqdm(val_dataloader)

    with torch.no_grad():
        for i, batch in enumerate(loop):
            d = [frame.to(device) for frame in batch]
            d0, padding = pad(d[0], p=2**(4+2))
            d1, _ = pad(d[1], p=2**(4+2))
            out_net = model([d0, d1])
            out_net["x_hat"] = [crop(x, padding) for x in out_net["x_hat"]]
            out_criterion = criterion(out_net, d, args.lmbda, epoch)

            if aux_optimizer is not None:
                out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)
                aux_loss.update(out_aux_loss.item())

            loss.update(out_criterion["loss"].item())
            bpp_loss.update((out_criterion["bpp_loss"]).item())
            
            metric_loss.update(out_criterion[metric_name].item())
        
            left_bpp.update(out_criterion["bpp0"].item())
            right_bpp.update(out_criterion["bpp1"].item())

            if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
                left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
                right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
                left_db.update(left_metric)
                right_db.update(right_metric)
                metric_dB.update((left_metric+right_metric)/2)

            loop.set_description('[{}/{}]'.format(i, len(val_dataloader)))
            loop.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
                metric_dB_name:metric_dB.avg})

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg, "right_bpp": right_bpp.avg,
            left_db_name:left_db.avg, right_db_name: right_db.avg,}

    return out


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/Instereo2K/', help="Training dataset"
    )
    parser.add_argument(
        "--data-name", type=str, default='instereo2K', help="Training dataset"
    )
    parser.add_argument(
        "--model-name", type=str, default='LDMIC', help="Training dataset"
    )

    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=2048,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", help="Save model to disk"
    )
    parser.add_argument(
        "--resize", action="store_true", help="training use resize or randomcrop"
    )
    parser.add_argument(
        "--seed", type=float, default=1, help="Set random seed for reproducibility"
    )
    parser.add_argument("--num_heads", type=int, default=12, help="Set random seed for reproducibility")
    parser.add_argument("--depths", type=int, default=8, help="Set random seed for reproducibility")
    parser.add_argument("--win_size", type=int, default=8, help="Set random seed for reproducibility")
    parser.add_argument(
        "--iq",
        type=int,
        default=6,
        help="IFrame_quality",
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--i_model_path", type=str, default="./pretrained_ckpt/ELIC_0450_ft_3980_Plateau.pth.tar", help="Path to a checkpoint")
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms-ssim")
    parser.add_argument("--sche_name", type=str, default="steplr", help="metric: mse, ms-ssim")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Save model to disk"
    )
    parser.add_argument(
        "--fix_model", action="store_true", help="Save model to disk"
    )
    parser.add_argument(
        "--resume", action="store_true", help="load previous optimizer and lr scheduler!"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Warning, the order of the transform composition should be kept.
    train_dataset = StereoImageDataset(ds_type='train', ds_name=args.data_name, root=args.dataset, crop_size=args.patch_size)
    test_dataset = StereoImageDataset(ds_type='test', ds_name=args.data_name, root=args.dataset, crop_size=args.patch_size)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
        shuffle=True, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"))

    if args.model_name in ["CAMSIC"]:
        net = CAMSIC(arch_name="ELIC", num_channels=320, context_len=1, embed_dim=768, depths=[args.depths], 
            num_heads=args.num_heads, window_size=args.win_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
            drop_path_rate=0., norm_layer=nn.LayerNorm, scratch=False)

  

    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    if args.sche_name == "steplr":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], 0.5)
    else:
        lr_scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)
    if args.metric == "mse":
        criterion = MSE_Loss() #MSE_Loss(lmbda=args.lmbda)
    else:
        criterion = MS_SSIM_Loss(device) #(device, lmbda=args.lmbda)
    last_epoch = 0
    best_loss = float("inf")

    if args.pretrained and not args.i_model_path:
        print("load pretrained image compression model from CompressAI")
        url = model_urls["cheng2020-anchor"][args.metric][args.iq]
        checkpoint = load_state_dict_from_url(url, progress=True, map_location="cpu")
        checkpoint = load_pretrained(checkpoint)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if "g_a" in k or "g_s" in k}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        if args.fix_model:
            net.fix_encoder_decoder()
    
    elif args.i_model_path:
        print(f"Loading model:{args.i_model_path}")
        checkpoint = torch.load(args.i_model_path, map_location="cpu")
        if "state_dict" in checkpoint.keys():
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "g_a" in k or "g_s" in k} #if k in model_dict}
        #print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        if args.resume:  # load from previous checkpoint 
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if aux_optimizer is not None:
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])


    log_dir, experiment_id = get_output_folder('./checkpoints/{}/{}/{}/lamda{}/'.format(args.data_name, args.metric, args.model_name, int(args.lmbda)), 'train')
    display_name = "{}_{}_lmbda{}".format(args.model_name, args.metric, int(args.lmbda))
    writer = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    
    # tags = "lmbda{}".format(args.lmbda)
    # project_name = "CAMSIC_" + args.data_name
    # wandb.init(project=project_name, name=display_name, tags=[tags],) #notes="lmbda{}".format(args.lmbda))
    # wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
    # wandb.config.update(args) # config is a variable that holds and saves hyper parameters and inputs


    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    val_loss = test_epoch(0, test_dataloader, net, criterion, aux_optimizer, args)
    for epoch in range(last_epoch, args.epochs):
        #adjust_learning_rate(optimizer, aux_optimizer, epoch, args)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, args)
        # wandb.log({"train": {"loss": train_loss["loss"], metric_name: train_loss[metric_name], "bpp_loss": train_loss["bpp_loss"],
        #     "aux_loss": train_loss["aux_loss"], metric_dB_name: train_loss[metric_dB_name], "left_bpp": train_loss["left_bpp"], "right_bpp": train_loss["right_bpp"],
        #     left_db_name:train_loss[left_db_name], right_db_name: train_loss[right_db_name]}, }
        # )

        writer.add_scalar('train/loss', train_loss["loss"], epoch)
        writer.add_scalar(f'train/{metric_name}', train_loss[metric_name], epoch)
        writer.add_scalar('train/bpp_loss', train_loss["bpp_loss"], epoch)
        writer.add_scalar('train/aux_loss', train_loss["aux_loss"], epoch)  
        writer.add_scalar('train/left_bpp', train_loss["left_bpp"], epoch)
        writer.add_scalar('train/right_bpp', train_loss["right_bpp"], epoch)
        writer.add_scalar(f'train/{left_db_name}', train_loss[left_db_name], epoch)
        writer.add_scalar(f'train/{right_db_name}', train_loss[right_db_name], epoch)



        val_loss = test_epoch(epoch, test_dataloader, net, criterion, aux_optimizer, args)
        writer.add_scalar('val/loss', val_loss["loss"], epoch)
        writer.add_scalar(f'val/{metric_name}', val_loss[metric_name], epoch)
        writer.add_scalar('val/bpp_loss', val_loss["bpp_loss"], epoch)
        writer.add_scalar('val/aux_loss', val_loss["aux_loss"], epoch)
        writer.add_scalar(f'val/{metric_dB_name}', val_loss[metric_dB_name], epoch)  
        writer.add_scalar('val/left_bpp', val_loss["left_bpp"], epoch)
        writer.add_scalar('val/right_bpp', val_loss["right_bpp"], epoch)
        writer.add_scalar(f'val/{left_db_name}', val_loss[left_db_name], epoch)
        writer.add_scalar(f'val/{right_db_name}', val_loss[right_db_name], epoch)

        # wandb.log({ 
        #         "test": {"loss": val_loss["loss"], metric_name: val_loss[metric_name], "bpp_loss": val_loss["bpp_loss"],
        #         "aux_loss": val_loss["aux_loss"], metric_dB_name: val_loss[metric_dB_name], "left_bpp": val_loss["left_bpp"], "right_bpp": val_loss["right_bpp"],
        #         left_db_name:val_loss[left_db_name], right_db_name: val_loss[right_db_name],}
        #     })
        loss = val_loss["loss"]
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if args.sche_name == "steplr":
            lr_scheduler.step()
        else:
            lr_scheduler.step(loss)
  
        if args.save:
            # if epoch in [200, 250, 300, 350]:
            #     filename = f"{epoch}_ckpt.pth.tar"
            #     is_best = False
            # else:
            filename = "ckpt.pth.tar"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                },
                is_best, log_dir, filename=filename
            )

if __name__ == "__main__":
    main(sys.argv[1:])