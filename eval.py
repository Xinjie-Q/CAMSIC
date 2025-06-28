import argparse
import json
import math
import sys
import os
import time
import struct

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm
import compressai
from models.camsic import *
from lib.utils import CropCityscapesArtefacts, MinimalCrop
import torch.nn as nn
from compressai.zoo import *
from models.elic import ELIC

def collect_images(data_name:str, rootpath: str):
    if data_name == 'cityscapes':
        left_image_list, right_image_list = [], []
        path = Path(rootpath)
        for left_image_path in path.glob(f'leftImg8bit/test/*/*.png'):
            left_image_list.append(str(left_image_path))
            right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))

    elif data_name == 'instereo2k':
        path = Path(rootpath)
        path = path / "test"   
        folders = [f for f in path.iterdir() if f.is_dir()]
        left_image_list = [f / 'left.png' for f in folders]
        right_image_list = [f / 'right.png' for f in folders] #[1, 3, 860, 1080], [1, 3, 896, 1152]


    return [left_image_list, right_image_list]


def aggregate_results(filepaths: List[Path]) -> Dict[str, Any]:
    metrics = defaultdict(list)

    # sum
    for f in filepaths:
        with f.open("r") as fd:
            data = json.load(fd)
        for k, v in data["results"].items():
            metrics[k].append(v)

    # normalize
    agg = {k: np.mean(v) for k, v in metrics.items()}
    return agg



def pad(x: Tensor, p: int = 2 ** (4 + 1), pad_mode="center_zeros") -> Tuple[Tensor, Tuple[int, ...]]:
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    if "center" in pad_mode:
        padding_left = (new_w - w) // 2
        padding_top = (new_h - h) // 2
    elif "edge" in pad_mode:
        padding_left = padding_top = 0
    padding_right = new_w - w - padding_left
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    if "zeros" in pad_mode:
        x = F.pad(x, padding, mode="constant", value=0)
    elif "reflect" in pad_mode:
        x = F.pad(x, padding, mode="reflect")
    elif "replicate" in pad_mode:
        x = F.pad(x, padding, mode="replicate")
    elif "circular" in pad_mode:
        x = F.pad(x, padding, mode="circular")
    return x, padding


def crop(x: Tensor, padding: Tuple[int, ...]) -> Tensor:
    return F.pad(x, tuple(-p for p in padding))


def compute_metrics_for_frame(
    org_frame: Tensor,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,):
    
    psnr_float = -10 * torch.log10(F.mse_loss(org_frame, rec_frame))
    ms_ssim_float = ms_ssim(org_frame, rec_frame, data_range=1.0)
    org_frame = (org_frame * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_frame - rec_frame).pow(2).mean()
    psnr_rgb = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)
    ms_ssim_rgb = ms_ssim(org_frame, rec_frame, data_range=max_val)

    # psnr_float = -10 * torch.log10(F.mse_loss(org_frame, rec_frame))
    # ms_ssim_float = ms_ssim(org_frame, rec_frame, data_range=1.0)
    return psnr_rgb, ms_ssim_rgb, psnr_float, ms_ssim_float


def compute_bpp(likelihoods, num_pixels):
    bpp = sum(
        (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
        for likelihood in likelihoods.values()
    )
    return bpp


def read_image(crop_transform, filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    if crop_transform is not None:
        img = crop_transform(img)
    return transforms.ToTensor()(img)


@torch.no_grad()
def eval_model(IFrameCompressor:nn.Module, left_filepaths: Path, right_filepaths: Path, **args: Any) -> Dict[str, Any]:
    device = next(IFrameCompressor.parameters()).device
    num_frames = len(left_filepaths) 
    max_val = 2**8 - 1
    results = defaultdict(list)
    if args["crop"]:
        crop_transform = CropCityscapesArtefacts() if args["data_name"] == "cityscapes" else MinimalCrop(min_div=64)
    else:
        crop_transform = None

    results = defaultdict(list)
    pad_mode = args["pad_mode"]

    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):  
            x_left = read_image(crop_transform, left_filepaths[i]).unsqueeze(0).to(device)
            num_pixels = x_left.size(2) * x_left.size(3)
            x_right = read_image(crop_transform, right_filepaths[i]).unsqueeze(0).to(device)
            # left_height, left_width = x_left.shape[2:]
            # right_height, right_width = x_right.shape[2:]
            x_left, padding = pad(x_left, p=2**(4+2), pad_mode=pad_mode)
            x_right, padding = pad(x_right, p=2**(4+2), pad_mode=pad_mode)

            if args["single"]:
                start = time.time()
                out_enc_left = IFrameCompressor.compress(x_left)
                out_enc_right = IFrameCompressor.compress(x_right)
                enc_time = time.time() - start

                start = time.time()
                out_dec_left = IFrameCompressor.decompress(out_enc_left)
                out_dec_right = IFrameCompressor.decompress(out_enc_right)
                dec_time = time.time() - start
                x_left_rec, x_right_rec = out_dec_left["x_hat"], out_dec_right["x_hat"]

            else:
                start = time.time()
                out_enc = IFrameCompressor.compress([x_left, x_right])
                enc_time = time.time() - start

                start = time.time()
                out_dec = IFrameCompressor.decompress(out_enc)
                dec_time = time.time() - start

                x_left_rec, x_right_rec = out_dec["x_hat"][0], out_dec["x_hat"][1]

            x_left_rec = crop(x_left_rec.clamp(0, 1), padding)
            x_right_rec = crop(x_right_rec.clamp(0, 1), padding)

            metrics = {}
            metrics["left-psnr-rgb"], metrics["left-ms-ssim-rgb"], metrics["left-psnr-float"], metrics["left-ms-ssim-float"] = compute_metrics_for_frame(
                crop(x_left, padding), x_left_rec, device, max_val)
            metrics["right-psnr-rgb"], metrics["right-ms-ssim-rgb"], metrics["right-psnr-float"], metrics["right-ms-ssim-float"] = compute_metrics_for_frame(
                crop(x_right, padding), x_right_rec, device, max_val)
            
            metrics["psnr-rgb"] = (metrics["left-psnr-rgb"]+metrics["right-psnr-rgb"])/2
            metrics["ms-ssim-rgb"] = (metrics["left-ms-ssim-rgb"]+metrics["right-ms-ssim-rgb"])/2            
            metrics["psnr-float"] = (metrics["left-psnr-float"]+metrics["right-psnr-float"])/2
            metrics["ms-ssim-float"] = (metrics["left-ms-ssim-float"]+metrics["right-ms-ssim-float"])/2

            try:
                bpp = 0
                for out_bitstream in out_enc:
                    for s in out_bitstream["strings"]:
                        bpp += len(s[0]) * 8.0 / num_pixels
            except:
                bpp = 0
            metrics["bpp"] = torch.tensor(bpp)/2

            enc_time = torch.tensor(enc_time)
            dec_time = torch.tensor(dec_time)
            metrics["enc_time"] = enc_time
            metrics["enc_average_time"] = enc_time/2
            
            metrics["dec_time"] = dec_time
            metrics["dec_average_time"] = dec_time/2

            print(metrics)
            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }

    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results

@torch.no_grad()
def eval_model_entropy_estimation(IFrameCompressor:nn.Module, left_filepaths: Path, right_filepaths: Path, **args: Any) -> Dict[str, Any]:
    device = next(IFrameCompressor.parameters()).device
    num_frames = len(left_filepaths) 
    max_val = 2**8 - 1
    results = defaultdict(list)
    if args["crop"]:
        crop_transform = CropCityscapesArtefacts() if args["data_name"] == "cityscapes" else MinimalCrop(min_div=64)
    else:
        crop_transform = None

    pad_mode = args["pad_mode"]
    # print("num frames:", num_frames)
    # input()

    with tqdm(total=num_frames) as pbar: #97: 0-96
        for i in range(num_frames):

            x_left = read_image(crop_transform, left_filepaths[i]).unsqueeze(0).to(device)
            num_pixels = x_left.size(2) * x_left.size(3)
            x_right = read_image(crop_transform, right_filepaths[i]).unsqueeze(0).to(device)
            # left_height, left_width = x_left.shape[2:]
            # right_height, right_width = x_right.shape[2:]
            x_left, padding = pad(x_left, p=2**(4+2), pad_mode=pad_mode)
            x_right, padding = pad(x_right, p=2**(4+2), pad_mode=pad_mode)
            if args["single"]:
                out_l = IFrameCompressor(x_left)
                out_r = IFrameCompressor(x_right)
                x_left_rec, x_right_rec = out_l["x_hat"], out_r["x_hat"]
                left_likelihoods, right_likelihoods = out_l["likelihoods"], out_r["likelihoods"]
            else:
                out = IFrameCompressor([x_left, x_right])
                x_left_rec, x_right_rec = out["x_hat"][0], out["x_hat"][1]
                left_likelihoods, right_likelihoods = out["likelihoods"][0], out["likelihoods"][1]

               
            x_left_rec = crop(x_left_rec.clamp(0, 1), padding)
            x_right_rec = crop(x_right_rec.clamp(0, 1), padding)

            metrics = {}
            metrics["left-psnr-rgb"], metrics["left-ms-ssim-rgb"], metrics["left-psnr-float"], metrics["left-ms-ssim-float"] = compute_metrics_for_frame(
                crop(x_left, padding), x_left_rec, device, max_val)
            metrics["right-psnr-rgb"], metrics["right-ms-ssim-rgb"], metrics["right-psnr-float"], metrics["right-ms-ssim-float"] = compute_metrics_for_frame(
                crop(x_right, padding), x_right_rec, device, max_val)
            
            metrics["psnr-rgb"] = (metrics["left-psnr-rgb"]+metrics["right-psnr-rgb"])/2
            metrics["ms-ssim-rgb"] = (metrics["left-ms-ssim-rgb"]+metrics["right-ms-ssim-rgb"])/2            
            metrics["psnr-float"] = (metrics["left-psnr-float"]+metrics["right-psnr-float"])/2
            metrics["ms-ssim-float"] = (metrics["left-ms-ssim-float"]+metrics["right-ms-ssim-float"])/2


            metrics["left_bpp"] = compute_bpp(left_likelihoods, num_pixels)
            metrics["right_bpp"] = compute_bpp(right_likelihoods, num_pixels)
            if len(out['likelihoods']) > 2 and not args["single"]:
                metrics["additional_bpp"] = compute_bpp(out['likelihoods'][2], num_pixels)
                metrics['left_bpp'] = metrics['left_bpp'] + metrics["additional_bpp"]/2
                metrics["right_bpp"] = metrics["right_bpp"] + metrics["additional_bpp"]/2   

            metrics["bpp"] = (metrics["left_bpp"] + metrics["right_bpp"])/2
            
            if args["save_image"]:
                filepath = Path(args["output"])
                tensor_to_PIL(x_left_rec, filepath, left=True)
                tensor_to_PIL(x_right_rec, filepath, left=False)

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }
    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results

def tensor_to_PIL(tensor, filepath, left=True):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    file = filepath / 'left.png' if left else filepath / 'right.png'
    image.save(str(file))

def run_inference(
    filepaths,
    IFrameCompressor: nn.Module, 
    outputdir: Path,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any):

    left_filepath, right_filepath = filepaths[0], filepaths[1]
    #sequence_metrics_path = Path(outputdir) / f"{trained_net}.json"

    #if force:
    #    sequence_metrics_path.unlink(missing_ok=True)

    with amp.autocast(enabled=args["half"]):
        with torch.no_grad():
            if entropy_estimation:
                metrics = eval_model_entropy_estimation(IFrameCompressor, left_filepath, right_filepath, **args)
            else:
                metrics = eval_model(IFrameCompressor, left_filepath, right_filepath, **args)
    return metrics

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stereo image compression network evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="sequences directory")
    parser.add_argument("--data-name", type=str, required=True, help="sequences directory")
    parser.add_argument("--output", type=str, help="output directory")
    parser.add_argument(
        "-im",
        "--IFrameModel",
        default="LDMIC",
        help="Model architecture (default: %(default)s)",
    )

    parser.add_argument("-iq", "--IFrame_quality", type=int, default=4, help='Model quality')
    parser.add_argument("--denominator", type=int, default=8, help='mask regressive number')
    parser.add_argument("--net_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--i_model_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--crop", action="store_true", help="use crop")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--half", action="store_true", help="use AMP")
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--keep_binaries",
        action="store_true",
        help="keep bitstream files in output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms-ssim")
    parser.add_argument("--cpu_num", type=int, default=4)
    parser.add_argument("--single", action="store_true", help="use single image model")
    parser.add_argument("--num_heads", type=int, default=12, help="Set random seed for reproducibility")
    parser.add_argument("--depths", type=int, default=8, help="Set random seed for reproducibility")
    parser.add_argument("--win_size", type=int, default=8, help="Set random seed for reproducibility")
    parser.add_argument("--pad_mode", type=str, default="edge_replicate", help="metric: mse, ms-ssim")
    parser.add_argument("--save_image", action="store_true", help="use single image model")
    return parser


def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )
    filepaths = collect_images(args.data_name, args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        cpu_num = args.cpu_num # 这里设置成你想运行的CPU个数
        os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        torch.set_num_threads(cpu_num)
 
    if args.single:
        if args.IFrameModel == "ELIC":
            checkpoint = torch.load(args.net_path, map_location=device)
            IFrameCompressor = ELIC.from_state_dict(checkpoint)
            IFrameCompressor = IFrameCompressor.to(device)
            IFrameCompressor.update(force=True)
            # if args.net_path:
            #     print("Loading model:", args.net_path)
            #     checkpoint = torch.load(args.net_path, map_location=device)
            #     model_architectures[architecture].from_state_dict(state_dict)
            #     IFrameCompressor.load_state_dict(checkpoint)
    else:
        if args.IFrameModel in ["CAMSIC"]:
            IFrameCompressor = CAMSIC(arch_name="ELIC", num_channels=320, context_len=1, embed_dim=768, depths=[args.depths], 
                num_heads=args.num_heads, window_size=args.win_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                drop_path_rate=0., norm_layer=nn.LayerNorm, scratch=False)

      
        IFrameCompressor = IFrameCompressor.to(device)
        if args.net_path:
            print("Loading model:", args.net_path)
            checkpoint = torch.load(args.net_path, map_location=device)
            if "state_dict" in checkpoint:
                IFrameCompressor.load_state_dict(checkpoint["state_dict"])
            else:
                IFrameCompressor.load_state_dict(checkpoint)
    IFrameCompressor.eval()
    # create output directory
    outputdir = args.output
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    results = defaultdict(list)
    args_dict = vars(args)

    trained_net = f"{args.IFrameModel}-{args.metric}-{description}"
    metrics = run_inference(filepaths, IFrameCompressor, outputdir, trained_net=trained_net, description=description, **args_dict)
    for k, v in metrics.items():
        results[k].append(v)

    output = {
        "name": f"{args.IFrameModel}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }

    with (Path(f"{outputdir}/{args.IFrameModel}-{args.metric}-{description}.json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])