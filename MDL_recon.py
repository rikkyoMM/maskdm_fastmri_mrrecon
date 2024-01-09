import os
import argparse
import numpy as np
import pandas as pd

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import math
import torch
from torchvision.utils import save_image

from gaussian_ddpm import GaussianDiffusion
from models import MaskedUViT, MaskedDWTUViT
from utils.config import parse_yml, combine

from PIL import Image
from torchvision import transforms

import random

import matplotlib.pyplot as plt

from scipy.fft import fft2, ifft2, fftshift
import pdb  # デバッガのインポート

import torch.nn.functional as F  # この行を追加


def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--overwrite", default="command-line", type=str, help="overwrite config/command-line arguments when conflicts occur")
    parser.add_argument("--bs", type=int, help="batch size used in evaluation")
    parser.add_argument("--seed", type=int, default=1232, help="seed for random number generator")

    parser.add_argument("--total_samples", type=int, default=3000, help="samples to generate")
    parser.add_argument("--sampling_steps", type=int, default=500, help="DDIM sampling steps")
    parser.add_argument("--ddim_sampling_eta", type=float, default=1.0, help="DDIM sampling eta coefficient")
    parser.add_argument("--sampler", type=str, default="ddim", help="sampler to use [ddim]")
    parser.add_argument("--output", nargs="+", help="list of output path to save images")
    parser.add_argument("--ckpt", nargs="+",  help="list of path to model checkpoint")

    parser.add_argument("--guidance_weight", type=float, default=-1,  help="guidance_weight, should be >=0, -1 means no guidance")
    parser.add_argument("--use_corrector", action="store_true",  help="use langevin corrector or not")

    return parser.parse_args()

# ...その他の関数定義...
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def build_model(args):
    
    name = args.network["name"]
    if name == "maskdm":
        return MaskedUViT(**args.network)
    elif name =="maskdwt":
        return MaskedDWTUViT(**args.network)
    else:
        raise NotImplementedError(f"Unsupported network type: {name}")


def imgtrans(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # バッチ次元を追加


# 画像を保存する関数
def saveimg_(img, filename):
    transform = transforms.ToPILImage()
    image = transform(img.squeeze(0))  # バッチ次元を削除
    # image.save(filename)
    plt.imsave(filename, image, cmap='gray')

def saveimg(img, filename):
    # 正規化: 最小値を0、最大値を1に
    img = img - img.min()
    img = img / img.max()

    # 0から255の範囲にスケーリング
    img_np = img.cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    # numpy配列からPILイメージに変換
    # バッチ次元とチャネル次元を除去 (バッチサイズが1の場合)
    image_np = img_np[0, 0, ...]  # チャネル次元を除去
    image = Image.fromarray(image_np, 'L')  # グレースケールイメージとして読み込み
    image.save(filename)

# ここに実験プログラムを記述
@torch.no_grad()
def evaluation():
    accelerator = Accelerator(
        split_batches=True,
        mixed_precision='fp16',
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    accelerator.native_amp = True

    setup_for_distributed(accelerator.is_main_process)

    args = parse_terminal_args()
    config = parse_yml(args.config)
    if config is not None:
        args = combine(args, config)


    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = accelerator.device
    model = build_model(args)

# 画像のロード
    # 画像の準備（例としてランダムな画像を生成）


    print(f"normalization: {getattr(args.dataset, 'NORMALIZATION', True)}")
    diffusion_model = GaussianDiffusion(
        model,
        image_size=args.network["img_size"],
        timesteps=500,
        sampling_timesteps=getattr(args, "sampling_steps", 250),
        ddim_sampling_eta=getattr(args, "ddim_sampling_eta", 1),
        clip_denoised=getattr(args, "clip_denoised", True),
        clip_max=getattr(args, "clip_max", 1),
        clip_min=getattr(args, "clip_min", -1),
        normalization=getattr(args.dataset, "NORMALIZATION", True),
        loss_type='l2',
        shift=getattr(args, "shift", False),
        channels=args.network.get("in_chans", 3),
    )
    diffusion_model.to(device)
    diffusion_model.eval()

    # モデルの重みをロード
    for ckpt in args.ckpt:
        if ckpt != "":
            raw_state_dict = torch.load(ckpt, map_location="cpu")["ema"]
            state_dict = {k[10:]: v for k, v in raw_state_dict.items() if k.startswith("ema_model")}
            diffusion_model.load_state_dict(state_dict, strict=False)
        else:
            print("Empty checkpoint path provided")



    # ここからノイズ化とデノイズ化===========================================================
        # 画像の読み込みとノイズ化
        img_path = "/path/data/fastmri/val2_under/00026.jpg"
        # 元画像のロード
        img = Image.open(img_path).convert("L")
        img = img.resize((128, 128), Image.ANTIALIAS)
        # print(np.array(img).max())
        # 元画像をreconフォルダに保存
        # origin_image_path = "/path/code/reconimage/MDL_slightnoise/1_original.jpg"
        plt.imsave('/path/code/reconimage/MDL_slightnoise/1_original.jpg', img, cmap='gray')

        # モデルに通すように画像を変換
        x_start = imgtrans(img).to(device)







        # 高周波画像をノイズ化
        # t = torch.tensor([int(diffusion_model.num_timesteps * tslr)], device=device, dtype=torch.long)

        t = torch.tensor([int((diffusion_model.num_timesteps-1) * 1)], device=device)
        # print(t.numpy())
        print("===x_start====")
        print(x_start.max())
        print(x_start.min())
        noisy_image = diffusion_model.q_sample(x_start, t)
        print("===noizeimage====")
        print(noisy_image.max())
        print(noisy_image.min())
        # # 高周波ノイズ画像を保存
        noisy_image_image_path = "/path/code/reconimage/MDL_slightnoise/4_noize.jpg"
        saveimg(noisy_image, noisy_image_image_path)

        # # noisy_image_shape = noisy_image.shape  # noisy_imageの形状を取得
        # # random_noise = torch.randn(noisy_image_shape, device=device)
        # # randomnoise_image_path = "/path/data/mri_sample/after/randomnoise.jpg"
        # # save_image(random_noise, randomnoise_image_path)

        # # 高周波ノイズ画像をデノイズ化
        # # denoised_image = denoise_image(diffusion_model, noisy_image, t)
        # # denoised_image = denoise_image(diffusion_model, random_noise, t)
        noisy_image = noisy_image.to(torch.float32)
        denoised_image = diffusion_model.ddim_sample_gpt(noisy_image, 500)

        # # デノイズ化された高周波画像を保存
        denoise_image_path = "/path/code/reconimage/MDL_slightnoise/5_denoizehigh.jpg"
        saveimg(denoised_image, denoise_image_path)
    

        # デノイズ高周波画像と低周波画像を結合した画像を保存




        # ======検証プログラム いずれやる RGBのマスクモデルとのRMSEの比較ができれば============

        # 高解像度画像のロード

        # デノイズ高周波画像と低周波画像を結合した画像と高解像度画像のMSEを計算






if __name__ == "__main__":
    evaluation()