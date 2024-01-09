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
    parser.add_argument("--sampling_steps", type=int, default=250, help="DDIM sampling steps")
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
        timesteps=1000,
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



        # ここで再度cosine_beta_scheduleを実行してalphas_cumprodを再設定する。


        # def cosine_beta_schedule(timesteps, s = 0.008, shift=False, d=64):
        #     """
        #     Create a schedule where alpha_cumprod is constant (all ones) for all timesteps.
        #     """
        #     alphas_cumprod = torch.ones(timesteps + 1, dtype=torch.float64)
        #     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        #     return torch.clip(betas, 0, 1)



        # def cosine_beta_schedule(timesteps, s=0.008, max_noise_level=0, shift=False, d=64):
        #     """
        #     Modified cosine schedule to limit maximum noise level.
        #     """

        #     steps = timesteps + 1
        #     x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        #     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2 / 2 + 0.5

        #     # Ensure alphas_cumprod does not go below the desired threshold (70% data retention)
        #     min_alphas_cumprod = 1 - max_noise_level
        #     alphas_cumprod = torch.clamp(alphas_cumprod, min=min_alphas_cumprod)

        #     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        #     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        #     # print("gpt cos schedule")
        #     # betas_alphas_df = pd.DataFrame({
        #     #     'betas': betas.numpy(),
        #     #     'alphas_cumprod': alphas_cumprod[1:].numpy()  # Skip the first element to match the length of betas
        #     # })
        #     # betas_alphas_df.to_csv("./cosschedule_1.csv", index=False)
        #     return torch.clip(betas, 0, 0.999)


        def cosine_beta_schedule(timesteps, s = 0.008, shift=False, d=64):
            """
            cosine schedule
            as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
            """
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2 / 2 + 0.

            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            print("0.5 cos schdule!")
            return torch.clip(betas, 0, 0.5)
        # ここで再度cosine_beta_scheduleを実行してalphas_cumprodを再設定する
        timesteps = diffusion_model.num_timesteps
        new_betas = cosine_beta_schedule(timesteps)
        new_alphas = 1. - new_betas
        new_alphas_cumprod = torch.cumprod(new_alphas, dim=0)
        new_alphas_cumprod_prev = F.pad(new_alphas_cumprod[:-1], (1, 0), value = 1.)

        # diffusion_model の alphas_cumprod を更新
        diffusion_model.alphas_cumprod = new_alphas_cumprod
        diffusion_model.alphas_cumprod_prev = new_alphas_cumprod_prev

        # 新しいalphas_cumprodを計算
        new_alphas_cumprod = diffusion_model.alphas_cumprod

        # 必要な値の再計算
        new_sqrt_alphas_cumprod = torch.sqrt(new_alphas_cumprod)
        new_sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - new_alphas_cumprod)

        # GaussianDiffusionオブジェクトのバッファを更新
        diffusion_model.register_buffer('alphas_cumprod', new_alphas_cumprod)
        diffusion_model.register_buffer('sqrt_alphas_cumprod', new_sqrt_alphas_cumprod)
        diffusion_model.register_buffer('sqrt_one_minus_alphas_cumprod', new_sqrt_one_minus_alphas_cumprod)

        # 既存のコードに続けて追加
        new_alphas_cumprod = new_alphas_cumprod.to(device)
        new_alphas_cumprod_prev = new_alphas_cumprod_prev.to(device)
        new_sqrt_alphas_cumprod = new_sqrt_alphas_cumprod.to(device)
        new_sqrt_one_minus_alphas_cumprod = new_sqrt_one_minus_alphas_cumprod.to(device)

        # これらをモデルの対応する属性に割り当てる
        diffusion_model.alphas_cumprod = new_alphas_cumprod
        diffusion_model.alphas_cumprod_prev = new_alphas_cumprod_prev
        diffusion_model.sqrt_alphas_cumprod = new_sqrt_alphas_cumprod
        diffusion_model.sqrt_one_minus_alphas_cumprod = new_sqrt_one_minus_alphas_cumprod

        print("diffusion_model.alpha_cumprod")
        print(diffusion_model.alphas_cumprod)


        # ...実験コード...
        # 高解像画像の読み込みとフーリエ分解
        img_path = "/path/data/fastmri/val2/00026.jpg"
        # 元画像のロード
        img = Image.open(img_path).convert("L")
        img = img.resize((128, 128), Image.ANTIALIAS)
        # 元画像をreconフォルダに保存
        plt.imsave('/path/code/reconimage/test/1_original_full.jpg', img, cmap='gray')
        plt.imsave('/path/code/reconimage/test/6_mix_teacher.jpg', img, cmap='gray')
        # 高周波と低周波に分離
        # NumPy配列に変換
        img_np = np.array(img)
        # フーリエ変換
        f_img = fft2(img_np)
        f_img_shifted = fftshift(f_img)
        # マスクの作成（中心の縦横10マスを除く）
        mask = np.ones((128, 128), dtype=np.uint8)
        center_x, center_y = 128 // 2, 128 // 2
        mask[center_x-5:center_x+5, center_y-5:center_y+5] = 0
        # マスクを適用して高周波成分のみを取り出す
        f_img_shifted_low_ = f_img_shifted * (np.ones((128, 128), dtype=np.uint8) - mask)
        f_img_shifted_high_ = f_img_shifted * mask
        # 高周波画像を保存
        # 逆フーリエ変換
        f_img_shifted_high = ifft2(fftshift(f_img_shifted_high_))
        img_high = np.abs(f_img_shifted_high)
        # モデルに入力用に0-255に正規化
        img_high = img_high - img_high.min()  # 最小値を0に
        img_high = img_high / img_high.max()  # 最大値を1に
        img_high = (img_high * 255).astype(np.uint8)  # 0-255の範囲にスケーリング
        img_high = Image.fromarray(img_high)
        plt.imsave('/path/code/reconimage/test/3_high_full.jpg', img_high, cmap='gray')
        # 低周波画像を保存
        # 逆フーリエ変換
        f_img_shifted_low = ifft2(fftshift(f_img_shifted_low_))
        img_low = np.abs(f_img_shifted_low)
        # モデルに入力用に0-255に正規化
        img_low = img_low - img_low.min()  # 最小値を0に
        img_low = img_low / img_low.max()  # 最大値を1に
        img_low = (img_low * 255).astype(np.uint8)  # 0-255の範囲にスケーリング
        # PIL Imageに変換
        img_low = Image.fromarray(img_low)
        plt.imsave('/path/code/reconimage/test/2_low_full.jpg', img_low, cmap='gray')

    # ここからノイズ化とデノイズ化===========================================================
        # 画像の読み込みとノイズ化
        img_path = "/path/data/fastmri/val2_under/00026.jpg"
        # 元画像のロード
        img = Image.open(img_path).convert("L")
        img = img.resize((128, 128), Image.ANTIALIAS)
        print(np.array(img).max())
        # 元画像をreconフォルダに保存
        # origin_image_path = "/path/code/reconimage/test/1_original.jpg"
        plt.imsave('/path/code/reconimage/test/1_original.jpg', img, cmap='gray')


        # 高周波と低周波に分離
        # NumPy配列に変換
        img_np = np.array(img)
        # フーリエ変換
        f_img = fft2(img_np)
        f_img_shifted = fftshift(f_img)
        # マスクの作成（中心の縦横10マスを除く）
        mask = np.ones((128, 128), dtype=np.uint8)
        center_x, center_y = 128 // 2, 128 // 2
        mask[center_x-5:center_x+5, center_y-5:center_y+5] = 0
        # マスクを適用して高周波成分のみを取り出す
        f_img_shifted_low_ = f_img_shifted * (np.ones((128, 128), dtype=np.uint8) - mask)
        f_img_shifted_high_ = f_img_shifted * mask

        # 高周波画像を保存
        # 逆フーリエ変換
        f_img_shifted_high = ifft2(fftshift(f_img_shifted_high_))
        img_high = np.abs(f_img_shifted_high)
        # モデルに入力用に0-255に正規化
        img_high = img_high - img_high.min()  # 最小値を0に
        img_high = img_high / img_high.max()  # 最大値を1に
        img_high = (img_high * 255).astype(np.uint8)  # 0-255の範囲にスケーリング
        img_high = Image.fromarray(img_high)
        plt.imsave('/path/code/reconimage/test/3_high.jpg', img_high, cmap='gray')


        # 低周波画像を保存
        # 逆フーリエ変換
        f_img_shifted_low = ifft2(fftshift(f_img_shifted_low_))
        img_low = np.abs(f_img_shifted_low)
        # モデルに入力用に0-255に正規化
        img_low = img_low - img_low.min()  # 最小値を0に
        img_low = img_low / img_low.max()  # 最大値を1に
        img_low = (img_low * 255).astype(np.uint8)  # 0-255の範囲にスケーリング
        # PIL Imageに変換
        img_low = Image.fromarray(img_low)
        plt.imsave('/path/code/reconimage/test/2_low.jpg', img_low, cmap='gray')



        # モデルに通すように画像を変換
        x_start = imgtrans(img_high).to(device)







        # 高周波画像をノイズ化
        # t = torch.tensor([int(diffusion_model.num_timesteps * tslr)], device=device, dtype=torch.long)

        t = torch.tensor([int((diffusion_model.num_timesteps-1) * 0.1)], device=device)
        # print(t.numpy())
        print("===x_start====")
        print(x_start.max())
        print(x_start.min())
        noisy_image = diffusion_model.q_sample(x_start, t)
        print("===noizeimage====")
        print(noisy_image.max())
        print(noisy_image.min())
        # # 高周波ノイズ画像を保存
        noisy_image_image_path = "/path/code/reconimage/test/4_noize.jpg"
        saveimg(noisy_image, noisy_image_image_path)

        # # noisy_image_shape = noisy_image.shape  # noisy_imageの形状を取得
        # # random_noise = torch.randn(noisy_image_shape, device=device)
        # # randomnoise_image_path = "/path/data/mri_sample/after/randomnoise.jpg"
        # # save_image(random_noise, randomnoise_image_path)

        # # 高周波ノイズ画像をデノイズ化
        # # denoised_image = denoise_image(diffusion_model, noisy_image, t)
        # # denoised_image = denoise_image(diffusion_model, random_noise, t)
        noisy_image = noisy_image.to(torch.float32)
        denoised_image = diffusion_model.ddim_sample_gpt(noisy_image, 1000)

        # # デノイズ化された高周波画像を保存
        denoise_image_path = "/path/code/reconimage/test/5_denoizehigh.jpg"
        saveimg(denoised_image, denoise_image_path)

        # デノイズ高周波画像と低周波画像を結合
        # NumPy配列に変換
        denoised_image_np = denoised_image.squeeze(0).squeeze(0).cpu().numpy()
        # モデルに入力用に0-255に正規化
        denoised_image_np = denoised_image_np - denoised_image_np.min()  # 最小値を0に
        denoised_image_np = denoised_image_np / denoised_image_np.max()  # 最大値を1に
        denoised_image_np = (denoised_image_np * 255).astype(np.uint8)  # 0-255の範囲にスケーリング

        # フーリエ変換
        denoised_img = fft2(denoised_image_np)
        denoised_img_shifted_high_ = fftshift(denoised_img)
        # print(np.shape(denoised_img_shifted_high_))

        f_img_shifted_mix = ifft2(fftshift(f_img_shifted_low_ + denoised_img_shifted_high_))
        # f_img_shifted_mix = ifft2(fftshift(f_img_shifted_low_ + f_img_shifted_high_))
        img_mix = np.abs(f_img_shifted_mix)
        # モデルに入力用に0-255に正規化
        img_mix = img_mix - img_mix.min()  # 最小値を0に
        img_mix = img_mix / img_mix.max()  # 最大値を1に
        img_mix = (img_mix * 255).astype(np.uint8)  # 0-255の範囲にスケーリング
        # PIL Imageに変換
        img_mix = Image.fromarray(img_mix)
        plt.imsave('/path/code/reconimage/test/6_mix.jpg', img_mix, cmap='gray')        

        # デノイズ高周波画像と低周波画像を結合した画像を保存




        # ======検証プログラム いずれやる RGBのマスクモデルとのRMSEの比較ができれば============

        # 高解像度画像のロード

        # デノイズ高周波画像と低周波画像を結合した画像と高解像度画像のMSEを計算






if __name__ == "__main__":
    evaluation()