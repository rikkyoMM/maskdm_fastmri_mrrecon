import os
import argparse
import numpy as np

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import torch
from torchvision.utils import save_image

from gaussian_ddpm import GaussianDiffusion
from models import MaskedUViT, MaskedDWTUViT
from utils.config import parse_yml, combine

from PIL import Image
from torchvision import transforms


def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--overwrite", default="command-line", type=str, help="overwrite config/command-line arguments when conflicts occur")
    parser.add_argument("--bs", type=int, help="batch size used in evaluation")
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

# 画像保存に関する関数追加
# 画像を読み込み、適切な形式に変換する関数
def load_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # バッチ次元を追加

# 画像を保存する関数
def save_image(tensor, filename):
    transform = transforms.ToPILImage()
    image = transform(tensor.squeeze(0))  # バッチ次元を削除
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


        @torch.no_grad()
        def denoise_image(diffusion_model, noisy_image, t):
            """
            ノイズ化された画像をデノイズ化する関数。
            :param diffusion_model: 使用する拡散モデル。
            :param noisy_image: ノイズ化された画像。
            :param t: ノイズ化に使用されたタイムステップ。
            :return: デノイズ化された画像。
            """
            for i in reversed(range(400)):  # t.item() でスカラー値を取得
                print(i)
                noisy_image, _ = diffusion_model.p_sample(noisy_image, i)  # ここを修正
            return noisy_image


        # ...実験コード...
        # 画像の読み込みとノイズ化
        image_path = "/path/data/mri_sample/before/00082.jpg"
        x_start = load_image(image_path, diffusion_model.image_size).to(device)
        print(x_start.shape)
        # torch.Size([1, 3, 128, 128])
        
        # t = torch.tensor([int(diffusion_model.num_timesteps * tslr)], device=device, dtype=torch.long)
        t = torch.tensor([int((diffusion_model.num_timesteps-1) * 0.1)], device=device)
        noisy_image = diffusion_model.q_sample(x_start, t)

        # ノイズ化された画像を保存
        noisy_image_image_path = "/path/data/mri_sample/after/noiseimage.jpg"
        save_image(noisy_image, noisy_image_image_path)


        # ノイズ化された画像をデノイズ化
        denoised_image = denoise_image(diffusion_model, noisy_image, t)

        # デノイズ化された画像を保存
        denoised_image_path = "/path/data/mri_sample/after/denoisedimage.jpg"
        save_image(denoised_image, denoised_image_path)




if __name__ == "__main__":
    evaluation()