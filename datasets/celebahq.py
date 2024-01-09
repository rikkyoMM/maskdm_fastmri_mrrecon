import os


from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt


import PIL
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

import pywt
import torch
from torch import nn
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as F

from .base import MaskdmDataset
import random


class CelebAHQ(MaskdmDataset):
    # img_couter追加、消しても良い
    # def __init__(self):
    #     super().__init__()  # 基底クラスの初期化を呼び出す
    #     self.load_img_counter = 0  # カウンターの初期化

    def init_dataset(self):
        self.use_dwt = getattr(self.cfg, "USE_DWT", False)
        if self.use_dwt:
            print("Using Discrete Wavelet transformation")

    def build_dataset(self):
        data_path = os.path.join(self.path)

        if self.verbose:
            tqdm.write("Reading CelebA-HQ...")

        imgs = os.listdir(data_path)

        self.package = [
            {
                "img_path": os.path.join(data_path, img),
            }
            for img in imgs
        ]


# # 画像読み込み 空間領域ver グレースケース版
#     def load_img(self, img_path):
#         img = Image.open(img_path).convert("L")  # グレースケールに変換
#         # img = Image.open(img_path).convert("RGB")  # RGBスケールに変換
#         img = img.resize((128, 128), Image.ANTIALIAS)

#         # imgはPIL→tensorで管理、NumPy配列にして中身を確認
#         # img_np = np.array(img)
#         # # 最大値、最小値、形状を出力
#         # print("Max value:", img_np.max())
#         # print("Min value:", img_np.min())
#         # print("Shape:", img_np.shape)
        
#         return F.to_tensor(img)


# 周波数領域ver 
    def load_img(self, img_path):
        # 画像を読み込み、グレースケールに変換
        img = Image.open(img_path).convert("L")
        img = img.resize((128, 128), Image.ANTIALIAS)

        # NumPy配列に変換
        img_np = np.array(img)

        # フーリエ変換
        f_img = fft2(img_np)
        f_img_shifted = fftshift(f_img)

        # マスクの作成
        mask = np.ones((128, 128), dtype=np.uint8)
        center_x, center_y = 128 // 2, 128 // 2
        mask[center_x-5:center_x+5, center_y-5:center_y+5] = 0

        # マスクを適用して高周波成分のみを取り出す
        f_img_shifted = f_img_shifted * mask

        # 逆フーリエ変換
        f_img_shifted = ifft2(fftshift(f_img_shifted))
        img_high = np.abs(f_img_shifted)

        # 画像を0から255の範囲にスケーリング
        img_high = img_high - img_high.min()  # 最小値を0に
        img_high = img_high / img_high.max()  # 最大値を1に
        img_high = (img_high * 255).astype(np.uint8)  # 0-255の範囲にスケーリング

        # PIL Imageに変換
        img_high = Image.fromarray(img_high)

        # load画像の確認用コード。すぐに切らないと画像がいっぱいになる
        # # 4桁の乱数を生成
        # random_number = random.randint(1000, 9999)
        # # 乱数をファイル名として使用して画像を保存
        # save_path = os.path.join('/path/code/testimg2/1', f"{random_number}.png")
        # img_high.save(save_path)

        return F.to_tensor(img_high)


    def __getitem__(self, index):

        # try until success
        while self.MAXIMUM_RETRY:
            try:
                pack = self.package[index]
                img_path = pack["img_path"]
                img = self.load_img(img_path) # torch.Tensor
                
                break
            except Exception as e:
                self.log_error(str(e) + ", image path:"+ img_path + "\n")
                index = random.randint(0, len(self.package)-1)

        # print("Image shape:", img.shape)
        if self.use_dwt:
            cA, (cH, cV, cD) = pywt.dwt2(img,"haar") # c, h, w
            img = torch.from_numpy(np.concatenate((cA, cH, cV, cD), axis=0)) # 12 dimensions

        m = self.generate_mask()
        m = torch.from_numpy(m)

        if self.task == "cond":
            raise NotImplementedError(f"Conditional training on CelebA-HQ is not supported")
        elif self.task == "uncond":
            m = torch.cat([torch.tensor([0]), m], dim=0) # skip time token
            batch = [img]
        else:
            raise ValueError("Unsupported task for vggface")

        batch.insert(1, m)

        return batch