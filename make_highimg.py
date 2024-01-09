from PIL import Image
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import os
from pathlib import Path

class ImageProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

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

        return img_high

    def process_images(self):
        for img_file in os.listdir(self.input_dir):
            img_path = os.path.join(self.input_dir, img_file)
            if os.path.isfile(img_path):
                high_freq_img = self.load_img(img_path)
                output_path = os.path.join(self.output_dir, img_file)
                high_freq_img.save(output_path)

if __name__ == "__main__":
    input_dir = "/path/data/fastmri/train2"
    output_dir = "/path/data/fastmri/train_high"
    processor = ImageProcessor(input_dir, output_dir)
    processor.process_images()