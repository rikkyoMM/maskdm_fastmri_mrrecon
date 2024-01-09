from PIL import Image
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


def load_img(img_path):
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

    return F.to_tensor(img_high)


# def load_img(img_path):
#     # 画像を読み込み、グレースケールに変換
#     img = Image.open(img_path).convert("L")
#     img = img.resize((128, 128), Image.ANTIALIAS)

#     # NumPy配列に変換
#     img_np = np.array(img)

#     # フーリエ変換
#     f_img = fft2(img_np)
#     f_img_shifted = fftshift(f_img)

#     # マスクの作成（中心の縦横10マスを除く）
#     mask = np.ones((128, 128), dtype=np.uint8)
#     center_x, center_y = 128 // 2, 128 // 2
#     mask[center_x-5:center_x+5, center_y-5:center_y+5] = 0

#     # マスクを適用して高周波成分のみを取り出す
#     f_img_shifted = f_img_shifted * mask

#     # 逆フーリエ変換
#     f_img_shifted = ifft2(fftshift(f_img_shifted))
#     img_high = np.abs(f_img_shifted)

#     # PIL Imageに変換
#     img_high = Image.fromarray(img_high)

#     return F.to_tensor(img_high)




def load_img_(img_path):
    # 画像を読み込み、グレースケールに変換
    img = Image.open(img_path).convert("L")
    img = img.resize((128, 128), Image.ANTIALIAS)

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

    # 逆フーリエ変換
    f_img_shifted_low = ifft2(fftshift(f_img_shifted_low_))
    img_low = np.abs(f_img_shifted_low)
    # PIL Imageに変換
    img_low = Image.fromarray(img_low)
    plt.imsave('/path/code/testimg/image_low_.jpg', img_low, cmap='gray')

    f_img_shifted_high = ifft2(fftshift(f_img_shifted_high_))
    img_high = np.abs(f_img_shifted_high)
    # PIL Imageに変換
    img_high = Image.fromarray(img_high)
    plt.imsave('/path/code/testimg/image_high_.jpg', img_high, cmap='gray')

    # 画像を再び合成
    # 逆フーリエ変換
    f_img_shifted = ifft2(fftshift((f_img_shifted_low_ + f_img_shifted_high_)))
    img= np.abs(f_img_shifted)
    # PIL Imageに変換
    img = Image.fromarray(img)
    plt.imsave('/path/code/testimg/image_remix_.jpg', img, cmap='gray')

    return F.to_tensor(img_high)


def main():
    img_path = '/path/data/fastmri/train2/00034.jpg'
    img_tensor = load_img(img_path)
    print(img_tensor)
    # 結果を表示
    plt.imsave('/path/code/testimg/sample.jpg', img_tensor.numpy().squeeze(), cmap='gray')


if __name__ == '__main__':
    main()