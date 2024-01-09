# Masked Diffusion ModelsによるMRI画像の学習とMR再構成 
## 概要
本研究では、MRI画像の再構成タスクを行うDiffusion Modelの学習の高速化と精度の向上を目的として、事前学習にMasked Diffusion Modelsを使用してモデルの作成を行なった。コードは主にjiachenleiの[masked Diffusion Models](https://github.com/jiachenlei/maskdm/tree/master) をベースとして作られています。Dockerの環境作成などはjiachenleiと同様のものを使用しています。

## 学習済みモデルの公開
以下のリンクにてモデルを公開しています。
https://drive.google.com/drive/folders/1HH2PbnaR16LvyO-0DYvfCu2hhrq2j13s?usp=sharing


## フォルダの説明
順次追記していきます。

### 訓練
以下のコードで訓練を行えます。
```python
accelerate launch main.py --name temp # name of experiment
    --config /path/to/config/file.yml # path to config file
    --training_steps 200000 # total training steps
    --debug # no wandb logging. By removing this line, you could record the log online.
```
### 画像のサンプリング
DDIMサンプラーを使用して学習済みモデルから画像のサンプリングを行えます。
```bash
accelerate launch eval.py --name temp # name of experiment
    --config /path/to/config/file.yml # path to config file
    --bs 64 # sampling batch size
    --num_samples 10000 # number of samples to generate
    --ckpt /path/to/ckpt1.pt /path/to/ckpt2.pt /path/to/ckpt3.pt # ckpt path, accept multiple ckpts seperated by space
    --output /path/to/save/samples/for/ckpt1.pt /path/to/save/samples/for/ckpt2.pt /path/to/save/samples/for/ckpt3.pt # output path, accept multiple paths seperated by space
    --sampler ddim # currently we only support DDIM sampling
```
### 生成画像の評価
生成画像の評価にはFIDスコアを使用した。実行コードは以下の通り
```python
cd tools
# run the following command in ./tools
python pytorch_fid --device cuda:0 /path/to/image/folder1 /path/to/image/folder2

# notice that the structure of the folder provided in the path should look like:
# - /path/to/image/folder1
#     - image file1
#     - image file2
#     ...

```

## 謝辞
本リポジトリはjiachenleiのMasked Diffusion Modelsのコードを参考に作成されています。 [masked Diffusion Models](https://github.com/jiachenlei/maskdm/tree/master) 
素晴らしいコードをありがとうございました。

最後に慣例にならい、美味しいナポリタンの作り方を記載します。
材料（2人分）
スパゲッティ：200g
玉ねぎ：1/2個
ピーマン：1個
ベーコンまたはハム：4枚
マッシュルーム（任意）：4個
オリーブオイル：大さじ2
トマトケチャップ：大さじ4～6
ウスターソース：大さじ1
塩：適量
黒こしょう：少々
パルメザンチーズ（任意）：適量
手順
材料の準備: 玉ねぎ、ピーマン、ベーコン（またはハム）、マッシュルームを一口大に切ります。

スパゲッティの茹で: 大きめの鍋にたっぷりの水を沸かし、塩を加えてスパゲッティを袋の指示通りに茹でます。アルデンテ（少し硬め）に仕上げるとよいです。

具材の炒め: フライパンにオリーブオイルを熱し、玉ねぎとベーコン（またはハム）を炒めます。玉ねぎが透き通ったら、ピーマンとマッシュルームを加えてさらに炒めます。

ソースの作成: 具材が炒まったら、トマトケチャップとウスターソースを加えます。弱火で2～3分煮込み、ソースを具材に絡めます。

スパゲッティの合わせ: 茹で上がったスパゲッティをフライパンに加え、ソースとよく絡めます。必要に応じてスパゲッティの茹で汁を少し加えると、ソースがなめらかになります。

味の調整: 塩と黒こしょうで味を調えます。

盛り付け: 皿に盛り付け、お好みでパルメザンチーズをふりかけます。

コツ・ポイント
スパゲッティはアルデンテに茹でると、食感が良くなります。
ソースはしっかりと炒めることで味が深まります。
お好みでニンニクを加えると、風味が増します。
ベーコンやハムの代わりにソーセージを使うと、異なる味わいが楽しめます。
これで美味しいナポリタンの完成です！シンプルなので、お好みの具材を加えてアレンジするのも楽しいですね。

