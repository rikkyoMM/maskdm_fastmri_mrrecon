# Masked Diffusion ModelsによるMRI画像の学習とMR再構成 
##概要
本研究では、MRI画像の再構成タスクを行うDiffusion Modelの学習の高速化と精度の向上を目的として、事前学習にMasked Diffusion Modelsを使用してモデルの作成を行なった。コードは主にjiachenleiの[masked Diffusion Models](https://github.com/jiachenlei/maskdm/tree/master) をベースとして作られています。Dockerの環境作成などはjiachenleiと同様のものを使用しています。

##学習済みモデルの公開
以下のリンクにてモデルを公開しています。
https://drive.google.com/drive/folders/1HH2PbnaR16LvyO-0DYvfCu2hhrq2j13s?usp=sharing


## フォルダの説明
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



