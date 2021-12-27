"""
BN-Inception 特徴抽出　メインコード
"""
import os
import sys
import cv2
import torch
import shutil
import numpy as np
import torch.nn as nn
from PIL import Image
from natsort import natsorted
from torchvision import transforms

# 「pretrained-models」に含まれるbninceptionモデルを読み込むために使用
# 通常のfromでは読み込むことができない。ディレクトリ名に「.」が入っているため。
# そこでsys.pathを使用することで解決
sys.path.append("pretrained-models.pytorch/pretrainedmodels")
from models.bninception import bninception  # 一見、エラーしているように見えるがエラーではない


def BN_inception_main(movie_name, batch_size):
    basename = os.path.splitext(os.path.basename(movie_name))[0]  # MP4ファイルの名前だけを抽出
    save_path = './frame/' + basename

    os.makedirs("./frame", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # MP4ファイルを読み込んで各フレームを保存する
    cap = cv2.VideoCapture(movie_name)
    num = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(save_path + f'/{num}.jpg', frame)
            num += 1
        else:
            break

    files = os.listdir(save_path)  # 抽出された各フレームの名前を確認
    img_sort = natsorted(files)  # フレームの名前をソートする

    transform = transforms.Compose(
        [
            transforms.Resize(224),  # (224, 224) で切り抜く。
            transforms.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
            transforms.ToTensor(),  # テンソルにする。
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 標準化する。
        ]
    )

    img_list = []
    # 各フレーム変形
    for i in range(len(img_sort)):
        img = Image.open(save_path + '/' + str(i) + '.jpg')
        img = transform(img)  # 各フレームを上記で設定したようにtransformさせる
        img_list.append(img)

    # モデル読み込み
    model = bninception()
    if torch.cuda.is_available():  # CUDAが使用できるならば
        if torch.cuda.device_count() > 1:  # GPUが複数あるならば
            model = nn.DataParallel(model)
        model.to("cuda")

    model.eval()

    result = []
    length = len(img_list)

    flag = False
    if length < batch_size:
        flag = True

    with torch.no_grad():
        i = 0

        while True:
            length -= batch_size
            if length > 0:
                if torch.cuda.is_available():
                    input = torch.stack(img_list[i:i + batch_size]).to(
                        'cuda')  # 0 ~ 399 , 400 ~   when , batch_size=400
                else:
                    input = torch.stack(img_list[i:i + batch_size])
                output = model(input)
                output = torch.squeeze(output)
                result.append(output)

            else:
                if torch.cuda.is_available():
                    input = torch.stack(img_list[i:i + length + batch_size]).to('cuda')
                else:
                    input = torch.stack(img_list[i:i + length + batch_size])
                output = model(input)
                last_output = torch.squeeze(output)
                break
            i += batch_size

    if flag:
        outputs = last_output
    else:
        base = result[0]
        for i in range(len(result)):
            if i == 0:
                continue
            base = torch.cat((base, result[i]), 0)
        outputs = torch.cat((base, last_output), 0)

    np_output = outputs.to('cpu').detach().numpy().copy()
    path = './feature/' + basename + '_bn.npy'
    np.save(path, np_output)

    shutil.rmtree(save_path)  # 各フレームを破棄する
    os.rmdir("./frame")  # frameフォルダを破棄する
    print("save bn-inception feature  -->", path)
