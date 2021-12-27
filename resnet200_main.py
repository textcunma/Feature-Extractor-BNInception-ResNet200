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

sys.path.append("pytorch-image-models")
import timm


def Resnet200_main(movie_name, batch_size):
    basename = os.path.splitext(os.path.basename(movie_name))[0]
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
    files = os.listdir(save_path)
    img_sort = natsorted(files)

    transform = transforms.Compose(
        [
            transforms.Resize(256),  # (256, 256) で切り抜く。
            transforms.CenterCrop(256),  # 画像の中心に合わせて、(256, 256) で切り抜く
            transforms.ToTensor(),  # テンソルにする。
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                 ),  # 標準化する。
        ]
    )
    img_list = []
    # 画像変形
    for i in range(len(img_sort)):
        img = Image.open(save_path + '/' + str(i) + '.jpg').convert('RGB')
        img = transform(img)
        img_list.append(img)

    # モデル作成
    model = timm.create_model('resnet200d', pretrained=True)
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
    path='./feature/' + basename + '_resnet.npy'
    np.save(path, np_output)

    shutil.rmtree(save_path)  # 各フレームを破棄する
    os.rmdir("./frame")  # frameフォルダを破棄する
    print("save resnet200 feature  -->", path)
