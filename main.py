"""
BN-Inception と　ResNet200 　特徴抽出
"""
import os
from bn_inception_main import BN_inception_main
from resnet200_main import Resnet200_main


def main():
    MP4File = "./test1.mp4"  # MP4ファイル名指定（名前ではなくパスを指定）
    batchSize = 300  # PCスペックによって下げるべき
    os.makedirs("./feature", exist_ok=True)  # featureディレクトリ作成

    BN_inception_main(MP4File, batchSize)  # BN-Inception　特徴抽出
    Resnet200_main(MP4File, batchSize)  # ResNet200　特徴抽出


if __name__ == '__main__':
    main()
