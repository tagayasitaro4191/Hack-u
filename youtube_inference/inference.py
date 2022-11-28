#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep
#----- 専用ライブラリ -----#
import torch
import torchvision.transforms as tf
import numpy as np
import cv2

#----- 自作モジュール -----#
from youtube_inference.Resnet import Resnet50




class AI_model():
    def __init__(self):
        # モデル設定
        model = Resnet50(pretrained=False).cuda("cuda:0")
        model.load_state_dict(torch.load("youtube_inference/save_model/model.pth"))
        model.eval()
        self.model = model

        self.transform = tf.Compose([Totensor(),
                                tf.Resize((240, 320)),
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])
    
    def __call__(self, img):
        img = self.transform(img).float().to("cuda:0")
        view_count = self.model(img)
        view_count = 10 ** (view_count.item())
        return view_count


# Totensor
class Totensor():
    def __init__(self):
        pass

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        return torch.from_numpy(img).permute(2, 0, 1)[None]
