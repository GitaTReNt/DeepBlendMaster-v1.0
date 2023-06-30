import torch
import numpy as np
import torch
import imageio as iio
import u2net_human_seg_test
from PIL import Image
from skimage.io import imsave
from torchvision.utils import save_image
import argparse
import pdb
import os
from PIL import Image
import argparse
import pdb
import torch.nn.functional as F
import cv2
from PIL import Image
def imgreshape(source_file):
    image=Image.open(source_file)
    image = image.convert('RGB')
    w, h = image.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(255, 255, 255))  # 创建背景图，颜色值为255
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(image, box)
    image_data=background.resize((max(w,h),max(w,h)))#缩放   
    image_data.save(source_file)
    image_data.save('test_data/test_human_images/1_source.png')
    return image_data

def Clickchoose(target_file,ts):
    
    img2 = cv2.imread(target_file)
    img2 = cv2.resize(img2, (ts,ts))
    size = img2.shape
   # img2 = np.array(Image.open(target_file).convert('RGB').resize((ss, ss)))
    size = img2.shape
    x2, y2, w2, h2 = cv2.selectROI('roi',img2)
    cv2.destroyWindow('roi')
    w = size[1]  # 宽度
    h = size[0]  # 高度
    x_start,y_start = x2+(w2/2),y2+(h2/2)
   # right_bottom_source = (x2 + w2, y2 + h2)
    return y_start,x_start