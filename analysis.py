# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:24:10 2021

@author: luohenyueji
"""
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import cv2 as cv

def cutFilename(imgname):
    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')

    # --- 边界框信息
    box = box.split('_')
    box = [list(map(int, i.split('&'))) for i in box]

    # --- 关键点信息
    points = points.split('_')
    points = [list(map(int, i.split('&'))) for i in points]
    # 将关键点的顺序变为从左上顺时针开始
    points = points[-2:] + points[:2]

    return points


def resizePic(path):
    filelist = os.listdir(path)
    total_num = len(filelist)

    i = 0

    for item in filelist:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(path), item)
            src_img = os.path.join('./unet_datasets/train_image', str(i)+'.png')
            src_lab = os.path.join('./unet_datasets/train_label', str(i)+'.png')

            # 图像名
            imgname = os.path.basename(src).split('.')[0]

            points = cutFilename(imgname)

            ImgShow(src,points,src_img,src_lab)

            i = i + 1


            # 存储图片
            # src1 = os.path.join(os.path.abspath(path), label+'.png')
            # cv.imencode('.png', dst)[1].tofile(src1)
            # os.remove(src)

    print('total %d to rename & resized %d jpgs' % (total_num, i))


# --- 绘制边界框


def DrawPoint(im, points):

    draw = ImageDraw.Draw(im)

    draw.polygon([tuple(points[0]), tuple(points[1]), tuple(points[2]), tuple(points[3])], (255, 255, 255, 255))

# --- 绘制车牌


def DrawLabel(im, label):
    draw = ImageDraw.Draw(im)
   # draw.multiline_text((30,30), label.encode("utf-8"), fill="#FFFFFF")
    font = ImageFont.truetype('simsun.ttc', 64)
    draw.text((30, 30), label, font=font)

# --- 图片可视化


def ImgShow(imgpath, points, src1, src2):
    # 打开图片
    im1 = Image.open(imgpath)
    im2 = Image.new(mode='RGB',size=(720,1160), color=(0,0,0))
    DrawPoint(im2, points)

    # 缩放图片
    im1 = im1.resize((512,512))
    im2 = im2.resize((512,512))

    # 存储图片
    im1.save(src1)
    im2.save(src2)

def main():
    path = './test'
    resizePic(path)


main()

