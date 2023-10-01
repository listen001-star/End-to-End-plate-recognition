import numpy as np
import cv2 as cv
import os

provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]

def resizePic(path):
    filelist = os.listdir(path)
    total_num = len(filelist)

    i = 0

    for item in filelist:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(path), item)
            img = cv.imread(src)
            dst = cv.resize(img, (512, 512))

            i = i + 1

            # 图像名
            imgname = os.path.basename(src).split('.')[0]

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

            # --- 读取车牌号
            label = label.split('_')
            # 省份缩写
            province = provincelist[int(label[0])]
            # 车牌信息
            words = [wordlist[int(i)] for i in label[1:]]
            # 车牌号
            label = province + ''.join(words)

            # 存储图片
            src1 = os.path.join(os.path.abspath(path), label+'.png')
            cv.imencode('.png', dst)[1].tofile(src1)
            os.remove(src)

    print('total %d to rename & resized %d jpgs' % (total_num, i))


if __name__ == '__main__':
    path = './test'
    resizePic(path)


