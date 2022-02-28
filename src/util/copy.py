# -*- coding: UTF-8 -*-
import os
import random
import shutil
import skimage.io as io
import pdb


def copyFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)
    pathDir.sort()
    # for filename in pathDir:
        # print (filename)

    coll = io.ImageCollection(str)
    print(len(coll))  # 打印图片数量
    num = int((2 * len(coll)) / 10)
    print(num)
    sample = []
    # pdb.set_trace()
    a=0
    for i in pathDir:
        a += 1
        if a == 8:
            sample.append(i)
            print("add ", i)
            a=0

    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)


if __name__=='__main__':
    fileDir = "/home/ruian/视频/refer/"  # 填写要读取图片文件夹的路径
    tarDir = "/home/ruian/视频/refer2/"  # 填写保存随机读取图片文件夹的路径
    str = fileDir+'*.png'  # fileDir的路径+*.jpg表示文件下的所有jpg图片
    copyFile(fileDir, tarDir)