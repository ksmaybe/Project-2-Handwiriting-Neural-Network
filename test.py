import struct


import random

import gzip
import cv2
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

# #read train image to integer values
# no_of_train=1
# train_lst=[]
# p="train_img/"
# x=os.listdir("train_img")
# for i in range(no_of_train):
#     image=cv2.imread(p+x[i],0)
#     img=cv2.bitwise_not(image)
#     img1=[]
#     for c in img:
#         img1.extend(c)
#     print(img1)
#
#
# print(train_lst)

#read train image to integer values
# no_of_train=3000
# train_lst=[]
# p="train_img/"
# x=os.listdir("train_img")
# for i in range(no_of_train):
#     image=cv2.imread(p+x[i],0)
#     img=cv2.bitwise_not(image)
#     img1=[]
#     for c in img:
#         img1.extend(c)
#     train_lst.append(img1)
#
# print(train_lst)

#
#
# train_image="train_images.raw"
#
# def byteToPixel(file,width,length):
#     stringcode='>'+'B'*len(file)
#     x=struct.unpack(stringcode,file)
#
#     data=np.array(x)
#
#     data=data.reshape(int(len(file)/(width*length)),width*length)/255
#
#     return data
#
# ff=open(train_image,'rb')
# bytefile=ff.read()
# train_lst=byteToPixel(bytefile,28,28)
# x=train_lst[0]
#
# z=np.random.random((50,5))
# xz=np.array([1]*50)
# for h in z:
#     for k in h:
#         k=np.random.normal()
#
# print(z)x=259
# y=77
# z=(x%y)
# print(z)
# print(y%z)
# x=1

# f=gzip.open('t10k-images-idx3-ubyte.gz','r')
# f.read(16)
# buf=f.read(28*28*5)
# image_size=28
# num_images=5
# data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
# data = data.reshape(num_images, image_size, image_size, 1)
# image=np.asarray(data[2]).squeeze()
# plt.imshow

f = gzip.open('train-labels-idx1-ubyte.gz','r')
for i in range(0,3):
    buf = f.read(8)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)
for i in range(0):
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)