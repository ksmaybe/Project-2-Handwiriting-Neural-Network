import struct

import cv2
import os
import copy
import rawpy
import imageio
import numpy as np
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
# print(z)
# x=1

x=259
y=77
z=(x%y)
print(z)
print(y%z)