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

z=np.array([1,2,3])
l=np.array([4,3,2])
x=z*l

print(x)