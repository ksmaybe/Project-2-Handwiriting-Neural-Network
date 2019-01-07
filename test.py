import struct


import random

import gzip
import cv2
import os
import copy
import numpy as np


#
# f = gzip.open('train-labels-idx1-ubyte.gz','r')
# train_labeler=np.array([])
#
# for i in range(1):
#     buf = f.read(8)
#     labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#     print(labels)
# for i in range(60000):
#     buf = f.read(1)
#     labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#     train_labeler=np.append(train_labeler,labels)
# train_label=[]
# for j in range(len(train_labeler)):
#     x=[0]*10
#     x[int(train_labeler[j])]=1
#     train_label.append(x)
# print(len(train_label))


f = gzip.open('train-images-idx3-ubyte.gz','r')
train_lst=np.array([])

for i in range(2):
    buf = f.read(8)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)
kk=0
kkk=0
buf = f.read(28*28*60001)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
train_lst=np.append(train_lst,labels)
print(train_lst,'train_lst')
for j in range(len(train_lst)):
    if train_lst[j]>0:
        kk+=1
    if train_lst[j]>kkk:
        kkk=train_lst[j]

print(kk)
print(kkk)
train_lst=train_lst.reshape(int(len(train_lst)/(28*28)),28*28)/255
print(len(train_lst))



# for i in range(2):
#     buf = f.read(8)
#     labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#     print(labels)
# for i in range(28):
#     buf = f.read(28)
#     labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#     print(labels)