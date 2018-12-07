import numpy as np
import cv2
import os
import math


def sigmoid(x):
    res=1/(1+math.e^(-x))
    return res

def sigmoid_derivative(x):
    f=sigmoid(x)
    res=f*(1-f)
    return res

class Connection:
    def __init__(self, con_neuron):
        self.con_neuron=con_neuron
        self.weight=np.random.normal()
        self.dWeight = 0.0




image=cv2.imread("train_img/train_image_2.bmp",0)
img=cv2.bitwise_not(image)
print(img)
