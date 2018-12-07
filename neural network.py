import numpy as np
import cv2
import os


class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(y.shape)



image=cv2.imread("train_img/train_image_0.bmp",1)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()