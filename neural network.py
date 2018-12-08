import numpy as np
import cv2
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
        self.d_weight = 0.0


class Neuron:
    E=0.001
    A=0.01

    def __init__(self,layer):
        self.connectors=[]
        self.error=0.0
        self.gradient=0.0
        self.output=0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con=Connection(neuron)
                self.connectors.append(con)

    def add_error(self,error):
        self.error+=error

    def set_error(self,error):
        self.error=error

    def set_output(self,output):
        self.ouput=output

    def get_output(self):
        return self.output

    def feed_forward(self):
        sum=0
        if len(self.connectors)!=0:
            for con in self.connectors:
                sum+=con.con_neuron.output*con.weight
        else:
            return
        self.output=sigmoid(sum)

    def back_propagate(self):
        self.gradient=self.error*sigmoid_derivative(self.output)
        for con in self.connectors:
            con.d_weight=Neuron.E*(con.con_neuron.utput*self.gradient)+self.A*con.d_weight
            con.weight+=con.d_weight
            con.con_neuron.add_error(con.weight*self.gradient)
        self.error=0

class N_Network:
    def __init__(self,set):
        self.layer_list=[]
        for n in set:
            layer=[]
            for i in range(n):
                if len(self.layer_list)==0:
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layer_list[-1]))
            layer.append(Neuron(None))
            last_neuron=layer[-1]
            last_neuron.set_output(1)
            self.layer_list.append(layer)

    def set_input(self,input_list):
        for i in range(len(input_list)):
            self.layer_list[0][i].set_output(input_list[i])

    def get_error(self,x):
        error=0
        for i in range(len(x)):
            err=(x[i]-self.layer_list[-1][i].get_output())
            error+=err**2
        error/=len(x)
        error=math.sqrt(error)
        return error
image=cv2.imread("train_img/train_image_2233.bmp",0)
img=cv2.bitwise_not(image)
print(img)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()