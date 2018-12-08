import numpy as np
import cv2
import math


def sigmoid(x):
    res=1/(1+math.e**(-x))
    return res

def sigmoid_derivative(x):
    res=x*(1-x)
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
        self.output=output

    def get_output(self):
        return self.output

    def feed_forward(self):
        sum=0
        if len(self.connectors)!=0:
            for con in self.connectors:
                sum+=con.con_neuron.get_output()*con.weight
        else:
            return
        self.output=sigmoid(sum)

    def back_propagate(self):
        self.gradient=self.error*sigmoid_derivative(self.output)
        for con in self.connectors:
            con.d_weight=Neuron.E*(con.con_neuron.output*self.gradient)+self.A*con.d_weight
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

    def feed_forward(self):
        for layer in self.layer_list[1:]:
            for n in layer:
                n.feed_forward()

    def back_propagate(self,prev):
        for i in range(len(prev)):
            self.layer_list[-1][i].set_error(prev[i]-self.layer_list[-1][i].get_output())
        for layer in self.layer_list[::-1]:
            for n in layer:
                n.back_propagate()
    def get_results(self):
        output=[]
        for n in self.layer_list[-1]:
            out=n.get_output()
            if out>0.5:
                o=1
            else:
                o=0
            output.append(out)
        output.pop()     #remove bias neuron
        return output

def main():
    set=[]
    set.append(2)
    set.append(3)
    set.append(2)
    net=N_Network(set)
    Neuron.E=0.09
    Neuron.A=0.015
    inputs=[[0,0],[0,1],[1,0],[1,1]]
    outputs=[[0,0],[1,0],[1,0],[0,1]]
    while True:
        err=0
        for i in range(len(inputs)):
            net.set_input(inputs[i])
            net.feed_forward()
            net.back_propagate(outputs[i])
            err+=net.get_error(outputs[i])
        print("error: ",err)
        if err<0.05:
            break
    while True:
        a=int(input("type 1st input: "))
        b=int(input("type 2nd input: "))
        net.set_input([a,b])
        net.feed_forward()
        print(net.get_results())
main()

# image=cv2.imread("train_img/train_image_2233.bmp",0)
# img=cv2.bitwise_not(image)
# print(img)
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()