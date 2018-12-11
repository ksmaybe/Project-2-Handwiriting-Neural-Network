import struct

import numpy as np
import cv2
import math
import os
from numba import vectorize

np.random.seed(0)
def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

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
            x=[None]*len(self.connectors)
            y=[None]*len(self.connectors)
            i=0
            for con in self.connectors:
                x[i]=con.con_neuron.output
                y[i]=con.weight

                i+=1
            x=np.array(x)
            y=np.array(y)
            sum=np.dot(x,y)
            self.output=sigmoid(sum)
            xx=0
        else:
            self.output=0.5


    def back_propagate(self):
        self.gradient=self.error*sigmoid_derivative(self.output)
        l=len(self.connectors)
        x=[None]*l
        y=[None]*l
        z=[None]*l
        w=[None]*l
        cons=self.connectors
        for con in self.connectors:
        #     x[i]=con[i].con_neuron.output
        #     y[i]=con[i].d_weight
        #     z[i]=con[i].con_neuron.error
        #     w[i]=con[i].weight
        # x=np.array(x)
        # y=np.array(y)
        # z=np.array(z)
        # w=np.array(w)
        # for i in range(l):
        #     y[i]=self.E*(x[i]*self.gradient)+self.A*y[i]
        #     w[i]=w[i]+y[i]
        #     z[i]=z[i]+(w[i]*self.gradient)


        # aa=self.E*x*self.gradient
        # bb=self.A*y
        # y=aa+bb
        # w=w+y
        # z=z+(w*self.gradient)
            con.d_weight=self.E*(con.con_neuron.output*self.gradient)+self.A*con.d_weight
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

    def get_error(self,y):
        error=0
        zz=[None]*len(y)
        for i in range(len(y)):

            zz[i]=(y[i]-self.layer_list[-1][i].output)
            #error+=err**2
        zz=np.array(zz)
        k=np.power(zz,2)
        k=np.divide(k,len(y))
        k=np.sum(k)
        error=np.sqrt(k)
        # error/=len(y)
        # error=math.sqrt(error)
        return error

    def feed_forward(self):
        for layer in self.layer_list[1:]:
            for n in layer:
                n.feed_forward()

    def back_propagate(self,prev):
        kk=[None]*len(prev)
        gg=[None]*len(prev)
        for i in range(len(prev)):
            kk[i]=prev[i]
            gg[i]=self.layer_list[-1][i].output

            #self.layer_list[-1][i].set_error(prev[i]-self.layer_list[-1][i].output)
        kk=np.array(kk)
        gg=np.array(gg)
        for i in range(len(prev)):
            self.layer_list[-1][i].error=kk[i]-gg[i]
        for layer in self.layer_list[::-1]:
            for n in layer:
                n.back_propagate()

    def get_results(self):
        output=[]
        for n in self.layer_list[-1]:
            out=n.output
            # if out>0.5:
            #     out=1
            # else:
            #     out=0
            output.append(out)
        output.pop()     #remove bias neuron
        return output

train_image="train_images.raw"

def byteToPixel(file,width,length):
    stringcode='>'+'B'*len(file)
    x=struct.unpack(stringcode,file)

    data=np.array(x)

    data=data.reshape(int(len(file)/(width*length)),width*length,1)/255
    return data

ff=open(train_image,'rb')
bytefile=ff.read()
train_lst=byteToPixel(bytefile,28,28)


# #read train image to integer values

# train_lst=[]
# p="train_img/"
# x=os.listdir("train_img")
# no_of_train=10 #len(x)
# for i in range(no_of_train):
#     image=cv2.imread(p+x[i],0)
#     img=cv2.bitwise_not(image)
#     img1=[]
#     for c in img:
#         img1.extend(c)
#     train_lst.append(img1)

#read training labels
f=open("train_labels.txt",'r')
read_lines_train=f.readlines()
train_label=[]
for line in read_lines_train:
    mlst=[]
    for c in line:
        if c.isnumeric():
            mlst.append(int(c))
    train_label.append(mlst)
train_label=train_label #[:no_of_train]

#read test image to integer values

test_image="test_images.raw"


fg=open(test_image,'rb')
bytefile1=fg.read()
test_lst=byteToPixel(bytefile1,28,28)
no_of_test=len(test_lst)

# test_lst=[]
# p="train_img/"
# k=os.listdir("train_img")
#no_of_test=len(k)
# for i in range(no_of_test):
#     image=cv2.imread(p+k[i],0)
#     img=cv2.bitwise_not(image)
#     img1=[]
#     for c in img:
#         img1.extend(c)
#     test_lst.append(img1)

#read test labels
g=open("test_labels.txt",'r')
read_lines_test=g.readlines()
test_label=[]
for line in read_lines_test:
    mlst=[]
    for c in line:
        if c.isnumeric():
            mlst.append(int(c))
    test_label.append(mlst)
test_label=test_label #[:no_of_test]



#begin neural network
set=[]
set.append(28*28)
set.append(50)
set.append(5)
net=N_Network(set)
Neuron.E=0.09
Neuron.A=0.015
inputs=train_lst
outputs=train_label
while True:
    err=0
    zz=1
    for i in range(len(inputs)):
        net.set_input(inputs[i][0])
        net.feed_forward()
        net.back_propagate(outputs[i])
        err=net.get_error(outputs[i])
        print(zz,"output train: ",net.get_results())
        print("train_label: ", train_label[i])
        zz+=1
        break
    print("total err: ",err)
    if err<0.1:
        break
for z in range(no_of_test):
    k=test_lst[z]
    net.set_input(k)
    net.feed_forward()
    print("Results: ",net.get_results())
    print("Label: ",test_label[z])

