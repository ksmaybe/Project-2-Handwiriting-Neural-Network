import struct

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

bias =1

class Neural_NetWork(object):
    def __init__(self):
        #parameters
        self.input_size=784
        self.hidden_size=50
        self.output_size=5

        #The weight matrixes
        self.Weight_1=np.random.rand(self.input_size,self.hidden_size)
        self.Weight_2=np.random.rand(self.hidden_size,self.output_size)


    def feed_forward(self,X):

        self.z=np.dot(X,self.Weight_1)+bias
        self.z2=sigmoid(self.z)
        self.z3=np.dot(self.z2,self.Weight_2)+bias
        o=sigmoid(self.z3)
        return o

    def back_propagation(self,X,y,o):
        self.o_error=np.sum((y-o)**2)/2
        self.d_error_output=-(y-o)
        self.d_o_net=sigmoid_derivative(o)






        # self.o_delta=self.o_error*sigmoid_derivative(o)
        #
        # self.z2_error=self.o_delta.dot(self.Weight_2.T)
        # self.z2_delta=self.z2_error*sigmoid_derivative(self.z2)
        #
        # self.Weight_1+=X.T.dot(self.z2_delta)
        # self.Weight_2+=self.z2.T.dot(self.o_delta)

    def train(self,X,y):
        o=self.feed_forward(X)
        self.back_propagation(X,y,o)


train_image="train_images.raw"

def byteToPixel(file,width,length):
    stringcode='>'+'B'*len(file)
    x=struct.unpack(stringcode,file)

    data=np.array(x)

    data=data.reshape(int(len(file)/(width*length)),width*length)/255

    return data

ff=open(train_image,'rb')
bytefile=ff.read()
train_lst=byteToPixel(bytefile,28,28)



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
train_label=np.array(train_label) #[:no_of_train]

#read test image to integer values

test_image="test_images.raw"

fg=open(test_image,'rb')
bytefile1=fg.read()
test_lst=byteToPixel(bytefile1,28,28)
no_of_test=len(test_lst)


g=open("test_labels.txt",'r')
read_lines_test=g.readlines()
test_label=[]
for line in read_lines_test:
    mlst=[]
    for c in line:
        if c.isnumeric():
            mlst.append(int(c))
    test_label.append(mlst)
test_label=np.array(test_label) #[:no_of_test]


X=train_lst
y=train_label

net=Neural_NetWork()

for e in range(1):
    for i in range(1):
        X=train_lst[i]
        y=train_label[i]
        print("e: ",e)
        o=net.feed_forward(X)
        print("Input: ",X)
        print("Actual output: ",y)
        print("Predicted ouput: ",o)
        MSE=sum((y-o)**2)/2
        print("Loss: ",MSE)
        print()
        net.train(X,y)
        if MSE<0.01:
            break
o=net.feed_forward(train_lst[0])
print(o)