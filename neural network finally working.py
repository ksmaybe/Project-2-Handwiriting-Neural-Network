import struct

import numpy as np


lr=0.5
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

bias =1

class Neural_NetWork(object):
    def __init__(self):
        #parameters
        self.input_size=784
        self.hidden_size=300
        self.output_size=5
        self.old_error=99999
        self.new_error=0
        self.o_error=999

        #The weight matrixes
        self.Weight_1=np.random.uniform(-2,2,(self.input_size,self.hidden_size))
        self.Weight_2=np.random.uniform(-2,2,(self.hidden_size,self.output_size))
        # for h in self.Weight_1:
        #     for k in h:
        #         k=np.random.normal()
        # for h in self.Weight_2:
        #     for k in h:
        #         k=np.random.normal()

    def feed_forward(self,X):

        self.z=np.dot(X,self.Weight_1)+bias
        self.z2=sigmoid(self.z)
        self.z3=np.dot(self.z2,self.Weight_2)+bias
        o=sigmoid(self.z3)
        return o

    def back_propagation(self,X,y,o):
        self.o_error=np.sum((y-o)**2)/2

        self.d_Et_Ot=-(y - o)
        self.d_o_net=sigmoid_derivative(o).reshape((1,5))
        self.d_net_w=self.z2.repeat(5).reshape(self.hidden_size,5)*(self.Weight_2**0)

        xx= self.d_Et_Ot * self.d_o_net
        self.d_error_w= xx*self.d_net_w
        self.Weight_2-=lr*self.d_error_w

        self.d_Eo_No=self.d_Et_Ot*self.d_o_net
        self.d_No_Oh=self.Weight_2

        self.d_Eo_Oh=self.d_Eo_No*self.d_No_Oh
        self.d_Et_Oh=np.sum(self.d_Eo_Oh,axis=1)

        self.d_Oh_Nh=sigmoid_derivative(self.z2)
        yy=self.d_Et_Oh*self.d_Oh_Nh
        self.d_Et_w=X.repeat(self.hidden_size).reshape(784,self.hidden_size)*yy.reshape((1,self.hidden_size))
        self.Weight_1-=lr*self.d_Et_w


        # self.o_error=y-o
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
lstp=[]
for e in range(100):
    print("e:",e)
    for i in range(10000):
        X=train_lst[i]
        y=train_label[i]
        o=net.feed_forward(X)
        net.train(X,y)
        net.new_error+=net.o_error
    lstp.append(net.new_error)
    print(net.new_error)
    if net.old_error-net.new_error<10:
        break
    net.old_error=net.new_error
    net.new_error=0



    # net.new_error=net.old_error
    # if net.new_error-net.old_error<0.01:
    #     break
confusion_matrix=np.array([0]*25).reshape(5,5)
for i in range(len(test_label)):

    o=net.feed_forward(test_lst[i])
    x=0
    y=0
    for j in range(5):
        if test_label[i][j]==1:
            x=j
            break

    for j in range(len(o)):
        if max(o)==o[j]:
            y=j
            break
    confusion_matrix[x][y]+=1

x=0


print()
print("confusion matrix")
print(confusion_matrix)