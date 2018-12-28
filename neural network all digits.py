import struct

import numpy as np



lr=0.5   #learning rate

#sigmoid function/activation
def sigmoid(x):
    return 1/(1+np.exp(-x))
#sigmoid prime
def sigmoid_derivative(x):
    return x*(1-x)

bias =1

class Neural_NetWork(object):
    def __init__(self):
        #parameters
        self.input_size=784
        self.hidden_size=300
        self.output_size=10
        self.old_error=99999    #sum of error
        self.new_error=0
        self.o_error=999

        #The weight matrixes
        self.Weight_1=np.random.uniform(-2,2,(self.input_size,self.hidden_size))
        self.Weight_2=np.random.uniform(-2,2,(self.hidden_size,self.output_size))


    def feed_forward(self,X):

        self.z=np.dot(X,self.Weight_1)+bias  #sum of Weight and output
        self.z2=sigmoid(self.z)                      #hidden layer activation
        self.z3=np.dot(self.z2,self.Weight_2)+bias
        o=sigmoid(self.z3)                      #output layer activation
        return o

    def back_propagation(self,X,y,o):
        self.o_error=np.sum((y-o)**2)/2         #get sum of error/ accuracy check

        #get Err
        self.d_Et_Ot=-(y - o)
        self.d_o_net=sigmoid_derivative(o).reshape((1,self.output_size))
        self.d_net_w=self.z2.repeat(self.output_size).reshape(self.hidden_size,self.output_size)*(self.Weight_2**0)

        #get dError/dWeight for output layer
        xx= self.d_Et_Ot * self.d_o_net
        self.d_error_w= xx*self.d_net_w
        self.Weight_2-=lr*self.d_error_w

        #get dError/dWeight for hidden layer
        self.d_Eo_No=self.d_Et_Ot*self.d_o_net
        self.d_No_Oh=self.Weight_2

        self.d_Eo_Oh=self.d_Eo_No*self.d_No_Oh
        self.d_Et_Oh=np.sum(self.d_Eo_Oh,axis=1)

        self.d_Oh_Nh=sigmoid_derivative(self.z2)
        yy=self.d_Et_Oh*self.d_Oh_Nh
        self.d_Et_w=X.repeat(self.hidden_size).reshape(784,self.hidden_size)*yy.reshape((1,self.hidden_size))
        self.Weight_1-=lr*self.d_Et_w


    def train(self,X,y):            #forward and back once/train once
        o=self.feed_forward(X)
        self.back_propagation(X,y,o)


train_image="train_images.raw"

#turn raw file to np array
def byteToPixel(file,width,length):
    stringcode='>'+'B'*len(file)
    x=struct.unpack(stringcode,file)

    data=np.array(x)

    data=data.reshape(int(len(file)/(width*length)),width*length)/255

    return data

ff=open(train_image,'rb')           #read raw
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

#start of training
net=Neural_NetWork()
lstp=[]
for e in range(100):
    print("e:",e)
    for i in range(len(train_lst)):
        X=train_lst[i]
        y=train_label[i]
        o=net.feed_forward(X)
        net.train(X,y)
        net.new_error+=net.o_error
    lstp.append(net.new_error)
    print(net.new_error)
    if net.old_error-net.new_error<5 and e>10 and net.new_error<1000:  #after 10 epoches and change in sum of error between epoch very small
        break
    net.old_error=net.new_error
    net.new_error=0

#draw confusion matrix
confusion_matrix=np.array([0]*25).reshape(5,5)
success=0
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
    if x==y:
        success+=1





print()
print("confusion matrix")
print(confusion_matrix)
print()
print("success: ",success,'/',len(test_label))
print("success rate: ",float(success/len(test_label)))
