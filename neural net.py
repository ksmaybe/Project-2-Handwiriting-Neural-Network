import struct
import numpy as np
import cv2
import os

np.random.seed(0)
weights=np.random.rand(28*28)
bias=np.random.rand(1)
lr=0.1  #learning rate
class NeuralNetwork:
    def __init__(self,hidden,output):
        self.error=0
        self.input=input
        #[weight,d_change]
        self.h=hidden
        self.hidden=[[None,None] for zz in range(hidden)]
        self.h_out=np.array([None]*self.h)

        self.o=output
        self.output=[[None,None] for zz in range(output)]
        self.o_out=np.array([None]*self.o)
        d_weight1=np.array([None]*self.h)
        self.h_dw=d_weight1
        d_weight2=np.array([None]*self.o)
        self.o_dw=d_weight2
        d_total1=np.array([None]*self.h)
        self.h_dt=d_total1
        d_total2=np.array([None]*self.o)
        self.o_dt=d_total2




def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

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
train_label=np.array(train_label) #[:no_of_train]

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
test_label=np.array(test_label) #[:no_of_test]

net=NeuralNetwork(30,5)
for i in range(net.h):
    weight=np.array(np.random.rand(28*28))
    net.hidden[i][0]=weight
    net.hidden[i][1]=np.array([None]*28*28)

for i in range(net.o):
    weight=np.array(np.random.rand(net.h))
    net.output[i][0]=weight
    net.output[i][1]=np.array([None]*net.h)

train_no=len(train_lst)
for epoch in range(1):
    for i in range(train_no):
        net.input=train_lst[i]
        for j in range(net.h):
            XW=np.dot(net.input,net.hidden[j][0])+bias
            z=sigmoid(XW)
            net.h_out[j]=z
            #print('h',XW)

        for k in range(net.o):
            XW=np.dot(net.h_out,net.output[k][0])+bias
            z=sigmoid(XW)
            net.o_out[k]=z

            #print('o',XW)
        error=net.o_out-train_label[i]
        print(error)
        err=[None]*5
        for ii in range(5):
            err[ii]=error[ii]**2
        MSE=np.divide(np.sum(err),2)
        net.error+=MSE
        print(MSE)
        print("total error: ",net.error)
        print(net.o_out)
        print("image: ",i+1)
        for pp in range(5):

            net.o_dw[pp]=sigmoid_derivative(net.o_out[pp])
            net.o_dt[pp]=-(train_label[i][pp]-net.o_out[pp])

            net.output[pp][1]=net.o_dw[pp]*net.o_dt[pp]*net.h_out
            net.output[pp][0]=np.subtract(net.output[pp][0],net.output[pp][1])


        for ll in range(net.h):
            net.h_dw[ll]=sigmoid_derivative(net.h_out[ll])
            zzz=0
            for llp in range(net.o):
                zzz+=net.output[llp][0][ll]
            print(zzz,"haha")
            zz=np.sum(net.o_dw*zzz)
            #net.o_dt[llp]*net.o_dw[llp]*net.output[llp][0][ll]
            net.h_dt[ll]=zz*net.o_dt
            # for lll in range(28*28):
            #     z=net.input[lll]
                #net.hidden[ll][1][lll]=net.h_dw[ll]*net.h_dt[ll]*z

            print(net.h_dw[ll], net.h_dt[ll])

            net.hidden[ll][1]=net.input#*zzzz

            net.hidden[ll][0]=np.subtract(net.hidden[ll][0],net.hidden[ll][1])
        print("dw")
        print(net.h_dw)
        print(net.o_dw)
        print("else")

        # error= z-train_label[i]
        # print("image: ",i+1)
        # print("epoch: ",epoch+1)
        # print("error sum: ",error.sum())
        #
        #
        # dcost_dpred=error
        # dpred_dz=sigmoid_derivative(z)
        #
        # z_delta=dcost_dpred*dpred_dz
        #
        # inputs=train_lst[i].T
        # weights-=lr*np.dot(inputs,z_delta)
        #
        # for num in z_delta:
        #     bias-=lr*num
    print("epoch: ",epoch+1)
    if net.error<0.01:
        break





