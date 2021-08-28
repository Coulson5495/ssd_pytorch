import torch
import torch.nn as nn

'''
该代码用于获得VGG主干特征提取网络的输出。
输入变量i代表的是输入图片的通道数，通常为3。

一般来讲，输入图像为(300, 300, 3)，随着base的循环，特征层变化如下：
300,300,3 -> 300,300,64 -> 300,300,64 -> 150,150,64 -> 150,150,128 -> 150,150,128 -> 75,75,128 -> 75,75,256 -> 75,75,256 -> 75,75,256 
-> 38,38,256 -> 38,38,512 -> 38,38,512 -> 38,38,512 -> 19,19,512 ->  19,19,512 ->  19,19,512 -> 19,19,512
到base结束，我们获得了一个19,19,512的特征层

a、输入一张图片后，被resize到300x300的shape

b、conv1，经过两次[3,3]卷积网络，输出的特征层为64，输出为(300,300,64)，再2X2最大池化，该最大池化步长为2，输出net为(150,150,64)。

c、conv2，经过两次[3,3]卷积网络，输出的特征层为128，输出net为(150,150,128)，再2X2最大池化，该最大池化步长为2，输出net为(75,75,128)。

d、conv3，经过三次[3,3]卷积网络，输出的特征层为256，输出net为(75,75,256)，再2X2最大池化，该最大池化步长为2，输出net为(38,38,256)。

e、conv4，经过三次[3,3]卷积网络，输出的特征层为512，输出net为(38,38,512)，再2X2最大池化，该最大池化步长为2，输出net为(19,19,512)。
    conv4中第三个卷积激活后会进行先验框的回归和种类的预测

f、conv5，经过三次[3,3]卷积网络，输出的特征层为512，输出net为(19,19,512)，再3X3最大池化，该最大池化步长为1，输出net为(19,19,512)。
    conv5中第二个卷积激活后会新型先验框的回归和种类的预测


之后进行pool5、conv6、conv7。
'''

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512]  # M是最大池化层；C是inplace=true


def vgg(i):
    layers = []
    in_channels = i
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=7)]
    return layers


if __name__ == "__main__":
    vgg_net = vgg(3)
    print(len(vgg_net))
