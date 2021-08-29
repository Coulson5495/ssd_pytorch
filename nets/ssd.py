import torch
import torch.nn as nn
from nets.vgg import vgg as add_vgg
import torch.nn.functional as F
# from utils.config import Config
from ssd_layer import L2Norm


def add_extras(i, backbone_namne):
    '''
    输入extra layers 的通道数为1024
    :param i: 输入此模块的通道
    :param backbone_namne: backbone采用什么网络 vgg 或者轻量级网络
    :return: 返回extra部分
    '''

    in_channels = i
    layers = []
    if backbone_namne == 'vgg':
        # block 6
        # 19,19,1024->10,10,512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

        # Block 7
        # 10,10,512 -> 5,5,256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # Block 8
        # 5,5,256 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

        # Block 9
        # 3,3,256 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    else:
        pass

    return layers


def get_ssd(phase, num_classes, backbone_name, confidence=0.5, nms_iou=0.45):
    '''

    :param phase: training or test
    :param num_class: 预测种类个数
    :param backbone_name: 主干网络
    :param confidence: 置信度
    :param nms_iou: 非极大抑制的iou
    :return:
    '''

    if backbone_name == 'vgg':
        backbone, extra_layers = add_vgg(3), add_extras(1024, backbone_name)
        mbox = [4, 6, 6, 6, 4, 4]
    else:
        pass

    loc_layers = []  # 先验框的回归list
    conf_layers = []  # 种类预测的list
    if backbone_name == 'vgg':
        # 首先对vgg中conv4中第三个卷积激活和conv5中第二个卷积激活进行先验框的回归和种类的预测
        backbone_source = [21, -2]  # 21->conv4中第三个卷积激活; -2->conv5中第二个卷积激活
        # ---------------------------------------------------#
        #   在add_vgg获得的特征层里
        #   第21层和-2层可以用来进行回归预测和分类预测。
        #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
        # ---------------------------------------------------#
        for k, v in enumerate(backbone_source):
            # 位置回归
            loc_layers += [nn.Conv2d(backbone[v].out_channels,
                                     mbox[k] * 4, kernel_size=3, padding=1)]
            # 上面mbox[k]*4的含义是：一共有mbox[k]个框，每个先验框需要四个维度，即x,y,w,h

            conf_layers += [nn.Conv2d(backbone[v].out_channels,
                                      mbox[k] * num_classes, kernel_size=3, padding=1)]

        # -------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        # -------------------------------------------------------------#
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                      * num_classes, kernel_size=3, padding=1)]
    else:
        pass


class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes, confidence, nms_iou, backbone_name):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.backbone_name = backbone_name

        # self.cfg = Config
        if backbone_name == 'vgg':
            self.vgg = nn.ModuleList(base)
            self.L2Norm = L2Norm(512, 20)
        else:
            pass
        self.extras = nn.ModuleList(extras)
        # 未完成

    def forward(self):
        # 将backbone与extras连接起来
        sources = list()
        loc = list()
        conf = list()

        # ---------------------------#
        #   获得conv4_3的内容
        #   shape为38,38,512
        # ---------------------------#
        if self.backbone_name == 'vgg':
            for k in range(23):
                x = self.vgg[k](x)
        else:
            pass
        # ---------------------------#
        #   conv4_3的内容
        #   需要进行L2标准化
        # ---------------------------#
        sources.append(self.L2Norm(x))

        # ---------------------------#
        #   获得conv7的内容
        #   shape为19,19,1024
        # ---------------------------#
        if self.backbone_name == 'vgg':
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
            else:
                pass
        sources.append(x)

        # -------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        # -------------------------------------------------------------#
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == "vgg":
                if k % 2 == 1:
                    sources.append(x)
            else:
                pass

        # -------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        # -------------------------------------------------------------#
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # -------------------------------------------------------------#
        #   进行reshape方便堆叠
        # -------------------------------------------------------------#
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # -------------------------------------------------------------#
        #   loc会reshape到batch_size,num_anchors,4
        #   conf会reshap到batch_size,num_anchors,self.num_classes
        #   如果用于预测的话，会添加上detect用于对先验框解码，获得预测结果
        #   不用于预测的话，直接返回网络的回归预测结果和分类预测结果用于训练
        # -------------------------------------------------------------#
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output
