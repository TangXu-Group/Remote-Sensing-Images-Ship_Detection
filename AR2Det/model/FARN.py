import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.backends
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from math import pi
import math

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f.append(x)
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        x = self.layer4(x)
        f.append(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        '''
        f中的每个元素的size分别是 bs 256 w/4 h/4， bs 512 w/8 h/8， 
        bs 1024 w/16 h/16， bs 2048 w/32 h/32
        '''
        return f

def resnet9(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        
        #model.load_state_dict(torch.load("./resnet50-19c8e357.pth"))
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, progress=True, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained=False, progress=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False, progress=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained=False, progress=True, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels = 3, out_channels = 128):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class FARN(nn.Module):
    def __init__(self,input_channel = 3, backbone_pretrained = True, trans_channel_num = 64, resnet_type = 'resnet34', resnet_layer = [512,256,128,64], boxes_dx_dy = 40, boxes_w_h = 512, centers_dx_dy = 512):
        super(FARN,self).__init__()
        if resnet_type == 'resnet9':
            self.resnet = resnet9(backbone_pretrained)
        if resnet_type == 'resnet18':
            self.resnet = resnet18(backbone_pretrained)
        if resnet_type == 'resnet34':
            self.resnet = resnet34(backbone_pretrained)
        if resnet_type == 'resnet50':
            self.resnet = resnet50(backbone_pretrained)
        if resnet_type == 'resnet101':
            self.resnet = resnet101(backbone_pretrained)
        if resnet_type == 'resnet152':
            self.resnet = resnet152(backbone_pretrained)
            
        conv1_inchannel_num = trans_channel_num*2
        conv2_inchannel_num = trans_channel_num*3
        conv3_inchannel_num = trans_channel_num*4
        conv4_inchannel_num = trans_channel_num*5
 
        self.conv1 = nn.Conv2d(conv1_inchannel_num, trans_channel_num, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(trans_channel_num)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(conv2_inchannel_num, trans_channel_num, 3,padding=1)
        self.bn2 = nn.BatchNorm2d(trans_channel_num)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(conv3_inchannel_num, trans_channel_num, 3,padding=1)
        self.bn3 = nn.BatchNorm2d(trans_channel_num)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(conv4_inchannel_num, trans_channel_num, 3,padding=1)
        self.bn4 = nn.BatchNorm2d(trans_channel_num)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.convF = nn.Conv2d(trans_channel_num, 2, 1)
        self.sigmoidF = nn.Sigmoid()

        self.convG1 = nn.Conv2d(trans_channel_num, trans_channel_num//2, 3, padding=1)
        self.bnG1 = nn.BatchNorm2d(trans_channel_num//2)
        self.reluG1 = nn.ReLU(inplace=True)

        self.convG2 = nn.Conv2d(trans_channel_num, trans_channel_num//2, 3, padding=1)
        self.bnG2 = nn.BatchNorm2d(trans_channel_num//2)
        self.reluG2 = nn.ReLU(inplace=True)

        self.convG3 = nn.Conv2d(trans_channel_num//2, 2, 1)
        self.sigmoidG3 = nn.Sigmoid()
        self.convG4 = nn.Conv2d(trans_channel_num//2, 2, 1)
        self.sigmoidG4 = nn.Sigmoid()
        self.convG5 = nn.Conv2d(trans_channel_num//2, 1, 1)
        self.sigmoidG5 = nn.Sigmoid()
        
        self.convG6 = nn.Conv2d(trans_channel_num//2, 2, 1)
        self.sigmoidG6 = nn.Sigmoid()
         
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.convl1 = nn.Conv2d(resnet_layer[0],trans_channel_num,1)
        self.convl2 = nn.Conv2d(resnet_layer[1],trans_channel_num,1)
        self.convl3 = nn.Conv2d(resnet_layer[2],trans_channel_num,1)
        self.convl4 = nn.Conv2d(resnet_layer[3],trans_channel_num,1)
        self.convl5 = DoubleConv(input_channel,trans_channel_num)
        self.boxes_dx_dy = boxes_dx_dy
        self.boxes_w_h = boxes_w_h
        self.centers_dx_dy = centers_dx_dy
    def forward(self,images):
        f = self.resnet(images)
        c5 = self.convl1(f[4])
        c4 = self.convl2(f[3])
        c3 = self.convl3(f[2])
        c2 = self.convl4(f[1])
        c1 = self.convl5(images) 
           
        p5 = self.unpool(c5)
        p4 = self.conv1(torch.cat((p5,c4), 1))
        p4 = self.bn1(p4)
        p4 = self.relu1(p4)
        
        p4 = self.unpool(p4) 
        p5 = self.unpool(p5)
        p3 = self.conv2(torch.cat((p4,p5,c3), 1))
        p3 = self.bn2(p3)
        p3 = self.relu2(p3)

        p3 = self.unpool(p3) 
        p4 = self.unpool(p4)
        p5 = self.unpool(p5)
        p2 = self.conv3(torch.cat((p3,p4,p5,c2), 1))
        p2 = self.bn3(p2)
        p2 = self.relu3(p2)
        
        p1 = self.conv4(torch.cat((p2,p3,p4,p5,c1), 1)) 
        p1 = self.bn4(p1)
        p1 = self.relu4(p1)
        
        F_scores_bodies = self.convF(p1)
        F_scores_bodies = self.sigmoidF(F_scores_bodies)
        
        g1 = self.convG1(p1) 
        g1 = self.bnG1(g1)
        g1 = self.reluG1(g1)
        
        g2 = self.convG2(p1)
        g2 = self.bnG2(g2)
        g2 = self.reluG2(g2)
        
        geo_mapdxy = self.convG3(g1)
        geo_mapdxy = (self.sigmoidG3(geo_mapdxy)-0.5) * self.boxes_dx_dy
        geo_mapwh = self.convG4(g1)
        geo_mapwh = self.sigmoidG4(geo_mapwh) * self.boxes_w_h
        angle_map = self.convG5(g1)
        angle_map = (self.sigmoidG5(angle_map)-0.5) * pi
        
        Fxy = self.convG6(g2)
        Fxy = (self.sigmoidG6(Fxy)-0.5) * self.centers_dx_dy
        F_geometry = torch.cat((geo_mapdxy,geo_mapwh, angle_map,Fxy), 1)
        
        return F_scores_bodies, F_geometry
