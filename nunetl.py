import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Convde(nn.Module):
    def __init__(self, filtersin, filters, dilation=(1, 1), kernel_size=[3, 3], padding='same'):
        super(Convde, self).__init__()
        '''w0 = torch.empty(filters, filtersin, 3, 3)
        nn.init.kaiming_uniform_(w0, a=math.sqrt(5))
        self.weight0 = nn.Parameter(w0)'''

        self.bias0 = nn.Parameter(torch.randn(1,filters,1,1))

        w1 = torch.empty(filters, filtersin, 3, 3)
        nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
        self.weight1 = nn.Parameter(w1)
        #self.bias1 = nn.Parameter(torch.randn(1,filters,1,1))
        ker1 = np.ones([filters, filtersin, 3, 3], dtype=np.float32)

        self.ker1 = torch.tensor(ker1, dtype=torch.float32, device=torch.device('cuda:0'))
        wo = torch.empty(filters, filtersin, 3, 3)
        nn.init.kaiming_uniform_(wo, a=math.sqrt(5))
        self.weighto = nn.Parameter(wo)
        filters = filters//2

        wc = torch.empty(filters*2, filters*2, 3, 3)
        #nn.init.kaiming_uniform_(wc, a=math.sqrt(5))
        self.weightc = nn.Parameter(wc)
        nn.init.kaiming_uniform_(self.weightc, a=math.sqrt(5))

        self.dilation = dilation
        #self.batch0 = torch.nn.InstanceNorm2d(filters, track_running_stats=False)
        #self.convconcat = nn.Conv2d(filters*2, filters*2, kernel_size=[3, 3], padding='same', dilation=self.dilation)
        self.batchc0 = torch.nn.InstanceNorm2d(filters*2, track_running_stats=False, affine=False)
        self.batchco = torch.nn.InstanceNorm2d(filters*2, track_running_stats=False, affine=False)

    def forward(self, inputs):
        w1 = torch.where(self.ker1>0.5, self.weight1, 0)
        mean = torch.sum(w1, dim=(2,3), keepdim=True)/9
        x1 = self.weight1-mean
        xo = F.conv2d(inputs, self.weighto, padding='same', dilation=self.dilation)
        #x0 = F.conv2d(inputs, torch.where(self.ker2>0.5, x1*2.25, x1), padding='same', dilation=self.dilation)
        x0 = F.conv2d(inputs, x1, padding='same', dilation=self.dilation)
        x0 = self.batchc0(x0)
        xo = self.batchco(xo)
        #x0 = torch.cat([x0,xo], dim=1)
        x0 = x0+xo
        #x0 = self.batchc0(x0+self.bias00)
        x0 = F.relu(x0)
        #x0 = F.conv2d(torch.cat([x0,xo], dim=1), F.sigmoid(self.weightc), padding='same', dilation=self.dilation)
        x0 = F.conv2d(x0, self.weightc, padding='same', dilation=self.dilation)
        #x0 = self.convconcat(x0)
        return x0 + self.bias0# * x1



class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.conv1 = Convde(in_ch, mid_ch, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.bn1 = nn.InstanceNorm2d(mid_ch, track_running_stats=False, affine=False)
        self.conv2 = Convde(mid_ch, out_ch, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.bn2 = nn.InstanceNorm2d(out_ch, track_running_stats=False, affine=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = F.relu(x)

        return output

'''class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.conv1 = Convde(in_ch, mid_ch, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.bn1 = nn.InstanceNorm2d(mid_ch, track_running_stats=False, affine=False)
        self.conv2 = Convde(mid_ch, mid_ch, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.bn2 = nn.InstanceNorm2d(mid_ch, track_running_stats=False, affine=False)
        self.conv3 = Convde(mid_ch, mid_ch, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.bn3 = nn.InstanceNorm2d(mid_ch, track_running_stats=False, affine=False)
        self.conv4 = Convde(mid_ch, out_ch, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.bn4 = nn.InstanceNorm2d(out_ch, track_running_stats=False, affine=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        output = F.relu(x)

        return output'''

#Nested Unet

class ConnectUnet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=3):
        super(ConnectUnet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] #通道数除了第一次，下采样每次加倍，上采样减半

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)#池化
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#上采样
        #这是左边半个U,通道数分别是从0->1,1->2,2->3,3->4,其他的看图上结构即可，对应图上相应的编号
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))#0->1,1->2,2->3,3->4的时候都有pool池化，减小图片大小
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))#当0层的图片与1层图片结合的时候，1层图片要做上采样使之与0层图片大小相同，x0_1代表图上第0层第1列的⚪

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))#x0_2是前面x0_0,x0_1,x1_1上采样结合，dense 连接，其他的类似

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        output = F.sigmoid(output)
        return [output], x0_4