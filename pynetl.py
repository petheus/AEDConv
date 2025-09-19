import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Convde(nn.Module):
    def __init__(self, filtersin, filters, dilation=(1, 1), kernel_size=[3, 3], padding='same'):
        super(Convde, self).__init__()
        
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

        

class Conv_block42(nn.Module):
    def __init__(self, filtersin, filters, imsize=None, block_nums=2, kernel_size=[3, 3]):
        super(Conv_block42, self).__init__()
        self.filt = filters
        
        self.conv0 = Convde(filtersin, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.conv0_ = Convde(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.batch0 = torch.nn.InstanceNorm2d(filters, track_running_stats=False, affine=False)
        self.batch0_ = torch.nn.InstanceNorm2d(filters, track_running_stats=False, affine=False)

        '''self.conv1 = nn.Conv2d(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.conv1_ = Convde(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.batch1 = torch.nn.InstanceNorm2d(filters, track_running_stats=False)
        self.batch1_ = torch.nn.InstanceNorm2d(filters, track_running_stats=False)'''

        '''self.conv2 = Convde(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.conv2_ = Convde(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.batch2 = torch.nn.InstanceNorm2d(filters, track_running_stats=False)
        self.batch2_ = torch.nn.InstanceNorm2d(filters, track_running_stats=False)'''

    def forward(self, inputs):
        x0 = inputs
        #x0 = F.pad(x0, (0, 1, 0, 1))
        x0 = self.conv0(x0)
        x0 = self.batch0(x0)
        x0 = torch.nn.functional.relu(x0)
        #x0 = F.interpolate(x0, size=[inputs.shape[-2], inputs.shape[-1]], mode='bilinear')
        #x0 = F.pad(x0, (1, 0, 1, 0))
        x0 = self.conv0_(x0)
        x0 = self.batch0_(x0)
        x0 = torch.nn.functional.relu(x0)
        #x0 = F.interpolate(x0, size=[inputs.shape[-2], inputs.shape[-1]], mode='bilinear')

        #x0 = F.pad(x0, (0, 2, 0, 2))
        '''x0 = self.conv1(x0)
        x0 = self.batch1(x0)
        x0 = torch.nn.functional.relu(x0)
        #x0 = F.interpolate(x0, size=[inputs.shape[-2], inputs.shape[-1]], mode='bilinear')
        #x0 = F.pad(x0, (0, 2, 0, 2))
        x0 = self.conv1_(x0)
        x0 = self.batch1_(x0)
        x0 = torch.nn.functional.relu(x0)'''

        '''x0 = self.conv2(x0)
        x0 = self.batch2(x0)
        x0 = torch.nn.functional.relu(x0)
        #x0 = F.interpolate(x0, size=[inputs.shape[-2], inputs.shape[-1]], mode='bilinear')
        #x0 = F.pad(x0, (0, 2, 0, 2))
        x0 = self.conv2_(x0)
        x0 = self.batch2_(x0)
        x0 = torch.nn.functional.relu(x0)'''
        return x0

class Unet_Encode(nn.Module):
    def __init__(self, filters):
        super(Unet_Encode, self).__init__()
        self.conv0 = Conv_block42(3, filters, imsize=512, block_nums=2)
        self.pool0 = nn.MaxPool2d(kernel_size=[2, 2], stride=2, padding=(0,0))
        self.conv1 = Conv_block42(filters, filters * 2, imsize=256, block_nums=2)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2, padding=(0,0))
        self.conv2 = Conv_block42(filters * 2, filters * 4, imsize=128, block_nums=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2, padding=(0,0))
        self.conv3 = Conv_block42(filters * 4, filters * 8, imsize=64, block_nums=2)

    def forward(self, inputs):
        x0 = inputs
        x0 = self.conv0(x0)
        
        x1 = 1 * x0
        x0 = self.pool0(x0)
        x0 = self.conv1(x0)
        x2 = 1 * x0
        x0 = self.pool1(x0)
        x0 = self.conv2(x0)
        x3 = 1 * x0
        x0 = self.pool2(x0)
        x0 = self.conv3(x0)
        x4 = 1*x0 
        return x1, x2, x3, x4


class Unet_Decode(nn.Module):
    def __init__(self, filters):
        super(Unet_Decode, self).__init__()
        self.convu1 = Conv_block42(filters * 4+filters * 8, filters * 4,imsize=128, block_nums=2)
        
        self.convu2 = Conv_block42(filters * 2+filters * 4, filters * 2,imsize=256, block_nums=2)
        '''self.convu2 = Conv_block42(filters * 4, filters * 2,imsize=256, block_nums=2)
        self.conv2 = nn.Conv2d(filters * 2, filters * 4, kernel_size=[1, 1], padding='same', dilation=(1, 1))'''
        #self.batch2 = torch.nn.InstanceNorm2d(filters * 4, track_running_stats=False)
        
        
        self.convu3 = Conv_block42(filters * 1+filters * 2, filters,imsize=512, block_nums=2)
        '''self.convu3 = Conv_block42(filters * 2, filters,imsize=512, block_nums=2)
        self.conv3 = nn.Conv2d(filters * 1, filters * 2, kernel_size=[1, 1], padding='same', dilation=(1, 1))'''

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs
        x0 = F.interpolate(x4, size=[x4.shape[-2]*2, x4.shape[-1]*2], mode='bilinear')
        x0 = torch.cat([x0, x3], dim=1)
        x3 = self.convu1(x0)
        x0 = F.interpolate(x3, size=[x3.shape[-2]*2, x3.shape[-1]*2], mode='bilinear')
        
        x0 = torch.cat([x0, x2], dim=1)
        '''x2 = self.conv2(x2)
        x0 = x0*(F.sigmoid(x2))'''
        x22 = self.convu2(x0)
        x0 = F.interpolate(x22, size=[x22.shape[-2]*2, x22.shape[-1]*2], mode='bilinear')
        x0 = torch.cat([x0, x1], dim=1)
        '''x1 = self.conv3(x1)
        x0 = x0*(F.sigmoid(x1))'''
        outputs = self.convu3(x0)

        return outputs, [x22, x3]

class Conv_block_b1(nn.Module):
    def __init__(self, filtersin, filters, imsize=None, strides=[1,1], kernel_size=[3, 3], dilation=(1, 1)):
        super(Conv_block_b1, self).__init__()
        self.conv0 = nn.Conv2d(filtersin, filters, kernel_size=kernel_size, padding='same', dilation=dilation)
        #self.conv0_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None, dilation_rate=(1, 1))
        #self.convdep0 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same')
        self.batch0 = nn.BatchNorm2d(filters, track_running_stats=False)

    def forward(self, inputs):
        x0 = inputs
        x0 = self.conv0(x0)
        #x0 = self.conv0_1(x0, training=training)
        #x0 = tf.concat([x0_0,x0_1],axis=-1)
        #x0 = self.convdep0(x0)
        x0 = self.batch0(x0)
        x0 = torch.nn.functional.gelu(x0)
        return x0

class Conv_block_c(nn.Module):
    def __init__(self, filtersin, filters, strides=[1,1], kernel_size=[3, 3], padding='same'):
        super(Conv_block_c, self).__init__()
        self.conv0 = nn.Conv2d(filtersin, filters, kernel_size=kernel_size, padding=padding, dilation=(1, 1))
        #self.conv0_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None, dilation_rate=(1, 1))
        #self.convdep0 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same')
        self.batch0 = nn.BatchNorm2d(filters, track_running_stats=False)

    def forward(self, inputs):
        x0 = inputs
        x0 = self.conv0(x0)
        #x0 = self.conv0_1(x0, training=training)
        #x0 = tf.concat([x0_0,x0_1],axis=-1)
        #x0 = self.convdep0(x0)
        x0 = self.batch0(x0)
        x0 = torch.nn.functional.gelu(x0)
        return x0

class Cenblock(nn.Module):
    def __init__(self, filters):
        super(Cenblock, self).__init__()
        self.conv0_00 = Conv_block_c(filters*1, filters)
        self.conv1_00 = Conv_block_c(filters*2, filters)
        self.conv2_00 = Conv_block_c(filters*4, filters)
        self.conv3_00 = Conv_block_c(filters*8, filters)
        
        self.conv0_0 = Conv_block_c(filters, filters)

        self.conv1_0 = Conv_block_c(filters, filters)

        self.conv2_0 = Conv_block_c(filters, filters)

        #self.conv2_1 = Conv_block_b1(filters, filters,imsize=256)

        #self.conv3_0 = Conv_block_b1(filters, filters, kernel_size=[3, 3])

        #self.conv3_1 = Conv_block_b1(filters, filters,imsize=256)

        #self.conv3_2 = Conv_block_b1(filters, filters,imsize=256)
        
        self.conv_out1 = nn.Conv2d(filters, 3, kernel_size=[1, 1], padding='same', bias=False)
        #self.conv_out2 = nn.Conv2d(filters, 3, kernel_size=[1, 1], padding='same')
        #self.conv_out3 = nn.Conv2d(filters, 3, kernel_size=[1, 1], padding='same')
        #self.conv_out4 = nn.Conv2d(filters, 3, kernel_size=[1, 1], padding='same')

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs
        x1 = self.conv0_00(x1)
        x2 = self.conv1_00(x2)
        x3 = self.conv2_00(x3)
        x4 = self.conv3_00(x4)
        
        x1 = self.conv0_0(x1)

        x2 = F.interpolate(x2, size=[x2.shape[-2]*2, x2.shape[-1]*2], mode='bilinear')
        x2 = self.conv0_0(x2)

        x3 = F.interpolate(x3, size=[x3.shape[-2]*2, x3.shape[-1]*2], mode='bilinear')
        x3 = self.conv0_0(x3)

        x3 = F.interpolate(x3, size=[x3.shape[-2]*2, x3.shape[-1]*2], mode='bilinear')
        x3 = self.conv1_0(x3)

        x4 = F.interpolate(x4, size=[x4.shape[-2]*2, x4.shape[-1]*2], mode='bilinear')
        x4 = self.conv0_0(x4)

        x4 = F.interpolate(x4, size=[x4.shape[-2]*2, x4.shape[-1]*2], mode='bilinear')
        x4 = self.conv1_0(x4)

        x4 = F.interpolate(x4, size=[x4.shape[-2]*2, x4.shape[-1]*2], mode='bilinear')
        x4 = self.conv2_0(x4)

        outputs1 = self.conv_out1(x1)
        outputs2 = self.conv_out1(x2)
        outputs3 = self.conv_out1(x3)
        outputs4 = self.conv_out1(x4)
        #outputs = self.batch0(outputs)
        #outputs = tf.nn.gelu(outputs)
        #outputs = self.conv_out1(outputs)
        outputs = torch.cat([outputs4, outputs3, outputs2, outputs1], dim=1)
        return F.sigmoid(outputs)

'''class ConnectUnet(nn.Module):
    def __init__(self):
        super(ConnectUnet, self).__init__()
        self.dt = 1
        filters = 64
        self.unet_decode0 = Unet_Decode(filters)
        self.unet_encode0 = Unet_Encode(filters)
        if self.dt:
            self.cenk = Cenblock(filters)
        self.conv_out0 = nn.Conv2d(64, 3, kernel_size=[1, 1], padding='same', bias=False)
    def cenkoff(self):
        self.dt = 0

    def cenkon(self):
        self.dt = 1

    def forward(self, inputs):
        inputs = inputs
        x0 = self.unet_encode0(inputs)
        if self.dt:
            output_seg1 = self.cenk([x0[0],x0[1],x0[2],x0[3]])
        x0, xout = self.unet_decode0(x0)
        #output_seg1 = self.cenk(xout)
        #x01 = tf.math.exp(x0[:,:,:,:64])
        #x0 = x0[:,:,:,64:]*x01
        output_seg0 = self.conv_out0(x0)
        output_seg0 = F.sigmoid(output_seg0)
        if self.dt:
            return [output_seg1[:,:3,:,:], output_seg1[:,3:6,:,:], output_seg1[:,6:9,:,:], output_seg1[:,9:,:,:], output_seg0], x0
        else:
            return [output_seg0], x0'''

class ConnectUnet(nn.Module):
    def __init__(self):
        super(ConnectUnet, self).__init__()
        filters = 64
        self.unet_decode0 = Unet_Decode(filters)
        self.unet_encode0 = Unet_Encode(filters)
        self.conv_out0 = nn.Conv2d(filters, 3, kernel_size=[1, 1], padding='same')

    def forward(self, inputs):
        inputs = inputs
        
        x0 = self.unet_encode0(inputs)
        x0, xout = self.unet_decode0(x0)
        #output_seg1 = self.cenk(xout)
        #x01 = tf.math.exp(x0[:,:,:,:64])
        #x0 = x0[:,:,:,64:]*x01
        output_seg0 = self.conv_out0(x0)
        
        output_seg0 = F.sigmoid(output_seg0)
        '''with open('output.txt', 'w') as f:
            print(output_seg0, file=f)'''
        return [output_seg0], x0