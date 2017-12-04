#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:43:23 2017

@author: jjcao
"""
import torch
import torch.nn as nn
import functools

#
class UnetBGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetBGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        
        # construct unet structure
        planes = [ngf, ngf*2, ngf*4, ngf*8]
        
        self.down = [UnetDown(input_nc, output_nc=planes[0], norm_layer=norm_layer)]
        self.down += [UnetDown(planes[0], output_nc=planes[1], norm_layer=norm_layer)]
        self.down += [UnetDown(planes[1], output_nc=planes[2], norm_layer=norm_layer)]
        self.down += [UnetDown(planes[2], output_nc=planes[3], norm_layer=norm_layer)]
        for i in range(num_downs - 5):
            self.down += [UnetDown(planes[3], output_nc=planes[3], norm_layer=norm_layer, use_dropout=use_dropout)]
        
        self.center = [UnetCenter(planes[3], output_nc=planes[3])]
        
        self.up = []
        for i in range(num_downs - 5):
            self.up += [UnetUp(planes[3], output_nc=planes[3], norm_layer=norm_layer, use_dropout=use_dropout)]
        self.up += [UnetUp(planes[3], output_nc=planes[2], norm_layer=norm_layer)]
        self.up += [UnetUp(planes[2], output_nc=planes[1], norm_layer=norm_layer)]
        self.up += [UnetUp(planes[1], output_nc=planes[0], norm_layer=norm_layer)]
        self.up += [UnetUp(planes[0], output_nc=output_nc, norm_layer=norm_layer, outermost=True)]
        
        self.down = nn.Sequential(*self.down)
#        self.model = self.down + self.center + self.up
#        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.netFeat.forward(input)
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            p1 = nn.parallel.data_parallel(self.pool1, x, self.gpu_ids)
            p2 = nn.parallel.data_parallel(self.pool2, x, self.gpu_ids)
            p4 = nn.parallel.data_parallel(self.pool4, x, self.gpu_ids)
            p8 = nn.parallel.data_parallel(self.pool8, x, self.gpu_ids)
            out = torch.cat([x,p1, p2, p4, p8], 1)
            return nn.parallel.data_parallel(self.final, out, self.gpu_ids)
        else:
            p1 = self.pool1(x)
            p2 = self.pool2(x)
            p4 = self.pool4(x)
            p8 = self.pool8(x)
            out = torch.cat([x,p1, p2, p4, p8], 1)
            return self.final(out)
        
                
class UnetCenter(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetCenter, self).__init__()

    def forward(self, inputs):



class UnetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetDown, self).__init__()

    def forward(self, inputs):


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(UnetUp, self).__init__()

    def forward(self, inputs1, inputs2):
        
