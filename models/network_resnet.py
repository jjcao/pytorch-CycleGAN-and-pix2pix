#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:16:43 2017

@author: jjcao
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import functools
from .base_network import BoxBlock, UBoxBlock

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class Resnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
                 drop_rate=0.5, n_blocks=9, padding_type='reflect',
                 n_downsampling = 2, fine_size = 256):
        assert(n_blocks >= 0)
        super(Resnet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #############
        #############
        model_head = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        for i in range(n_downsampling):
            mult = 2**i
            model_head += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model_head += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, drop_rate=drop_rate, use_bias=use_bias)]
        self.model_head = nn.Sequential(*model_head)

        #############
        # box
        #############
        im_size = fine_size // 2**n_downsampling 
        self.model_B = BoxBlock(ngf * mult, ngf, norm_layer=norm_layer, use_bias = use_bias,
                                im_size = im_size, drop_rate=drop_rate)
        
        #############
        #############
        model_tail = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_tail += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model_tail += [nn.ReflectionPad2d(3)]
        model_tail += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_tail += [nn.Tanh()]

        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, input):
        out = self.model_head(input)
        box = self.model_B(out)
        out = self.model_tail(out)
        return [out, box]


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, drop_rate, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, drop_rate, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, drop_rate, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if drop_rate:
            conv_block += [nn.Dropout(drop_rate)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out