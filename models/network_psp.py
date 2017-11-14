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
class PspNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(PspNetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        
        n_downsampling = 2
        self.netFeat = ResnetGeneratorP2p(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                                          n_blocks=9, gpu_ids=gpu_ids,n_downsampling = n_downsampling)      
        #model = [self.netFeat]
        self.pool1 = PspPooling(self.netFeat.output_nc, self.netFeat.output_nc/4.0, 64,norm_layer)
        self.pool2 = PspPooling(self.netFeat.output_nc, self.netFeat.output_nc/4.0, 32,norm_layer)
        self.pool4 = PspPooling(self.netFeat.output_nc, self.netFeat.output_nc/4.0, 16,norm_layer)
        self.pool8 = PspPooling(self.netFeat.output_nc, self.netFeat.output_nc/4.0, 8,norm_layer)
        
        mult = 2**n_downsampling
        self.final = [nn.Conv2d(self.netFeat.output_nc*2, ngf*mult, 3, padding=1, bias=False)]       
        self.final += [norm_layer(ngf*mult, momentum=.95), nn.ReLU(inplace=True), nn.Dropout(.1)]
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            self.final += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            
        self.final += [nn.ReflectionPad2d(3)]
        self.final += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.final += [nn.Tanh()]
        

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

class PspPooling(nn.Module):

    def __init__(self, in_features, out_features, downsize, upsize=64, norm_layer=nn.BatchNorm2d):
        super(PspPooling,self).__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(downsize, stride=downsize),
            nn.Conv2d(in_features, out_features, 1, bias=False),
            norm_layer(out_features, momentum=.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(upsize)
        )

    def forward(self, x):
        return self.features(x)
    
# Defines the generator that consists of Resnet blocks followed a few downsampling operations.
# Code and idea originally from pix2pix paper
class ResnetGeneratorP2p(nn.Module):
    # n_downsampling = 2 for image size 256 and 128. should be increased for 512, 1024
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, 
                 n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling = 2 ):
        assert(n_blocks >= 0)
        super(ResnetGeneratorP2p, self).__init__()
        self.input_nc = input_nc
        
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf, momentum=.95),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2, momentum=.95),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        self.output_nc = ngf * mult
        for i in range(n_blocks):
            model += [ResnetBlockP2p(self.output_nc, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
        
# Define a resnet block. It is just a copy from pix2pix paper.
class ResnetBlockP2p(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlockP2p, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
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
                       norm_layer(dim, momentum=.95),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

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
                       norm_layer(dim, momentum=.95)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out        