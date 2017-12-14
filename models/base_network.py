#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:43:23 2017

@author: jjcao
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import functools

class BoxBlock(nn.Module):
    """it is a net without name and it is for box regression.
    It is for Resnet, Densnet, etc
    """
    def __init__(self, input_nc, ngf, norm_layer, use_bias,
                 im_size = 64, n_downsampling = 2, drop_rate=0.2):

        super(BoxBlock, self).__init__()

        #################
        # img size: 64=>32, # 32 => 16 
        num_input_features = input_nc 
        num_output_features = ngf
        model_head = []
        for i in range(n_downsampling):        
            model_head += [nn.Conv2d(num_input_features, num_output_features, 
                                     kernel_size=3, stride=2, padding=1, bias=use_bias),
                               norm_layer(num_output_features), nn.ReLU(True),]
            num_input_features = num_output_features
            num_output_features = num_output_features // 2
        self.model_head = nn.Sequential(*model_head) 
        
        
        # 16*16*32 = 8192   
        im_size = im_size // 2**n_downsampling
        
        model_tail = [nn.Linear(im_size*im_size*num_input_features, 512), nn.Tanh(),
                      nn.Linear(512, 64), nn.Tanh(),
                      nn.Linear(64, 8), #nn.Sigmoid(), # sigmoid means positive, No need activation？
                      ]#nn.Tanh(), nn.ReLU(True)

        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        y = self.model_head(x)
        y = y.view(y.size(0), -1)
        box = self.model_tail(y)
        return box

class UBoxBlock(nn.Module):
    """it is a net without name and it is for box regression.
    It is for unet like newtork
    """
    def __init__(self, input_nc, ngf, norm_layer, use_bias, im_size, drop_rate=0.2):
        super(UBoxBlock, self).__init__()
        
        self.model_head = nn.Sequential()      
        # 8*8*896 => 8*8*ngf 
        block = nn.Sequential(OrderedDict([
                ('bh_conv1', nn.Conv2d(input_nc, ngf, kernel_size=1, stride=1, bias=use_bias)),
                ('bh_drop', nn.Dropout(drop_rate)),
                ('bh_norm1', norm_layer(ngf)),
                ('bh_relu1', nn.ReLU(True))
            ]))
        self.model_head.add_module('box_head', block) 

        self.model_tail = nn.Sequential()  
        # 8*8*ngf(48) = 3072 => 512       
        block = nn.Sequential(OrderedDict([
                ('bt_linear1', nn.Linear(im_size*im_size*ngf, 512)),
                ('bt_tanh1', nn.Tanh()),
                ('bt_linear2', nn.Linear(512, 64)),
                ('bt_tanh2', nn.Tanh()),
                ('bt_linear3', nn.Linear(64, 8)),
            ]))     
        self.model_tail.add_module('box_tail', block) 
    
    def forward(self, x):
        y = self.model_head(x)
        y = y.view(y.size(0), -1)
        box = self.model_tail(y)
        return box