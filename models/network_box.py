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

class BoxnetGenerator(nn.Module):
    """it is a net without name and it is for box regression.
    """
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
                 drop_rate=0.5, fine_size = 256, n_downsampling = 2):

        super(BoxnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 64=>32, # 32 => 16 
        num_input_features = input_nc 
        num_output_features = ngf
        model_head = []
        for i in range(2):        
            model_head += [nn.Conv2d(num_input_features, num_output_features, kernel_size=3, stride=2, padding=1, bias=use_bias),
                               norm_layer(num_output_features), nn.ReLU(True),]
            n_downsampling += 1
            num_input_features = num_output_features
            num_output_features = num_output_features // 2
        self.model_head = nn.Sequential(*model_head) 
        
        
        # 16*16*32 = 8192   
        imsize = fine_size//2**n_downsampling
        model_tail = [nn.Linear(imsize*imsize*num_input_features, 512), nn.Tanh(),
                      nn.Linear(512, 64), nn.Tanh(),
                      nn.Linear(64, 8), #nn.Sigmoid(), # sigmoid means positive, No need activation？
                      ]#nn.Tanh(), nn.ReLU(True)

        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        y = self.model_head(x)
        y = y.view(y.size(0), -1)
        box = self.model_tail(y)
        return box
    
    