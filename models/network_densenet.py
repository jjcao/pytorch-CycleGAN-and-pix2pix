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

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,norm_layer=nn.BatchNorm2d):
        super(_DenseLayer, self).__init__()
        
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm.1', norm_layer(bn_size * growth_rate))
        self.add_module('relu.1', nn.ReLU(inplace=True))

             
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False))                               
        self.add_module('norm.2', norm_layer(growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace=True))
        
        self.add_module('drop.2', nn.Dropout(drop_rate))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        #new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate,norm_layer=nn.BatchNorm2d):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, 
                                bn_size, drop_rate,norm_layer=norm_layer)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_layer=nn.BatchNorm2d):
        super(_Transition, self).__init__()
        
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=2, bias=False))
        self.add_module('norm', norm_layer(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        
        #self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
              
#
class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block. 0: use nn.Conv2d(kernel_size=3, stride=2)
        ngf (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
                 drop_rate=0.5, growth_rate=32, block_config=(0, 0, 9),
                 bn_size=4):

        super(DenseNet, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.model_head = nn.Sequential(OrderedDict([
            ('reflect0', nn.ReflectionPad2d(3)),
            ('conv70', nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,bias=use_bias)),
            ('norm0', norm_layer(ngf)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
    
    
        # downsampling 0.5 via conv2D or denseblock + transition
        num_features = ngf
        for i, num_layers in enumerate(block_config):
            if num_layers == 0: # downsampling 0.5
                block = nn.Sequential(OrderedDict([
                    ('conv%d'%(i+1), nn.Conv2d(num_features, num_features, kernel_size=3,
                                               stride=2, padding=1, bias=use_bias)),
                    ('norm%d'%(i+1), norm_layer(ngf)),
                    ('relu%d'%(i+1), nn.ReLU(True)),
                    ]))
                
                self.model_head.add_module('convblock%d'%(i+1), block)
                                           
            else: # downsampling 0.5
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.model_head.add_module('denseblock%d'%(i+1), block)
                num_features = num_features + num_layers * growth_rate
                
                if i != len(block_config) - 1:
                    block = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                    self.model_head.add_module('transition%d'%(i+1), block)
                    num_features = num_features // 2
         
        #####
        self.model_tail = nn.Sequential()
        n_downsampling = len(block_config)-1
        for i in range(n_downsampling):
            #mult = 2**(n_downsampling - i)
            block = nn.Sequential(OrderedDict([
                    ('convtrans%d'%i, nn.ConvTranspose2d(num_features, num_features // 2,
                                         kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias)),
                    ('norm%d'%i, norm_layer(num_features // 2)),
                    ('relu%d'%i, nn.ReLU(True)),
                    ]))
            num_features = num_features // 2        
            self.model_tail.add_module('convtblock%d'%i, block)
        
        block = nn.Sequential(OrderedDict([
                ('reflect1', nn.ReflectionPad2d(3)),
                ('conv71', nn.Conv2d(num_features, output_nc, kernel_size=7, padding=0)),
                ('tanh1', nn.Tanh()),
            ]))
        self.model_tail.add_module('conv71', block)

    def forward(self, x):
        out = self.model_head(x)
        out = self.model_tail(out)
        return out
    
    