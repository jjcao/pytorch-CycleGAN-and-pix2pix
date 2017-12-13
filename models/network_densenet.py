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

#
# In original paper, it is BatchNorm2d, ReLU, Conv2d without bias, 
# BatchNorm2d, ReLU, Conv2d without bias, Dropout
#
# Now, it is Conv2d, norm, ReLU, Conv2d, Dropout, norm, ReLU.
# if norm=BatchNorm2d then Conv2d without bias; if norm=InstanceNorm2d, 
# Conv2d with bias as used in the pixel 2 pixel paper
#
class _DenseLayer2(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate, norm_layer, use_bias, bn_size=4):
        super(_DenseLayer2, self).__init__()
        
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=use_bias))
        self.add_module('norm.1', norm_layer(bn_size * growth_rate))
        self.add_module('relu.1', nn.ReLU(inplace=True))

             
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=use_bias))      
        self.add_module('drop.2', nn.Dropout(drop_rate))                         
        self.add_module('norm.2', norm_layer(growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace=True))

    def forward(self, x):
        new_features = super(_DenseLayer2, self).forward(x)
        return torch.cat([x, new_features], 1)

# in One Hundred Layers Tiramisu, it is BatchNorm2d, ReLU, Conv2d without bias, Dropout
class _DenseLayer1(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate, norm_layer, use_bias):
        super(_DenseLayer1, self).__init__()
             
        self.add_module('conv.1', nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=use_bias))   
        self.add_module('drop.1', nn.Dropout(drop_rate))                            
        self.add_module('norm.1', norm_layer(growth_rate))
        self.add_module('relu.1', nn.ReLU(inplace=True))        
        
    def forward(self, x):
        new_features = super(_DenseLayer1, self).forward(x)
        return torch.cat([x, new_features], 1)    

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, 
                 drop_rate, norm_layer, use_bias, Layer = _DenseLayer2):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = Layer(num_input_features + i * growth_rate, growth_rate, 
                                drop_rate, norm_layer, use_bias)
            self.add_module('denselayer%d' % (i + 1), layer) 
        
#
# In original paper, it is called Transition, composed of BatchNorm2d, ReLU, Conv2d without bias, 2*2 max Pooling
# In One Hundred Layers Tiramisu, it is called TransitionDown, 
# composed of BatchNorm2d, ReLU, Conv2d without bias, Dropout, 2*2 max Pooling        
class _TransitionDown(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate, norm_layer, use_bias):
        super(_TransitionDown, self).__init__()
        
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=use_bias))
        self.add_module('drop', nn.Dropout(drop_rate))
        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))
        self.add_module('norm', norm_layer(num_output_features//2))
        self.add_module('relu', nn.ReLU(inplace=True))
        
class _TransitionUp(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionUp, self).__init__()
        
        self.add_module('convtrans', nn.ConvTranspose2d(num_input_features, num_output_features,
                                         kernel_size=3, stride=2,padding=1, output_padding=1,bias=False))
#    def forward(self, block_to_upsample):    
#        x = block_to_upsample[:,1:16,:,:]
#        new_features = super(_TransitionUp, self).forward(x)
#        return torch.cat([x, new_features], 1)  
        
#
class DenseNet(nn.Module):
    """Densenet-BC model class, based on
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
                 bn_size=4, compression=0.5):

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
                block = _DenseBlock(num_layers, num_features, growth_rate, drop_rate, norm_layer, use_bias)
                
                self.model_head.add_module('denseblock%d'%(i+1), block)
                num_features = num_features + num_layers * growth_rate
                
                if i != len(block_config) - 1:
                    block = _TransitionDown(num_features, int(num_features * compression),
                                            drop_rate, norm_layer, use_bias)
                    self.model_head.add_module('transition%d'%(i+1), block)
                    num_features = int(num_features * compression)
         
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

# "train with RMSprop [33], with an initial learning rate of 1e − 3 and 
# an exponential decay of 0.995 after each epoch." 
# It is 367 per epoch, 360*480 => 224*224 for initial train 
# 224, 112, 56, 28, 14, 7
#
#    
# init train: 256, 128, 64, 32, 16, 8  
# finetune: 512, 256, 128, 64, 32, 16 
# HeUniform: to do
class DenseUet(nn.Module):  
    def __init__(self, input_nc, output_nc, ngf=48, norm_layer=nn.BatchNorm2d, 
                 drop_rate=0.2, growth_rate=16, n_layers_per_block=(4, 5, 7, 10, 12, 15)):

        super(DenseNet, self).__init__()
        
        self.growth_rate = growth_rate
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
         
        # 1st conv
        self.model_head = nn.Sequential()
        block = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias)),
                ('norm1', norm_layer(ngf)),
                ('relu1', nn.ReLU(True))
            ]))
        self.model_head.add_module('conv1', block)

        # Downsampling path + bottleneck#
        num_features = ngf
        n_skip_connection_planes = []
        for i in range(len(n_layers_per_block)):
            # Dense Block
            block = _DenseBlock(n_layers_per_block[i], num_features, growth_rate, drop_rate, norm_layer, use_bias)
            self.model_head.add_module('denseblock%d'%(i+1), block)
            #skip_connection_list.append(block)
            
            num_features = num_features + n_layers_per_block[i] * growth_rate
            n_skip_connection_planes[i] = num_features
            # 48 => 48 + 4*16 = 112
            # 112=>112 + 5*16 = 192, ...
            
            if i != len(n_layers_per_block) - 1:
                block = _TransitionDown(num_features, num_features,
                                        drop_rate, norm_layer, use_bias)
                self.model_head.add_module('TD%d'%(i+1), block)
                
        #self.n_planes_bottleneck = n_layers_per_block[-1] * growth_rate # 15*16 = 240
        #num_features = num_features - self.n_planes_bottleneck # 896 - 15*16 = 656
        
        # Upsampling path #
        self.model_tail = nn.Sequential()
        n_layers_per_block = n_layers_per_block[::-1]
        self.n_layers_per_block = n_layers_per_block      
        n_skip_connection_planes = n_skip_connection_planes[::-1]
        
        for i in range(len(n_layers_per_block))[1:]:
            num_features = n_layers_per_block[i-1] * growth_rate
            block = _TransitionUp(num_features, num_features)
            self.model_tail.add_module('TU%d'%(i+1), block)
            
            
            block = _DenseBlock(n_layers_per_block[i], n_skip_connection_planes[i]+num_features, 
                                growth_rate, drop_rate, norm_layer, use_bias)
            # 656 + 15*16 => 656 + 15*16 + 12*16 = 1088
            # 464 + 15*16 => 464 + 12*16 + 10*16 = 816
            # ...
            self.model_tail.add_module('denseblock%d'%(i+1), block)
       
        # final
        block = nn.Sequential(OrderedDict([
                ('conv2', nn.Conv2d(n_skip_connection_planes[i]+num_features, output_nc, 
                                    kernel_size=1, stride=1, padding=1, bias=use_bias)),
                ('tanh2', nn.Tanh()),
            ]))
        self.model_tail.add_module('conv2', block)             
  
    def forward(self, x):
        x = self.model_head[0](x)#.__getitem__(0)
        
        skip_connection_list = []
        for name, module in self.model_head.named_modules():
            if 'denseblock' in name:
                x = module(x)
                skip_connection_list.append(x)
            elif 'TD' in name:
                x = module(x)
        
        i = 0
        skip_connection_list = skip_connection_list[1::-1]
        for name, module in self.model_tail.named_modules():
            if 'TU' in name:  
                n_planes_need = self.n_layers_per_block[i] * self.growth_rate
                x = x[:,n_planes_need-1:,:,:]
                x = module(x)
                x = torch.cat([skip_connection_list[i], x], 1)
                i = i + 1
            else:
                x = module(x)
                    
        return x    
   