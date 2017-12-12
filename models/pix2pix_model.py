import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.fine_size = float(opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, opt.drop_rate, 
                                      opt.init_type, self.gpu_ids, opt.fineSize)
        self.output_nc = opt.output_nc
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
   
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor) 

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        
        self.input_Box = None
        self.origin_im_size = None
        
        #jjcao
        if 'Box' in input.keys():
            self.input_Box = input['Box']
            self.origin_im_size = input['origi_size']        
        
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B) 
        #self.fake_B = self.netG.forward(self.real_A) 
        tmp = self.netG.module.model_head.forward(self.real_A) 
        self.fake_B = self.netG.module.model_tail.forward(tmp)     
        self.pred_Box = self.netG.module.model_B.forward(tmp)
        
        #jjcao
        if self.origin_im_size:
            self.input_Box = self.input_Box.cuda().float()
            if self.gpu_ids: 
                self.input_Box = self.input_Box.cuda()  
            self.input_Box = Variable(self.input_Box) 
         

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        # jjcao
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)
        
        # jjcao
        if self.origin_im_size:
            self.input_Box = Variable(self.input_Box, volatile=True)  
            
        loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        return loss_G_L1

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.module.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.module.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.module.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

#        m5 = self.netG.model.model[8].model[9]     
#        m3 = m5.model[9].model[9]
#        m1 = m3.model[9].model[9]
        
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        if self.origin_im_size:
            #torch.nn.L1Loss()
            self.loss_box = torch.nn.MSELoss()(self.pred_Box, self.input_Box) * self.opt.lambda_A * 0.5
            self.loss_G = self.loss_G_L1 + self.loss_box + self.loss_G_GAN
        else:
            self.loss_G = self.loss_G_L1 + self.loss_G_GAN
            
        #self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * 0.5 * self.opt.lambda_A
        #self.loss_G1_L1 = self.criterionL1(m1.output1, self.real_B) * 0.5 * self.opt.lambda_A
        #self.loss_G3_L1 = self.criterionL1(m3.output1, self.real_B) * 0.5 * self.opt.lambda_A
        #self.loss_G5_L1 = self.criterionL1(m5.output1, self.real_B) * 0.5 * self.opt.lambda_A

        #self.loss_G = self.loss_G_GAN + self.loss_G_L1 # original
        #self.loss_G = self.loss_G_L1 # jjcao
        #self.loss_G = self.loss_G_L1 + self.loss_G1_L1 + self.loss_G3_L1 + self.loss_G5_L1# jjcao

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        #jjcao
        if self.origin_im_size:
            return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                                ('G_L1', self.loss_G_L1.data[0]),
                                ('G_Box', self.loss_box.data[0]),
                                ('D_real', self.loss_D_real.data[0]),
                                ('D_fake', self.loss_D_fake.data[0]),
                                #('loss_G1_L1', self.loss_G1_L1.data[0]),
                                #('loss_G3_L1', self.loss_G3_L1.data[0]),
                                #('loss_G5_L1', self.loss_G5_L1.data[0])
                                ])
        else:
            return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                                ('G_L1', self.loss_G_L1.data[0]),
                                ('G_Box', 0.0),
                                ('D_real', self.loss_D_real.data[0]),
                                ('D_fake', self.loss_D_fake.data[0]),
                                #('loss_G1_L1', self.loss_G1_L1.data[0]),
                                #('loss_G3_L1', self.loss_G3_L1.data[0]),
                                #('loss_G5_L1', self.loss_G5_L1.data[0])
                                ])

        
    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        
        #jjcao
        if self.origin_im_size: 
            #origin_im_size = [float(self.origin_im_size[0].numpy()[0]), float(self.origin_im_size[1].numpy()[0])]
            #image_tensor[0].cpu().float().numpy() 
            #self.input_Box[0].cpu().data.numpy() 
            input_box = self.input_Box.data[0].cpu().float().numpy()       
            input_box = input_box * self.fine_size          
            util.draw_2lines(real_B, input_box, [self.fine_size,self.fine_size])

           
            pred_Box = self.pred_Box.data[0].cpu().float().numpy()        
            pred_Box = pred_Box * self.fine_size
            util.draw_2lines(fake_B, pred_Box, [self.fine_size,self.fine_size])
            
            print(input_box)
            print(pred_Box)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
