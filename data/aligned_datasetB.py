import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class AlignedDatasetB(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        #ratio = [self.opt.loadSize * 2.0/AB.size[0], float(self.opt.loadSize)/AB.size[1]]
        AB_origi_size = AB.size
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
    
        ## get image box coordinate
        Box = self.get_box_coordinate(AB_path, AB_origi_size)
        Box = torch.from_numpy(Box)  # A Tensor

        ##add Box
        return {'A': A, 'B': B, 'Box': Box,
                'A_paths': AB_path, 'B_paths': AB_path, 'origi_size': AB_origi_size}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDatasetB'
    
    ##get  image box coordinate  ## cch
    def get_box_coordinate(self, path, AB_origi_size):
        s = '/'
        orimage_name = path[path.rfind(s) + 1:-4] # a string
        #orimage_name='1_2_3_4_5_6_7_8'
        box  = orimage_name.split('_')        
            
        X_ = box[::2]
        Y_ = box[1::2]
              
        box[::2] = [(float(x)*2 / AB_origi_size[0]) for x in X_]
        box[1::2] = [(float(y) / AB_origi_size[1]) for y in Y_]

        if len(box) < 8:
            box = [0.5, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.5]
            
        Box = np.asarray(box)
        return Box
