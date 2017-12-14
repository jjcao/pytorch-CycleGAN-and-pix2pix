import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import os
import torch
import pdb; pdb.set_trace()

# 放到最前面，好使，放到TrainOptions().parse()后边，不好使。
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
#print("Let's use", torch.cuda.device_count(), "GPUs!")
opt = TrainOptions().parse()
#os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu_ids)
#print("Let's use", torch.cuda.device_count(), "GPUs!")
#torch.manual_seed(1337)
cuda = torch.cuda.is_available()
if cuda:
    #torch.cuda.manual_seed(1337)
    print("cuda devices: {} are ready".format(opt.gpu_ids))
    
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
tmp = dataset.dataset.__getitem__(0)
#tmp['Box']
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
        
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
