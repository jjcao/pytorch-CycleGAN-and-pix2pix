import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import pdb; pdb.set_trace()
        
opt = TrainOptions().parse()

# data for training
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# data for test
from copy import deepcopy 
test_opt = deepcopy(opt)
test_opt.phase = 'test'
test_opt.nThreads = 1   # test code only supports nThreads = 1
test_opt.batchSize = 1  # test code only supports batchSize = 1
test_opt.serial_batches = True  # no shuffle
test_opt.no_flip = True  # no flip
test_data_loader = CreateDataLoader(test_opt)
test_dataset = test_data_loader.load_data()
previous_score = 0.0


#########
model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

#########
import matlab.engine
eng = matlab.engine.start_matlab()

# create website
import os
from util import html
test_opt.results_dir = './results/'
web_dir = os.path.join(test_opt.results_dir, test_opt.name)
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s' % (test_opt.name, test_opt.phase))

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
            for j, test_data in enumerate(test_dataset):
                model.set_input(test_data)
                model.test()
                visuals = model.get_current_visuals()
                img_path = model.get_image_paths()
                print('process image... %s' % img_path)
                visualizer.save_images(webpage, visuals, img_path)
            
            score = eng.compute_metric(opt.dataroot)
            if score > previous_score:
                previous_score = score
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('%d-%d' % (epoch, i))

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

eng.quit()