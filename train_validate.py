import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
        
opt = TrainOptions().parse()

# data for training
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# data for validate
from copy import deepcopy 
val_opt = deepcopy(opt)
val_opt.phase = 'test' # val'
val_opt.nThreads = 1   # val code only supports nThreads = 1
val_opt.batchSize = 1  # val code only supports batchSize = 1
val_opt.serial_batches = True  # no shuffle
val_opt.no_flip = True  # no flip
val_data_loader = CreateDataLoader(val_opt)
val_dataset = val_data_loader.load_data()
previous_loss = 999999999.0


#########
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

        #validate
        if total_steps % opt.save_latest_freq == 0:
            loss = 0.0
            for j, val_data in enumerate(val_dataset):
                model.set_input(val_data)
                loss += model.test()                   
                #img_path = model.get_image_paths()
                #print('process image... %s' % img_path)
            loss = loss/len(val_dataset)
            import pdb; pdb.set_trace()
            model.update_learning_rate(loss)
            if loss < previous_loss:
                previous_loss = loss
                print('saving model (epoch %d, total_steps %d, loss %f)' %
                      (epoch, total_steps, loss))
                model.save('%d-%d-%f' % (epoch, i, loss))

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    
