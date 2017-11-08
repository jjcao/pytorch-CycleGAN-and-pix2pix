python train.py --dataroot ../../data/foot --name sc2_batch8 --batchSize 8 --gpu_ids 2 --save_epoch_freq 5 --display_port 8097 --loadSize 256 --fineSize 256 --which_model_netG unet_256 --dataset_mode aligned --no_lsgan --pool_size 0 --model pix2pix -lr 0.002 --niter 10
# --continue_train --which_epoch 100 --epoch_count 105
