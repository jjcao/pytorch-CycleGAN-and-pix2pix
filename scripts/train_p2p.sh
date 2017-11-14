python train.py --dataroot ../../data/sc/sc_p2p --name sc2_batch8 --batchSize 8 --gpu_ids 2 --save_epoch_freq 5 --display_port 8097 --loadSize 256 --fineSize 256 --which_model_netG unet_256 --dataset_mode aligned --no_lsgan --pool_size 0 --model pix2pix --lr 0.001 --lr_policy lambda --niter 5
# --continue_train --which_epoch 100 --epoch_count 105
# --lr 0.001 --lr_policy lambda --niter 5
# --lr 0.002 --lr_policy step --lr_decay_iters 10