python train.py --dataroot ../../data/foot --name sc2_batch --gpu_ids 1,2,3 --loadSize 256 --fineSize 256 --which_model_netG unet_256 --batchSize 1 --norm batch --save_epoch_freq 1 --display_port 8098 --which_direction AtoB --dataset_mode aligned --no_lsgan --pool_size 0 --model pix2pix
