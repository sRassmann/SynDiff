EXP=exp_syndiff_bs_96
N_EPOCH=20000

# Note that here one epoch means taking one slice from each volume / subject
torchrun --nproc_per_node=8 train_paired.py --exp $EXP --num_channels_dae 64 \
 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --num_epoch $N_EPOCH --ngf 64 \
 --embedding_type positional --use_ema --ema_decay 0.999 --save_ckpt_every 200 --r1_gamma 1. --z_emb_dim 256 \
 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --save_content --output_path output --batch_size 12

CUDA_VISIBLE_DEVICES=0 python test.py --exp $EXP --num_channels 2 --num_channels_dae 64 \
 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --ngf 64 --embedding_type positional \
 --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --batch_size=12 --output_path output --which_epoch=$N_EPOCH

CUDA_VISIBLE_DEVICES=0 python test.py --exp $EXP --num_channels 2 --num_channels_dae 64 \
 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --ngf 64 --embedding_type positional \
 --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --batch_size=12 --output_path output --which_epoch=$N_EPOCH  \
 --out_dir_name inference_wmh --dataset_json ../data/RS/RS_wmh_test.json --data_dir ../data/RS/conformed_test

python flairsyn/metrics_3d.py -s output/$EXP/inference

