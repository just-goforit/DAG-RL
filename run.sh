## S-IO
python main.py --max_episode_len 8192 --op_config 1 7 7 2 3 2 12 --reward_scale 0.5 \
               --lr 1e-3 1e-4 --total_timesteps 5e7 --n_steps 8192 --ncpu 8 --batch_size 1024 \
               --model GNN --ortho_init --gnn_class SAGE --features_dim 8 --node_wise \
               --topk_layer 3 --focus_num 256 --focus_vf --pi_conv_out 4 --vf_conv_out 1 \
               --tb_log_dir exp/log --save_path exp/saved_model --agent_name 17232s12_f256 \
               --sp_tensor --device cuda:1 --save_freq 0.25

# mlp
python main.py --max_episode_len 8192 --op_config 1 7 7 2 3 2 12 --reward_scale 0.5 \
                --lr 1e-3 1e-5 --total_timesteps 1e8 --n_steps 8192 --ncpu 8 --batch_size 1024 \
                --model MLP --ortho_init --features_dim 1024 --mlp_hid_net_arch 512 \
                --pi_net_arch 1024 --vf_net_arch 512 \
                --tb_log_dir exp/log --save_path exp/saved_model \
                --eval_mode --agent_name 03-20_21-03-08-927_17232s12_mlp_25000000_steps \
                --device cuda:3 --save_freq 0.25
## N-IO
#   base 1,4,4,1 3,3,1
#   dim -> H&W img=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  f64->96
# 2e5 8e4 1e5
# (checked)
python main.py --max_episode_len 128 --op_config 1 4 4 1 3 1 32 \
               --lr 3e-4 --total_timesteps 2e5 --n_steps 128 --ncpu 16 --batch_size 512 \
               --model MLP --ortho_init --features_dim 64 --pi_net_arch 64 --vf_net_arch 32 \
               --tb_log_dir exp/log --save_path exp/saved_model --agent_name 14131s32_mlp \
               --device cuda:7 --save_freq 1.0
python main.py --max_episode_len 256 --op_config 1 5 5 1 3 1 32 --reward_scale 0.5 \
               --lr 3e-4 --total_timesteps 1e6 --n_steps 256 --ncpu 16 --batch_size 1024 \
               --model MLP --ortho_init --features_dim 128 --pi_net_arch 128 --vf_net_arch 64 \
               --tb_log_dir exp/log --save_path exp/saved_model --agent_name 15131s32_mlp \
               --device cuda:0 --save_freq 1.0
python main.py --max_episode_len 512 --op_config 1 6 6 1 3 1 32 --reward_scale 0.5 \
               --lr 2e-4 --total_timesteps 1e7 --n_steps 512 --ncpu 16 --batch_size 2048 \
               --model MLP --ortho_init --features_dim 256 --pi_net_arch 256 --vf_net_arch 128 \
               --tb_log_dir exp/log --save_path exp/saved_model --agent_name 16131s32_mlp_lr_bs \
               --device cuda:5 --save_freq 0.2
