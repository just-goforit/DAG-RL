import argparse
from colorama import Fore, Style

def args_checker(args):
    # total timesteps type
    args.total_timesteps = int(args.total_timesteps)
    # learning rate check
    assert len(args.lr) <= 2 and Fore.RED + 'Warning: only support specify initial and final lr' + Style.RESET_ALL
    args.lr.extend([args.lr[0]] * (2 - len(args.lr)))
    # clip range vf check
    assert len(args.clip_range_vf) <= 2 and Fore.RED + 'Warning: only support specify initial and final lr' + Style.RESET_ALL
    if len(args.clip_range_vf) == 0:
        args.clip_range_vf = None
    elif len(args.clip_range_vf) == 1:
        args.clip_range_vf = [args.clip_range_vf[0], args.clip_range_vf[0]]
    # operator config check
    if len(args.op_config) < 7:
        raise ValueError(Fore.RED + 'op_config must have 7 elements, as [Batch, Height, Width, Channel, Kernel size, cache size]' + Style.RESET_ALL)
    # model config check
    if args.model != 'GNN':
        args.gnn_class = 'NONE'
    else:
        # args.ncloss = False
        if not args.node_wise:
            print(Fore.YELLOW + 'Warning: GNN model only support node wise feature [force node_wise=True]' + Style.RESET_ALL)
        args.node_wise = True # force node_wise in gnn model
    # if not args.ncloss:
    #     args.ncontrast_coef = 0.0
    # feature dim check
    if not args.node_wise:
        # args.focus_num = 0
        if args.pi_conv_out > 0 or args.vf_conv_out > 0:
            print(Fore.YELLOW + 'Warning: if not node_wise, no conv should have [force pi_conv_out=0, vf_conv_out=0]' + Style.RESET_ALL)
            args.pi_conv_out = args.vf_conv_out = 0
        # if args.layernorm:
        #     print(Fore.YELLOW + 'Warning: if not node_wise, no layernorm should have [force layernorm=False]' + Style.RESET_ALL)
        #     args.layernorm = False
    # focus config check
    # if args.focus_num <= 0:
    #     args.focus_vf = args.focus_ngn = False
    #     args.focus_coef = 0.0
    #     # if args.model == 'GNN' and not args.gp_vf:
    #     #     print(Fore.YELLOW + 'Warning: if no focus, only gp_vf is support [force gp_vf=True]' + Style.RESET_ALL)
    #     #     args.gp_vf = True
    # if args.focus_num == 1:
    #     if args.focus_vf:
    #         # because policy.predict_value() do not accept action_mask as param 
    #         print(Fore.YELLOW + 'Warning: if focus_num eqs 1, never focus_vf [force focus_vf=False]' + Style.RESET_ALL)
    #         args.focus_vf = False
    #     if not args.gp_vf:
    #         print(Fore.YELLOW + 'Warning: if no gp_vf when focus_num eqs 1, all node will in vf' + Style.RESET_ALL)
    #     if not args.floss_disable:
    #         print(Fore.YELLOW + 'Warning: if focus_num eqs 1, never focus_loss [force floss_disable=True]' + Style.RESET_ALL)
    #         args.floss_disable = True
    # topk_layer check
    # if args.topk_layer <= 0:
    #     print(Fore.YELLOW + 'Warning: topk_layer should be positive [force topk_layer=1]' + Style.RESET_ALL)
    #     args.topk_layer = 1
    # normlization check
    # if args.batchnorm and args.layernorm:
    #     print(Fore.YELLOW + 'Warning: add both batchnorm and layernorm' + Style.RESET_ALL)
    # transfm check
    if args.transfm:
        if args.pi_conv_out > 0:
            print(Fore.YELLOW + 'Warning: if transformer enable, pi_conv_out should disable [force disable=True]' + Style.RESET_ALL)
            args.pi_conv_out = 0
    else:
        args.canolab = False

    if args.video:
        assert args.filename is not None, Fore.RED + 'Error: must specify filename for generate video' + Style.RESET_ALL
    if args.filename:
        args.filename = "out/strategy/" + args.filename.split('/')[-1]
    return args

def get_args():
    parser = argparse.ArgumentParser(description="Train Agent with MaskablePPO")
    ## env config
    # parser.add_argument('--env', type=str, default='RedBluePebbleGameEnv')
    parser.add_argument('--history', type=int, default=1, help='history length including current state')
    # parser.add_argument('--reward_decay', action='store_true', help='compute reward decay with time') #deprecated
    # parser.add_argument('--timetick', action='store_true', help='timetick observable') # deprecated
    parser.add_argument('--max_episode_len', type=int, default=512, help='game episode length limit')
    parser.add_argument('--op_config', type=int, default=[1, 4, 4, 1, 3, 1, 64], nargs='+', help='operator config [Batch, Height, Width, Channel, Kernel size, Outplanes, cache size]')
    parser.add_argument('--obs_field', type=str, choices=['glb', 'loc'], default='loc', help='env observation mode')
    # reward config
    parser.add_argument('--reward_timecost', type=float, default=0.0, help='reward for timecost')
    parser.add_argument('--reward_delete', type=float, default=0.0, help='reward for delete')
    parser.add_argument('--reward_load', type=float, default=-1, help='reward for load')
    parser.add_argument('--reward_store', type=float, default=-1, help='reward for store')
    parser.add_argument('--reward_compute', type=float, default=1, help='reward for compute')
    parser.add_argument('--reward_recompute', type=float, default=0.0, help='reward for recompute')
    parser.add_argument('--reward_redundant', type=float, default=0.0, help='reward for redundant action required history > 1')
    parser.add_argument('--reward_done', type=float, default=1, help='reward for done')
    parser.add_argument('--reward_scale', type=float, default=1.0, help='reward scale')
    # env rules
    parser.add_argument('--load_lock_disable', action='store_true', help='load action lock')
    parser.add_argument('--autodel_lock_disable', action='store_true', help='auto delete action lock')
    parser.add_argument('--action_truncate', action='store_true', help='truncate game if no valid action after use all constraints')
    parser.add_argument('--verbose', action='store_true', help='set env verbose')
    # parser.add_argument('--summary', action='store_true', help='set env summary')
    ## model config
    parser.add_argument('--model', type=str, choices=['GNN', 'MLP', 'NONE'], default='GNN', help='extractor model type')
    parser.add_argument('--node_wise', action='store_true', help='node wise feature')
    parser.add_argument('--features_dim', type=int, default=32, help='output feature nums [per node if node_wise]')
    parser.add_argument('--ortho_init', action='store_true', help='orthogonal init MLPExtractor/policy/value')
    parser.add_argument('--activation_fn', type=str, default='relu', choices=['relu', 'tanh', 'gelu'], help='gnn activation fn')
    # parser.add_argument('--layernorm', action='store_true', help='layer normlization')
    # parser.add_argument('--norm_mode', type=str, choices=['graph', 'node'], default='graph', help='layer norm mode')
    # parser.add_argument('--batchnorm', action='store_true', help='batch normlization')
    # parser.add_argument('--addi_out_features', type=int, default=1, help='addi features output dims')
    # parser.add_argument('--feat_ex_net', type=int, default=[], nargs='+', help='cache feature_extractor')
    # GNN
    parser.add_argument('--gnn_class', type=str, choices=['GCN', 'GAT', 'SAGE'], default='SAGE', help='gnn model type')
    parser.add_argument('--blocks', type=int, default=0, help='extractor model net shape')
    # parser.add_argument('--block_features', type=int, default=32, help='gnn block feature nums') # deprecated keep same with features_dim
    parser.add_argument('--dropout', type=float, default=0.0, help='gnn dropout rate')
    # parser.add_argument('--topk_layer', type=int, default=3, help='add more layer')
    # parser.add_argument('--focus_num', type=int, default=0, help='focus on K nodes')
    # parser.add_argument('--floss_disable', action='store_true', help='disable focus loss')
    parser.add_argument('--gp_vf', action='store_true', help='using mean global pooling to value function')
    # parser.add_argument('--focus_ngn', action='store_true', help='focus loss calc negtive sample only')
    # parser.add_argument('--focus_vf', action='store_true', help='estimate value focus on K nodes')
    # parser.add_argument('--focus_coef', type=float, default=1.0, help='focus loss coefficient')
    # parser.add_argument('--show_focus', action='store_true', default=False, help='show focus and action masks')
    parser.add_argument('--transfm', action='store_true', help='use transformer architecture in policy_value net')
    parser.add_argument('--canolab', action='store_true', help='whether to use canonical labeling')
    # MLP
    parser.add_argument('--mlp_hid_net_arch', type=int, default=[], nargs='+', help='extractor model net shape')
    # MLP-Graph
    # parser.add_argument('--ncloss', action='store_true', help='disable NContrast loss')
    # parser.add_argument('--r_pow', type=int, default=2, help='r-th power adjacency matrix ')
    # parser.add_argument('--ncontrast_coef', type=float, default=1.0, help='NContrast loss coefficient')
    # parser.add_argument('--ncloss_tau', type=float, default=1.0, help='NContrast loss temperature coefficient')
    # policy/value
    parser.add_argument('--pi_conv_out', type=int, default=0, help='conv output feature nums before actor')
    parser.add_argument('--vf_conv_out', type=int, default=0, help='conv output feature nums before critic')
    parser.add_argument('--pi_net_arch', type=int, default=[], nargs='+', help='policy net arch')
    parser.add_argument('--vf_net_arch', type=int, default=[], nargs='+', help='value net arch')
    # ## train config
    parser.add_argument('--sp_tensor', action='store_true', help='enable sparse tensor')
    parser.add_argument('--seed', type=int, default=711630, help='agent random seed')
    parser.add_argument('--ncpu', type=int, default=4, help='cpu nums for training')
    parser.add_argument('--lr', type=float, default=[3e-4, 3e-4], nargs='+', help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--clip_range_vf', type=float, default=[], nargs='+', help='value function clip')
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--tb_log_dir', type=str, default='./test/', help='tensorboard log dir')
    parser.add_argument('--total_timesteps', type=float, default=1e7, help='total training timesteps')
    parser.add_argument('--restart', action='store_true', help='restart tensorboard') 
    parser.add_argument('--agent_name', type=str, default='test', help='agent name/tb_log_name/saved model file_name')
    parser.add_argument('--save_path', type=str, default='./saved_model', help='path to save model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 
                                                                      'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'], 
                                                             help='device to use for training')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
    ## custom
    # evaluate only
    parser.add_argument('--eval_mode', action='store_true', help='no training')
    parser.add_argument('--filename', type=str, default=None, help='evaluate output filename')
    parser.add_argument('--video', action='store_true', help='render game video')
    # parser.add_argument('--early_stop', action='store_true', help='early stop for training')
    parser.add_argument('--save_freq', type=float, default=1.0, help='save model freq (%)')
    # parser.add_argument('--save_model_freq', type=int, default=, help='evaluate episode nums')
    parser.add_argument('--debug', action='store_true', help='show debug info')
    parser.add_argument('--pretr', type=str, default='', help='pretrain model name')
    # human play game
    parser.add_argument('--human', action='store_true', help='human play')
    return args_checker(parser.parse_args())
