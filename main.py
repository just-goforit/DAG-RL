import os
import json
import torch
from tqdm import tqdm
from env.env_util import play
from util.parse import get_args
from colorama import Fore, Style
from env.env_const import IN_NODE_FEATURES, MID_NODE_FEATURES, OUT_NODE_FEATURES, ADD_FEATURE, ACTION_REWARD
from model.custom_policy import CustomGNN, CustomMlp
from env.graph import operator_info, naive_Conv2d_DAG
from env.RedBluePebbleGame import RedBluePebbleGameEnv
from model.custom_policy import CustomMaskableActorCriticPolicy

# from sb3_contrib import MaskablePPO
from custom_maskable_ppo import CustomMaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EventCallback, BaseCallback

reward_config = {
    "TIME_COST": 0,
    "DELETE": 0,
    "LOAD": 0,
    "STORE": 0,
    "COMPUTE": 1,
    "RECOMPUTE": 0,
    "REDUNDANT": 0, # penalty for redundant trap 2023.12.19, masked maybe better
    "DONE": 5       # reward for finish task
}

models = {
    "GNN": CustomGNN,
    "MLP": CustomMlp
}

activation_fns = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "gelu": torch.nn.GELU,
}

def show_net_arch(net:CustomMaskablePPO):
    features_extractor_net = str(net.policy.features_extractor)
    mlp_extractor_net = str(net.policy.mlp_extractor)
    action_net = str(net.policy.action_net)
    value_net = str(net.policy.value_net)
    return features_extractor_net +'\n'+ mlp_extractor_net +'\n'+ 'ActionNet: '+ action_net +' \n'+ 'ValueNet: ' + value_net
    
def show_model_param_num(net:CustomMaskablePPO):
    total = sum(p.numel() for p in net.policy.parameters())
    print('features_extractor: %.2fK' % (sum(p.numel() for p in net.policy.features_extractor.parameters()) / 1e3))
    print('mlp_extractor_net: %.2fK' % (sum(p.numel() for p in net.policy.mlp_extractor.parameters()) / 1e3))
    if net.policy.action_net is not None:
        print('action_net: %.2fK' % (sum(p.numel() for p in net.policy.action_net.parameters()) / 1e3))
    if net.policy.value_net is not None:
        print('value_net: %.2fK' % (sum(p.numel() for p in net.policy.value_net.parameters()) / 1e3))
    return "%.2fK" % (total/1e3)

def convert_utc_to_local(utc_time, zone):  
    import pytz  
    utc = pytz.utc  
    local_time = utc_time.replace(tzinfo=utc)  
    return local_time.astimezone(pytz.timezone(zone))    

def get_timestamp():
    # import time
    # timestamp = int(time.time() * 1000)
    # return str(timestamp)
    
    from datetime import datetime
    utc_now = datetime.utcnow()  
    local_now = convert_utc_to_local(utc_now, 'Asia/Shanghai') 
    return local_now.strftime("%Y-%m-%d_%H-%M-%S-%f")[5:-3] 

# def get_timefromstr(timestamp:str):
#     from datetime import datetime
#     timestamp = int(timestamp)
#     dt = datetime.fromtimestamp(timestamp / 1000.0)
#     timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')
#     return timestamp_str
    
def save_model(model, save_dir:str, model_name:str):
    """
    model_name: str {timestamp}_{agent_name}
    saved_model
    |---- model_name1
    |       |---- model_name1.zip (model)
    |       |---- model_name1_hParam.json (args)
    |---- model_name2
    |       |---- model_name2.zip (model)
    |       |---- model_name2_hParam.json (args)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.save(save_dir + model_name)
    
class TqdmProgressCallback(BaseCallback):
    """
    使用 tqdm 显示训练进度的回调
    """
    def __init__(self, total_timesteps: int, update_freq: int = 1000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.pbar = None
        
    def _on_training_start(self) -> None:
        """训练开始时初始化 tqdm 进度条"""
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Progress",
            unit="steps",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        # 设置初始位置
        if self.num_timesteps > 0:
            self.pbar.update(self.num_timesteps)
    
    def _on_step(self) -> bool:
        """每步更新进度条"""
        if self.pbar is not None and self.num_timesteps % self.update_freq == 0:
            # 更新进度条
            current = self.num_timesteps
            self.pbar.n = current
            self.pbar.refresh()
            
            # 更新描述信息（如果有 logger 信息）
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                if 'r' in ep_info and 'l' in ep_info:
                    self.pbar.set_postfix({
                        'ep_reward': f"{ep_info['r']:.2f}",
                        'ep_len': f"{ep_info['l']:.0f}"
                    })
        return True
    
    def _on_training_end(self) -> None:
        """训练结束时关闭进度条"""
        if self.pbar is not None:
            self.pbar.close()

class customEveryNTimesteps(EventCallback):
    """
    Trigger a callback every ``n_steps`` timesteps

    :param n_steps: Number of timesteps between two trigger.
    :param callback: Callback that will be called
        when the event is triggered.
    """

    def __init__(self, n_steps: int, callback: BaseCallback):
        super().__init__(callback)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            return self._on_event()
        return True
        
    def on_training_start(self, locals_, globals_) -> None:
        super().on_training_start(locals_, globals_)
        self.last_time_trigger = self.num_timesteps
        
if __name__ == '__main__':
    args=get_args()
    # if args.debug:
    #     print(args)
    
    op_info = operator_info(batch_size=args.op_config[0], 
                            in_height=args.op_config[1], 
                            in_width=args.op_config[2], 
                            inplanes=args.op_config[3],
                            kernel_size=args.op_config[4], 
                            outplanes=args.op_config[5], 
                            S=args.op_config[6])

    graph = naive_Conv2d_DAG(op_info)
    
    print('node_num:{}'.format(graph.node_num))
    print('input_num:{}'.format(graph.input_num))
    print('output_num:{}'.format(graph.output_num))
    # print('edge:{}'.format(graph.edge_index))
    
    reward_config={}
    reward_config["TIME_COST"] = args.reward_timecost * args.reward_scale
    reward_config["DELETE"] = args.reward_delete * args.reward_scale
    reward_config["LOAD"] = args.reward_load * args.reward_scale
    reward_config["STORE"] = args.reward_store * args.reward_scale
    reward_config["COMPUTE"] = args.reward_compute * args.reward_scale
    reward_config["RECOMPUTE"] = args.reward_recompute * args.reward_scale
    reward_config["REDUNDANT"] = args.reward_redundant * args.reward_scale
    reward_config["DONE"] = args.reward_done * args.reward_scale
    
    print('Reward check:', reward_config)
    
    env_config={"verbose":False,
                "summary": False,
                "obs_field":args.obs_field,
                "obs_mode":'vec' if args.model == 'MLP' and args.node_wise==False else 'box', # mlp mode better to use vec obs_mode
                # "align_action_space": args.focus_num != 0,
                "history":args.history,
                "max_episode_len": args.max_episode_len,
                # "no_permit_invalid": False,
                # "timetick": args.timetick, # deprecated
                # "reward_decay" : args.reward_decay, #deprecated
                "reward_config": reward_config,
                "op_info":op_info,
                "load_lock_enable": not args.load_lock_disable,
                "del_lock_enable": not args.autodel_lock_disable,
                "action_truncate": args.action_truncate,
                "render_mode": "human" if args.human else None,
                "filename": args.filename}
    
    # check filename dir existence
    if args.filename is not None:
        dir = args.filename.rsplit('/', 1)[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
            
    gnn_features_extractor_config = {
        "edge_index": graph.edge_index,
        "node_num": graph.node_num,
        "node_features": max(IN_NODE_FEATURES, MID_NODE_FEATURES, OUT_NODE_FEATURES),
        "addi_features": ADD_FEATURE if args.obs_field == 'glb' else 0,
        # "wind": args.history,
        # "net_arch": args.feat_ex_net, # net to handle addi features
        "gnn_class": args.gnn_class,
        "features_dim": args.features_dim,
        # "hidden_features": args.features_dim,
        # "blocks": args.blocks,
        "activation_fn": activation_fns[args.activation_fn], # "relu", "tanh", "gelu
        # "dropout": args.dropout,
        # "layerNorm": args.layernorm,
        # "norm_mode": args.norm_mode,
        "sp_tensor" : args.sp_tensor,
        "device" : args.device,
    }
    
    mlp_features_extractor_config = {
        "node_num": graph.node_num,
        # "edge_index": graph.edge_index,
        # "r_pow": args.r_pow,
        # "ncloss": args.ncloss,
        # "node_features": NODE_FEATURES * args.history,
        # "addi_features": ADD_FEATURE,
        "features_dim": args.features_dim, # extract output dim(omit the node_num)
        "node_wise": args.node_wise,
        "net_arch": args.mlp_hid_net_arch,
        "activation_fn": activation_fns[args.activation_fn],
        # "layerNorm": args.layernorm,
        # "batchNorm": args.batchnorm,
        # "dropout": args.dropout,
        # "device": args.device,
    }
    
    policys = {
        "GNN": {
            "features_extractor_class": CustomGNN,
            "features_extractor_kwargs": gnn_features_extractor_config,
        },
        "MLP": {
            "features_extractor_class": CustomMlp,
            "features_extractor_kwargs": mlp_features_extractor_config,
        },
        "NONE": {
            "features_extractor_class": None,
            "features_extractor_kwargs": None,
        }
    }
    policy_kwargs = dict(
        features_extractor_class=policys[args.model]["features_extractor_class"],
        features_extractor_kwargs=policys[args.model]["features_extractor_kwargs"],
        activation_fn=activation_fns[args.activation_fn],
        node_num=graph.node_num,
        edge_index=graph.edge_index if args.canolab else None,
        addi_features = 0 if args.obs_field == 'loc' or args.model == 'MLP' else ADD_FEATURE,
        pi_conv_out=args.pi_conv_out,
        vf_conv_out=args.vf_conv_out,
        # topk_layer=args.topk_layer,
        gp_vf=args.gp_vf,
        transformer = args.transfm,
        # focus_num=min(args.focus_num, graph.node_num),
        # focus_vf=args.focus_vf,
        # focus_ngn=args.focus_ngn,
        # show_focus=args.show_focus,
        # ncloss = args.ncloss,
        # ncloss_tau = args.ncloss_tau,
        # layerNorm = args.layernorm,
        # norm_mode = args.norm_mode,
        node_wise = args.node_wise,
        # myinit = args.ortho_init,
        ortho_init = args.ortho_init,
        optimizer_kwargs=dict(eps=1e-5, weight_decay=args.weight_decay),
        net_arch=dict(pi=args.pi_net_arch, vf=args.vf_net_arch) # default is dict(pi=[64, 64], vf=[64, 64]) need
    )
    
    simple_policy_kwargs = dict(
        activation_fn=activation_fns[args.activation_fn],
        net_arch=dict(pi=args.pi_net_arch, vf=args.vf_net_arch) # default is dict(pi=[64, 64], vf=[64, 64]) need
    )
    
    def Linear_schedule(initial_value: float, final_value: float):
        def func(progress_remaining: float):
            return progress_remaining * (initial_value - final_value) + final_value
        return func
    
    tstr = get_timestamp()
    model_name = tstr +'_'+ args.agent_name
    # model save path
    save_dir = args.save_path + '/' + model_name + '/'
    
    clip_range_vf = None
    if args.clip_range_vf != None:
        clip_range_vf = Linear_schedule(args.clip_range_vf[0], args.clip_range_vf[1]) if args.clip_range_vf[0] != args.clip_range_vf[1] else args.clip_range_vf[0]
    
    if not args.eval_mode and not args.human:
        # train
        # env_config["verbose"] = False
        # env_config["summary"] = False
        venv = make_vec_env(env_id=RedBluePebbleGameEnv, 
                           env_kwargs=dict(config=env_config), 
                           n_envs = args.ncpu, 
                           vec_env_cls=SubprocVecEnv)
        # venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=65536, clip_reward=10240)
        # Train MaskablePPO on the environment
        model = CustomMaskablePPO(env=venv,
                                #   focus=args.focus_num > 1 and not args.floss_disable,
                                #   focus_coef=args.focus_coef,
                                  policy=CustomMaskableActorCriticPolicy if args.model != 'NONE' else MaskableActorCriticPolicy, 
                                  learning_rate=Linear_schedule(args.lr[0], args.lr[1]) if args.lr[0] != args.lr[1] else args.lr[0],
                                  n_steps=args.n_steps if args.n_steps > 0 else args.max_episode_len,
                                  batch_size=args.batch_size,
                                  gamma=args.gamma,
                                  gae_lambda=args.gae_lambda,
                                  clip_range=args.clip_range,
                                  clip_range_vf=clip_range_vf,
                                  ent_coef=args.ent_coef,
                                  vf_coef=args.vf_coef,
                                  max_grad_norm=args.max_grad_norm,
                                  seed=args.seed,
                                  tensorboard_log=args.tb_log_dir,
                                  device = args.device,
                                  policy_kwargs=policy_kwargs if args.model != 'NONE' else simple_policy_kwargs,
                                  verbose=0)
        if args.debug:
            print(show_net_arch(model))
            print('Total: '+show_model_param_num(model))
            exit(0)

        conti = 0
        if args.pretr != '':
            conti = int(args.pretr.split('_')[-2])
            mname = args.save_path + '/' + args.pretr.rsplit('_', 2)[0] + '/' + args.pretr
            model = model.load(mname, 
                               venv,
                               custom_objects={'n_steps': args.n_steps,
                                               'learning_rate': Linear_schedule(args.lr[0], args.lr[1]) if args.lr[0] != args.lr[1] else args.lr[0],
                                               'clip_range_vf': clip_range_vf,
                                               'node_num': graph.node_num,
                                               'observation_space': venv.observation_space,
                                               'action_space': venv.action_space,
                                               'policy_kwargs': policy_kwargs if args.model != 'NONE' else simple_policy_kwargs}, 
                               device=args.device)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        with open(save_dir + f'{model_name}_hParam.json', 'w') as json_file:
            model_str = show_net_arch(model)
            model_param_num_str = show_model_param_num(model)
            cont = vars(args)
            cont['model_arch'] = model_str
            cont['model_param_num'] = model_param_num_str
            json.dump(cont, json_file)
            print('Total: ', model_param_num_str)
            
        print('Time: ' + Fore.BLUE + tstr + Style.RESET_ALL)
        # model save callback
        save_freq = int(args.save_freq * args.total_timesteps)
        checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=save_dir, name_prefix=model_name)
        event_callback = customEveryNTimesteps(n_steps=save_freq, callback=checkpoint_on_event)
        
        # 创建 tqdm 进度条回调
        tqdm_callback = TqdmProgressCallback(total_timesteps=int(args.total_timesteps), update_freq=args.n_steps)
        
        # 组合所有回调
        from stable_baselines3.common.callbacks import CallbackList
        callback_list = CallbackList([event_callback, tqdm_callback])
        
        # try:
        model.learn(total_timesteps = args.total_timesteps, 
                    tb_log_name = model_name, 
                    reset_num_timesteps = args.restart, 
                    callback=callback_list)

        if args.total_timesteps % save_freq != 0: 
            save_model(model, save_dir, model_name + f'_{args.total_timesteps}_steps')
            
        if args.pretr != '': 
            args.total_timesteps += conti
            
        ## eval using reward_config of training
        env_config["verbose"] = args.verbose
        env_config["summary"] = True
        env = Monitor(RedBluePebbleGameEnv(env_config))
        evaluate_policy(model, 
                        env, 
                        n_eval_episodes=1,
                        deterministic=True)
        
    if args.eval_mode:
        env_config["verbose"] = args.verbose
        env_config["summary"] = True
        ## using std reward_config
        env_config["reward_config"] = ACTION_REWARD
        env = Monitor(RedBluePebbleGameEnv(env_config))
        model = CustomMaskablePPO(env=env,
                                #   focus=args.focus_num > 1 and not args.floss_disable,
                                #   focus_coef=args.focus_coef,
                                  policy=CustomMaskableActorCriticPolicy if args.model != 'NONE' else MaskableActorCriticPolicy, 
                                  seed=args.seed,
                                  policy_kwargs=policy_kwargs if args.model != 'NONE' else simple_policy_kwargs, 
                                  verbose=0,
                                  _init_setup_model=False)
        
        mname = save_dir + '/' + model_name + f'_{args.total_timesteps}_steps' if not args.eval_mode else args.save_path + '/' + args.agent_name.rsplit('_', 2)[0] + '/' + args.agent_name
        "when eval only args.agent_name equals to timestamp_agent_name_nsteps_steps"
        model = model.load(mname, 
                           device=args.device, 
                           custom_objects={'n_steps': args.n_steps,
                                           'node_num': env.g.node_num,
                                           'observation_space': env.observation_space,
                                           'action_space': env.action_space,
                                           'policy_kwargs': policy_kwargs if args.model != 'NONE' else simple_policy_kwargs})
        
        if args.debug:
            print(show_net_arch(model))
            print('Total: '+show_model_param_num(model))
            exit(0)
        evaluate_policy(model, 
                        env, 
                        n_eval_episodes=1,
                        deterministic=True)
        if args.video:
            import time
            from util.strategy_vis import GridAnimation, set_config
            set_config(bar="none",
                       log_level="ERROR",
                       output_file=args.filename.split('.')[0].split('/')[-1] + '.mp4',
                       media_dir="out/media")
            start_time = time.time()
            animation = GridAnimation(op_info=op_info,
                                      filename=args.filename,
                                      a=0.25)
            animation.render()
            end_time = time.time()
            render_time = end_time - start_time
            print(f"Rendered game animation in {render_time:.2f} seconds")
            
    if args.human:
        speeds = []
        play(config=env_config, speeds=speeds)
