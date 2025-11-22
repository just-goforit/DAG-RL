"""
RLlib training script for Red-Blue Pebble Game.

This script provides a RLlib-compatible training interface that reuses
the existing environment and model components while supporting action masks.
"""

import os
import json
import torch
from typing import Dict, Any, Optional
from tqdm import tqdm
from colorama import Fore, Style

# RLlib imports
try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.utils.test_utils import check_learning_achieved
    from ray import tune
    from ray.tune.registry import register_env
except ImportError:
    raise ImportError(
        "RLlib is not installed. Please install it with: pip install 'ray[rllib]'"
    )

# Local imports
from util.parse import get_args
from env.env_const import IN_NODE_FEATURES, MID_NODE_FEATURES, OUT_NODE_FEATURES, ADD_FEATURE, ACTION_REWARD
from env.graph import operator_info, naive_Conv2d_DAG
from env.RedBluePebbleGame import RedBluePebbleGameEnv
from model.rllib_rl_module import CustomMaskableTorchRLModule

# Activation functions mapping
activation_fns = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "gelu": torch.nn.GELU,
}


def convert_utc_to_local(utc_time, zone):
    """Convert UTC time to local timezone."""
    import pytz
    utc = pytz.utc
    local_time = utc_time.replace(tzinfo=utc)
    return local_time.astimezone(pytz.timezone(zone))


def get_timestamp():
    """Generate timestamp string for model naming."""
    from datetime import datetime
    utc_now = datetime.utcnow()
    local_now = convert_utc_to_local(utc_now, 'Asia/Shanghai')
    return local_now.strftime("%Y-%m-%d_%H-%M-%S-%f")[5:-3]


def create_env_config(args) -> Dict[str, Any]:
    """
    Create environment configuration dictionary from args.
    
    :param args: Parsed command line arguments
    :return: Environment configuration dictionary
    """
    # Build reward config
    reward_config = {
        "TIME_COST": args.reward_timecost * args.reward_scale,
        "DELETE": args.reward_delete * args.reward_scale,
        "LOAD": args.reward_load * args.reward_scale,
        "STORE": args.reward_store * args.reward_scale,
        "COMPUTE": args.reward_compute * args.reward_scale,
        "RECOMPUTE": args.reward_recompute * args.reward_scale,
        "REDUNDANT": args.reward_redundant * args.reward_scale,
        "DONE": args.reward_done * args.reward_scale,
    }
    
    # Build operator info
    op_info = operator_info(
        batch_size=args.op_config[0],
        in_height=args.op_config[1],
        in_width=args.op_config[2],
        inplanes=args.op_config[3],
        kernel_size=args.op_config[4],
        outplanes=args.op_config[5],
        S=args.op_config[6]
    )
    
    # Build environment config
    env_config = {
        "verbose": False,
        "summary": False,
        "obs_field": args.obs_field,
        "obs_mode": 'vec' if args.model == 'MLP' and args.node_wise == False else 'box',
        "history": args.history,
        "max_episode_len": args.max_episode_len,
        "reward_config": reward_config,
        "op_info": op_info,
        "load_lock_enable": not args.load_lock_disable,
        "del_lock_enable": not args.autodel_lock_disable,
        "action_truncate": args.action_truncate,
        "render_mode": "human" if args.human else None,
        "filename": args.filename,
    }
    
    return env_config


def create_model_config(args, graph) -> Dict[str, Any]:
    """
    Create RLModule model configuration from args and graph.
    
    :param args: Parsed command line arguments
    :param graph: DAG graph object
    :return: Model configuration dictionary for RLModule
    """
    # Determine addi_features based on obs_field
    addi_features = ADD_FEATURE if args.obs_field == 'glb' else 0
    
    # Build net_arch from pi_net_arch and vf_net_arch
    net_arch = {
        "pi": args.pi_net_arch,
        "vf": args.vf_net_arch,
    }
    
    # Build model config
    model_config = {
        "node_num": graph.node_num,
        "edge_index": graph.edge_index,
        "addi_features": addi_features,
        "use_gnn": args.model == 'GNN',
        "gnn_class": args.gnn_class if args.model == 'GNN' else 'NONE',
        "features_dim": args.features_dim,
        "net_arch": net_arch,
        "activation_fn": args.activation_fn,
        "dropout": args.dropout,
        "sp_tensor": args.sp_tensor,
        "ortho_init": args.ortho_init,
        "pi_conv_out": args.pi_conv_out,
        "vf_conv_out": args.vf_conv_out,
        "gp_vf": args.gp_vf,
        "transformer": args.transfm,
        "node_wise": args.node_wise,
        "node_features": max(IN_NODE_FEATURES, MID_NODE_FEATURES, OUT_NODE_FEATURES),
    }
    
    return model_config


def create_ppo_config(args, env_config: Dict[str, Any], model_config: Dict[str, Any]) -> PPOConfig:
    """
    Create PPO configuration with RLModule and training parameters.
    
    :param args: Parsed command line arguments
    :param env_config: Environment configuration dictionary
    :param model_config: Model configuration dictionary
    :return: Configured PPOConfig instance
    """
    # Register environment
    register_env("RedBluePebbleGameEnv", lambda config: RedBluePebbleGameEnv(config=config))
    
    # Create PPO config
    ppo_config = (
        PPOConfig()
        .environment(
            env="RedBluePebbleGameEnv",
            env_config=env_config,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=CustomMaskableTorchRLModule,
                model_config=model_config,
            )
        )
        .training(
            lr=args.lr[0],  # Use initial learning rate, can be scheduled later
            train_batch_size=args.batch_size * args.n_epochs if args.n_steps > 0 else args.max_episode_len * args.n_epochs,
            sgd_minibatch_size=args.batch_size,
            num_sgd_iter=args.n_epochs,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            clip_param=args.clip_range,
            vf_clip_param=args.clip_range_vf[0] if args.clip_range_vf and len(args.clip_range_vf) > 0 else None,
            entropy_coeff=args.ent_coef,
            vf_loss_coeff=args.vf_coef,
            grad_clip=args.max_grad_norm,
        )
        .rollouts(
            num_rollout_workers=args.ncpu,
            num_envs_per_worker=1,
            rollout_fragment_length=args.n_steps if args.n_steps > 0 else args.max_episode_len,
        )
        .resources(
            num_gpus=1 if args.device.startswith('cuda') else 0,
        )
        .framework("torch")
        .debugging(seed=args.seed)
    )
    
    # Set TensorBoard logging
    if args.tb_log_dir:
        ppo_config = ppo_config.reporting(
            metrics_num_episodes_for_smoothing=10,
        )
    
    return ppo_config


def save_model_rllib(algo, save_dir: str, model_name: str, args: Any):
    """
    Save RLlib algorithm model and hyperparameters.
    
    :param algo: RLlib algorithm instance
    :param save_dir: Directory to save model
    :param model_name: Model name prefix
    :param args: Parsed arguments for hyperparameter saving
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save model using RLlib's save method
    checkpoint_path = algo.save(save_dir + model_name)
    
    # Save hyperparameters
    hparam_path = os.path.join(save_dir, f'{model_name}_hParam.json')
    with open(hparam_path, 'w') as json_file:
        cont = vars(args)
        # Add model info if available
        if hasattr(algo, 'get_policy'):
            try:
                policy = algo.get_policy()
                if hasattr(policy, 'model'):
                    cont['model_arch'] = str(policy.model)
            except:
                pass
        json.dump(cont, json_file, indent=2)
    
    print(f"Model saved to: {checkpoint_path}")
    return checkpoint_path


def show_model_param_num_rllib(algo) -> str:
    """
    Show model parameter count for RLlib algorithm.
    
    :param algo: RLlib algorithm instance
    :return: Total parameter count string
    """
    try:
        policy = algo.get_policy()
        if hasattr(policy, 'model'):
            total = sum(p.numel() for p in policy.model.parameters())
            return f"{total/1e3:.2f}K"
    except:
        pass
    return "N/A"


def train_rllib(args):
    """
    Main training function using RLlib.
    
    :param args: Parsed command line arguments
    """
    print("=" * 80)
    print("Starting RLlib Training")
    print("=" * 80)
    
    # Create graph structure
    op_info = operator_info(
        batch_size=args.op_config[0],
        in_height=args.op_config[1],
        in_width=args.op_config[2],
        inplanes=args.op_config[3],
        kernel_size=args.op_config[4],
        outplanes=args.op_config[5],
        S=args.op_config[6]
    )
    
    graph = naive_Conv2d_DAG(op_info)
    
    print(f'node_num: {graph.node_num}')
    print(f'input_num: {graph.input_num}')
    print(f'output_num: {graph.output_num}')
    
    # Create configurations
    env_config = create_env_config(args)
    model_config = create_model_config(args, graph)
    
    print('Reward config:', env_config["reward_config"])
    
    # Create PPO config
    ppo_config = create_ppo_config(args, env_config, model_config)
    
    # Build algorithm
    algo = ppo_config.build()
    
    # Generate model name
    tstr = get_timestamp()
    model_name = tstr + '_' + args.agent_name
    save_dir = os.path.join(args.save_path, model_name) + '/'
    
    # Create save directory and save initial hyperparameters
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir, f'{model_name}_hParam.json'), 'w') as json_file:
        cont = vars(args)
        cont['model_param_num'] = show_model_param_num_rllib(algo)
        json.dump(cont, json_file, indent=2)
    
    print('Time: ' + Fore.BLUE + tstr + Style.RESET_ALL)
    print('Total params: ' + show_model_param_num_rllib(algo))
    
    # Training loop
    total_timesteps = int(args.total_timesteps)
    save_freq = int(args.save_freq * total_timesteps)
    current_timesteps = 0
    
    # Load checkpoint if resuming
    if args.pretr != '':
        checkpoint_path = os.path.join(args.save_path, args.pretr)
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            algo.restore(checkpoint_path)
            # Extract timesteps from checkpoint name if possible
            try:
                conti = int(args.pretr.split('_')[-2])
                current_timesteps = conti
                total_timesteps += conti
            except:
                pass
    
    # Training with progress bar
    pbar = tqdm(
        total=total_timesteps,
        desc="Training Progress",
        unit="steps",
        ncols=100,
        initial=current_timesteps,
    )
    
    try:
        while current_timesteps < total_timesteps:
            # Train for one iteration
            result = algo.train()
            current_timesteps = result.get("timesteps_total", current_timesteps)
            
            # Update progress bar
            pbar.n = current_timesteps
            pbar.refresh()
            
            # Update progress bar description with metrics
            if "episode_reward_mean" in result:
                pbar.set_postfix({
                    'reward': f"{result['episode_reward_mean']:.2f}",
                    'len': f"{result.get('episode_len_mean', 0):.0f}"
                })
            
            # Save checkpoint periodically
            if current_timesteps % save_freq == 0:
                checkpoint_name = f"{model_name}_{current_timesteps}_steps"
                save_model_rllib(algo, save_dir, checkpoint_name, args)
        
        # Final save
        if current_timesteps % save_freq != 0:
            checkpoint_name = f"{model_name}_{current_timesteps}_steps"
            save_model_rllib(algo, save_dir, checkpoint_name, args)
    
    finally:
        pbar.close()
    
    print("\n" + "=" * 80)
    print("Training Completed")
    print("=" * 80)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    env_config["verbose"] = args.verbose
    env_config["summary"] = True
    env = RedBluePebbleGameEnv(config=env_config)
    
    # Simple evaluation
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < args.max_episode_len:
        # Get action mask from info if available
        action_mask = info.get('action_mask', None)
        action = algo.compute_single_action(obs, explore=False, action_mask=action_mask)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    
    print(f"Evaluation - Steps: {steps}, Reward: {total_reward:.2f}")


def evaluate_rllib(args):
    """
    Evaluation function using RLlib.
    
    :param args: Parsed command line arguments
    """
    print("=" * 80)
    print("Starting RLlib Evaluation")
    print("=" * 80)
    
    # Create graph structure
    op_info = operator_info(
        batch_size=args.op_config[0],
        in_height=args.op_config[1],
        in_width=args.op_config[2],
        inplanes=args.op_config[3],
        kernel_size=args.op_config[4],
        outplanes=args.op_config[5],
        S=args.op_config[6]
    )
    
    graph = naive_Conv2d_DAG(op_info)
    
    # Create configurations
    env_config = create_env_config(args)
    env_config["verbose"] = args.verbose
    env_config["summary"] = True
    env_config["reward_config"] = ACTION_REWARD  # Use standard reward config for evaluation
    
    model_config = create_model_config(args, graph)
    
    # Create PPO config
    ppo_config = create_ppo_config(args, env_config, model_config)
    
    # Build algorithm
    algo = ppo_config.build()
    
    # Load model
    if args.eval_mode:
        model_path = os.path.join(args.save_path, args.agent_name)
    else:
        model_name = args.agent_name.rsplit('_', 2)[0] if '_' in args.agent_name else args.agent_name
        model_path = os.path.join(args.save_path, model_name, args.agent_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    algo.restore(model_path)
    
    # Create environment
    env = RedBluePebbleGameEnv(config=env_config)
    
    # Evaluation
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < args.max_episode_len:
        # Get action mask from info if available
        action_mask = info.get('action_mask', None)
        action = algo.compute_single_action(obs, explore=False, action_mask=action_mask)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    
    print(f"\nEvaluation Results:")
    print(f"  Steps: {steps}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    
    # Generate video if requested
    if args.video and args.filename:
        import time
        from util.strategy_vis import GridAnimation, set_config
        
        set_config(
            bar="none",
            log_level="ERROR",
            output_file=args.filename.split('.')[0].split('/')[-1] + '.mp4',
            media_dir="out/media"
        )
        
        start_time = time.time()
        animation = GridAnimation(
            op_info=op_info,
            filename=args.filename,
            a=0.25
        )
        animation.render()
        end_time = time.time()
        render_time = end_time - start_time
        print(f"Rendered game animation in {render_time:.2f} seconds")


def main():
    """Main entry point."""
    args = get_args()
    
    # Check filename directory
    if args.filename is not None:
        dir_path = args.filename.rsplit('/', 1)[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    if args.eval_mode:
        evaluate_rllib(args)
    elif args.human:
        # Human play mode - reuse existing play function
        from env.env_util import play
        env_config = create_env_config(args)
        speeds = []
        play(config=env_config, speeds=speeds)
    else:
        train_rllib(args)


if __name__ == '__main__':
    main()

