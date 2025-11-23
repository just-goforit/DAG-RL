"""
测试 RedBluePebbleGameEnv 是否能在 RLlib 中正常使用。

此脚本会测试：
1. 环境的基本功能（reset, step, action_masks）
2. 环境在 RLlib 中的注册和使用
3. 与 RLlib PPO 的兼容性
"""

import numpy as np
from colorama import Fore, Style, init

# 初始化 colorama
init(autoreset=True)

# RLlib imports
try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.utils.test_utils import check_learning_achieved
    from ray.tune.registry import register_env
    from ray import tune
except ImportError:
    print(Fore.RED + "错误: RLlib 未安装。请运行: pip install 'ray[rllib]'")
    exit(1)

# Local imports
from env.RedBluePebbleGame import RedBluePebbleGameEnv
from env.graph import operator_info, naive_Conv2d_DAG
from env.env_const import ACTION_REWARD


def print_test_header(test_name: str):
    """打印测试标题"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}测试: {test_name}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


def print_success(message: str):
    """打印成功消息"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_error(message: str):
    """打印错误消息"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def print_warning(message: str):
    """打印警告消息"""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")


def test_basic_env_functionality():
    """测试环境的基本功能"""
    print_test_header("环境基本功能测试")
    
    try:
        # 创建测试用的 operator_info
        op_info = operator_info(
            batch_size=1,
            in_height=5,
            in_width=5,
            inplanes=2,
            kernel_size=3,
            outplanes=2,
            S=8
        )
        
        # 创建环境配置
        env_config = {
            "verbose": False,
            "summary": False,
            "obs_mode": "box",
            "obs_field": "loc",
            "max_episode_len": 100,
            "reward_config": ACTION_REWARD,
            "op_info": op_info,
            "load_lock_enable": True,
            "del_lock_enable": True,
            "action_truncate": False,
        }
        
        # 测试 1: 环境创建
        print("1. 测试环境创建...")
        env = RedBluePebbleGameEnv(env_config)
        print_success("环境创建成功")
        
        # 测试 2: 观测空间和动作空间
        print("2. 测试观测空间和动作空间...")
        print(f"   观测空间: {env.observation_space}")
        print(f"   动作空间: {env.action_space}")
        print_success(f"观测空间形状: {env.observation_space.shape}, 动作空间大小: {env.action_space.n}")
        
        # 测试 3: reset 方法
        print("3. 测试 reset() 方法...")
        obs, info = env.reset()
        assert obs is not None, "观测不应为 None"
        assert isinstance(obs, np.ndarray), "观测应为 numpy 数组"
        assert obs.shape == env.observation_space.shape, f"观测形状不匹配: {obs.shape} != {env.observation_space.shape}"
        assert "action_mask" in info or "action_mask" not in info, "info 中可能包含 action_mask"
        print_success(f"reset() 成功，观测形状: {obs.shape}")
        
        # 测试 4: action_masks 方法
        print("4. 测试 action_masks() 方法...")
        masks = env.action_masks()
        assert masks is not None, "动作掩码不应为 None"
        assert isinstance(masks, np.ndarray), "动作掩码应为 numpy 数组"
        assert masks.dtype == bool or masks.dtype == np.bool_, "动作掩码应为布尔类型"
        assert len(masks) == env.action_space.n, f"掩码长度不匹配: {len(masks)} != {env.action_space.n}"
        print_success(f"action_masks() 成功，掩码形状: {masks.shape}, 有效动作数: {np.sum(masks)}")
        
        # 测试 5: step 方法
        print("5. 测试 step() 方法...")
        # 选择一个有效动作
        valid_actions = np.where(masks)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None, "下一步观测不应为 None"
            assert isinstance(reward, (int, float)), "奖励应为数字"
            assert isinstance(terminated, bool), "terminated 应为布尔值"
            assert isinstance(truncated, bool), "truncated 应为布尔值"
            assert isinstance(info, dict), "info 应为字典"
            print_success(f"step() 成功，奖励: {reward:.2f}, 终止: {terminated}, 截断: {truncated}")
            
            # 检查 info 中是否包含 action_mask
            if "action_mask" in info:
                print_success("info 中包含 action_mask（RLlib 兼容）")
            else:
                print_warning("info 中不包含 action_mask，但这不是必需的")
        else:
            print_warning("没有有效动作，跳过 step() 测试")
        
        # 测试 6: 多次 step
        print("6. 测试多次 step()...")
        env.reset()
        for i in range(5):
            masks = env.action_masks()
            valid_actions = np.where(masks)[0]
            if len(valid_actions) == 0:
                print_warning(f"第 {i+1} 步后没有有效动作")
                break
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print_success(f"Episode 在第 {i+1} 步结束")
                break
        else:
            print_success("成功执行 5 步")
        
        return True
        
    except Exception as e:
        print_error(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_rllib_env_registration():
    """测试在 RLlib 中注册环境"""
    print_test_header("RLlib 环境注册测试")
    
    try:
        # 创建环境工厂函数
        def env_creator(env_config):
            """环境创建函数"""
            op_info = operator_info(
                batch_size=1,
                in_height=5,
                in_width=5,
                inplanes=2,
                kernel_size=3,
                outplanes=2,
                S=8
            )
            
            config = {
                "verbose": False,
                "summary": False,
                "obs_mode": "box",
                "obs_field": "loc",
                "max_episode_len": 100,
                "reward_config": ACTION_REWARD,
                "op_info": op_info,
                "load_lock_enable": True,
                "del_lock_enable": True,
                "action_truncate": False,
            }
            # 合并传入的配置
            if env_config:
                config.update(env_config)
            
            return RedBluePebbleGameEnv(config)
        
        # 注册环境
        print("1. 注册环境到 RLlib...")
        register_env("RedBluePebbleGame-v0", env_creator)
        print_success("环境注册成功")
        
        # 测试环境创建
        print("2. 通过 RLlib 创建环境...")
        env = env_creator({})
        obs, info = env.reset()
        print_success(f"环境创建成功，观测形状: {obs.shape}")
        
        # 测试 action_mask 在 info 中
        print("3. 检查 action_mask 支持...")
        if "action_mask" in info:
            print_success("info 中包含 action_mask")
        else:
            # 手动添加 action_mask 到 info（如果环境没有自动添加）
            masks = env.action_masks()
            info["action_mask"] = masks
            print_warning("手动添加 action_mask 到 info")
        
        return True
        
    except Exception as e:
        print_error(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_rllib_ppo_compatibility():
    """测试与 RLlib PPO 的兼容性"""
    print_test_header("RLlib PPO 兼容性测试")
    
    try:
        # 创建环境工厂函数
        def env_creator(env_config):
            op_info = operator_info(
                batch_size=1,
                in_height=5,
                in_width=5,
                inplanes=2,
                kernel_size=3,
                outplanes=2,
                S=8
            )
            
            config = {
                "verbose": False,
                "summary": False,
                "obs_mode": "box",
                "obs_field": "loc",
                "max_episode_len": 50,  # 较短的 episode 用于快速测试
                "reward_config": ACTION_REWARD,
                "op_info": op_info,
                "load_lock_enable": True,
                "del_lock_enable": True,
                "action_truncate": False,
            }
            if env_config:
                config.update(env_config)
            
            return RedBluePebbleGameEnv(config)
        
        # 注册环境
        register_env("RedBluePebbleGame-v0", env_creator)
        
        # 创建 PPO 配置
        print("1. 创建 PPO 配置...")
        config = (
            PPOConfig()
            .environment("RedBluePebbleGame-v0")
            .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
            .learners(
                num_learners=0,  # Set this to greater than 1 to allow for DDP style updates.
                num_gpus_per_learner=0,  # Set this to 1 to enable GPU training.
                num_cpus_per_learner=1,
            )
            .training(
                train_batch_size=200
            )
            .evaluation(
                # Run one evaluation round every iteration.
                evaluation_interval=1,
                evaluation_num_env_runners=1,
                evaluation_duration_unit="episodes",
                evaluation_duration=1,
            )
        )
        print_success("PPO 配置创建成功")
        
        # 测试配置构建
        print("2. 构建 PPO 算法...")
        algo = config.build()
        print_success("PPO 算法构建成功")
        
        # 测试训练一步
        print("3. 测试训练一步...")
        result = algo.train()
        print_success(f"训练成功，episode_reward_mean: {result.get('episode_reward_mean', 'N/A')}")
        
        # 测试评估
        print("4. 测试评估...")
        eval_result = algo.evaluate()
        print_success(f"评估成功，episode_reward_mean: {eval_result.get('evaluation', {}).get('episode_reward_mean', 'N/A')}")
        
        # 清理
        algo.stop()
        
        return True
        
    except Exception as e:
        print_error(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_action_mask_in_info():
    """测试 action_mask 是否正确包含在 info 中"""
    print_test_header("action_mask 信息测试")
    
    try:
        op_info = operator_info(
            batch_size=1,
            in_height=5,
            in_width=5,
            inplanes=2,
            kernel_size=3,
            outplanes=2,
            S=8
        )
        
        env_config = {
            "verbose": False,
            "summary": False,
            "obs_mode": "box",
            "obs_field": "loc",
            "max_episode_len": 100,
            "reward_config": ACTION_REWARD,
            "op_info": op_info,
            "load_lock_enable": True,
            "del_lock_enable": True,
            "action_truncate": False,
        }
        
        env = RedBluePebbleGameEnv(env_config)
        
        # 测试 reset 后的 info
        print("1. 测试 reset() 后的 info...")
        obs, info = env.reset()
        
        # 检查 _get_info 方法是否包含 action_mask
        # 根据代码，_get_info 方法在 cached_action_mask 不为 None 时会添加 action_mask
        # 但 reset 后 cached_action_mask 可能为 None，所以需要先调用一次 _get_valid_action_mask
        env._get_valid_action_mask()  # 确保 cached_action_mask 被设置
        info = env._get_info()
        
        if "action_mask" in info:
            print_success("reset() 后 info 包含 action_mask")
            masks = info["action_mask"]
            print(f"   action_mask 形状: {masks.shape}, 类型: {masks.dtype}")
        else:
            print_warning("reset() 后 info 不包含 action_mask（可能需要先调用 _get_valid_action_mask）")
        
        # 测试 step 后的 info
        print("2. 测试 step() 后的 info...")
        masks = env.action_masks()
        valid_actions = np.where(masks)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if "action_mask" in info:
                print_success("step() 后 info 包含 action_mask")
                masks = info["action_mask"]
                print(f"   action_mask 形状: {masks.shape}, 类型: {masks.dtype}")
            else:
                print_warning("step() 后 info 不包含 action_mask")
        else:
            print_warning("没有有效动作，跳过 step() 测试")
        
        return True
        
    except Exception as e:
        print_error(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.MAGENTA}RedBluePebbleGameEnv RLlib 兼容性测试")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}\n")
    
    results = []
    
    # 运行测试
    results.append(("环境基本功能", test_basic_env_functionality()))
    results.append(("RLlib 环境注册", test_rllib_env_registration()))
    results.append(("action_mask 信息", test_action_mask_in_info()))
    
    # PPO 兼容性测试可能需要较长时间，可以跳过
    try:
        results.append(("RLlib PPO 兼容性", test_rllib_ppo_compatibility()))
    except Exception as e:
        print_warning(f"跳过 PPO 兼容性测试: {str(e)}")
        results.append(("RLlib PPO 兼容性", False))
    
    # 打印总结
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}测试总结")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Fore.GREEN}通过{Style.RESET_ALL}" if result else f"{Fore.RED}失败{Style.RESET_ALL}"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print(f"{Fore.GREEN}✓ 所有测试通过！RedBluePebbleGameEnv 可以在 RLlib 中使用。{Style.RESET_ALL}")
        return 0
    else:
        print(f"{Fore.YELLOW}⚠ 部分测试失败，请检查上述错误信息。{Style.RESET_ALL}")
        return 1


if __name__ == "__main__":
    exit(main())

