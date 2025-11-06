# AI Coding Agent Instructions for GNN Test Project

## Project Overview
This is a **reinforcement learning (RL) project** that trains Graph Neural Network (GNN) and MLP agents to solve the **Red-Blue Pebble Game** optimization problem using **Masked Proximal Policy Optimization (MaskablePPO)**. The game simulates computational graph execution with cache/memory constraints, resembling tensor operator scheduling problems (e.g., Conv2d operations).

**Core Goal**: Train agents to minimize memory loads/stores when executing DAG-based computation graphs with limited cache capacity.

## Architecture Components

### 1. Environment (`env/`)
- **`RedBluePebbleGame.py`**: Custom Gymnasium environment implementing the pebble game
  - State: DAG nodes with features (cached/memed/computed/contributed)
  - Actions: LOAD, STORE, DELETE, COMPUTE on graph nodes
  - Rewards configured via `reward_config` dict (see `env_const.py`)
  - Observation modes: `'vec'` (flattened) or `'box'` (aligned), fields: `'loc'` (local) or `'glb'` (with cache size S)
- **`graph.py`**: DAG graph representation (`DAG_Graph` class) and `naive_Conv2d_DAG()` generator
- **`env_const.py`**: Constants for node features, action encoding, and default rewards

### 2. Model (`model/`)
- **`custom_policy.py`**: Custom policy classes extending `MaskableActorCriticPolicy`
  - `CustomMaskableActorCriticPolicy`: Main policy supporting both GNN and MLP feature extractors
  - `CustomGNN` & `CustomMlp`: Feature extractor implementations
- **`graph_extractor.py`**: GNN layers (custom `SAGEConv` with CUDA-accelerated SpMM via `spmm_ext`)
- **`policy_value.py`**: `CustomPolicyValueNet` implementing separate policy/value heads with optional graph pooling and transformer support
- **`transformer.py`**: Transformer-based pointer network for graph-aware policy (optional)

### 3. Training (`main.py`, `custom_maskable_ppo.py`)
- **`custom_maskable_ppo.py`**: Custom PPO trainer extending `sb3_contrib.MaskablePPO`
  - Overrides `train()` method to track custom metrics
  - Supports action masking for invalid actions
- **`main.py`**: Entry point with argument parsing, environment setup, training loop, and evaluation
  - Uses `stable_baselines3` with `SubprocVecEnv` for parallel environments
  - Model checkpointing via `CheckpointCallback`

### 4. CUDA Extension (`spmm_ext_src/`)
- **`spmm_ext.cu`**: Custom CUDA kernel `gspmm_src_mul_e_sum` for efficient sparse matrix operations
- **Build**: Run `python setup.py install` in `spmm_ext_src/` (Linux only, requires CUDA toolkit)
- **Note**: Project developed on Linux but viewed on Windows - CUDA extension won't build on Windows

## Key Workflows

### Training a Model
```bash
# GNN-based agent (node-wise features required)
python main.py --model GNN --gnn_class SAGE --features_dim 8 --node_wise \
               --pi_conv_out 4 --vf_conv_out 1 --ortho_init \
               --op_config 1 7 7 2 3 2 12 --max_episode_len 8192 \
               --total_timesteps 5e7 --n_steps 8192 --ncpu 8 --batch_size 1024 \
               --lr 1e-3 1e-4 --device cuda:0 --agent_name my_agent

# MLP-based agent (flatten features)
python main.py --model MLP --features_dim 128 --pi_net_arch 128 --vf_net_arch 64 \
               --op_config 1 4 4 1 3 1 32 --max_episode_len 128 \
               --total_timesteps 2e5 --n_steps 128 --ncpu 16 --batch_size 512 \
               --device cuda:0 --agent_name mlp_agent
```

### Evaluation
```bash
python main.py --eval_mode --agent_name 02-12_17-25-12-613_test_10000000_steps \
               --model GNN --gnn_class SAGE --device cuda:0 --verbose
```

### Generate Video from Strategy
```bash
python main.py --eval_mode --video --filename strategy_trace.log \
               --agent_name <model_name> --model GNN
```

## Project-Specific Conventions

### Configuration Patterns
- **Operator Config**: `--op_config B H W C K O S` (batch, height, width, channels, kernel, outplanes, cache_size)
  - Example: `1 7 7 2 3 2 12` creates a 7×7×2 Conv2d with 3×3 kernel, 2 output channels, cache=12
- **Reward Config**: Define via individual `--reward_*` flags in `util/parse.py`, combined into dict in `main.py`
- **Learning Rate Scheduling**: Pass 2 values `--lr 1e-3 1e-4` for linear schedule from initial to final LR

### Model Architecture Rules
- **GNN models**: Must use `--node_wise` (enforced in `util/parse.py`)
- **MLP models**: Can use `--node_wise` for per-node features or flatten globally
- **Feature extractors**: Controlled via `features_dim` (output dim) + `net_arch` (hidden layers)
- **Policy/Value heads**: Configured separately with `--pi_net_arch` / `--vf_net_arch` and `--pi_conv_out` / `--vf_conv_out`

### Action Space Design
- **Simplified**: Environment uses `spaces.Discrete(node_num)` - agent chooses node, action inferred from valid mask
- **Valid Actions**: Dynamically masked via `valid_action_map` in `RedBluePebbleGameEnv`
  - Input nodes: LOAD, DELETE
  - Mid nodes: LOAD, STORE, COMPUTE
  - Output nodes: STORE, COMPUTE

### Data Flow
1. `main.py` parses args → creates `operator_info` → generates `DAG_Graph` → builds env config
2. `RedBluePebbleGameEnv.__init__()` generates DAG, initializes state/masks
3. `CustomMaskableActorCriticPolicy` extracts features (GNN/MLP) → `CustomPolicyValueNet` → action/value heads
4. `CustomMaskablePPO.train()` collects rollouts with action masking → updates policy

## Dependencies
- **Core**: `torch`, `torch_geometric`, `torch_sparse`, `gymnasium`, `stable_baselines3`, `sb3_contrib`
- **Visualization**: `matplotlib`, `colorama`, `manim` (for video generation)
- **Build Tools**: CUDA toolkit (for `spmm_ext`)

## Debugging Tips
- Use `--debug` flag to print model architecture without training
- Set `--verbose` for environment step logging
- Check `saved_model/<model_name>/<model_name>_hParam.json` for saved hyperparameters
- TensorBoard logs in `--tb_log_dir` (default: `exp/log`)

## Important Caveats
- **Platform**: Developed on Linux, CUDA extension won't build on Windows (current environment)
- **Observation Space**: Changes based on `--obs_mode` ('vec'/'box') and `--obs_field` ('loc'/'glb')
- **Action Masking**: Always enabled - never use raw action space without masks
- **Model Loading**: Requires matching `policy_kwargs` and custom_objects (see `main.py` lines 321-327, 384-388)

## File Naming Conventions
- Models: `{timestamp}_{agent_name}_{steps}_steps.zip`
- Configs: `{model_name}_hParam.json`
- Strategies: Saved to `out/strategy/` when `--filename` specified
