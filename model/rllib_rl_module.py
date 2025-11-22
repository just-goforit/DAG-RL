"""
RLlib RLModule implementation for CustomMaskableActorCriticPolicy.

This module provides a RLlib-compatible RLModule that reuses the existing
CustomGNN, CustomMlp, and CustomPolicyValueNet components while supporting
action masks.

Usage Example:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from model.rllib_rl_module import CustomMaskableTorchRLModule
    from env.graph import operator_info
    
    op_info = operator_info(...)
    graph = naive_Conv2d_DAG(op_info)
    
    config = (
        PPOConfig()
        .environment(RedBluePebbleGameEnv, env_config={"op_info": op_info})
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=CustomMaskableTorchRLModule,
                model_config={
                    "node_num": graph.node_num,
                    "edge_index": graph.edge_index,
                    "addi_features": 1,  # if obs_field == 'glb'
                    "pi_conv_out": 8,
                    "vf_conv_out": 2,
                    "gp_vf": False,
                    "transformer": False,
                    "node_wise": True,
                    "use_gnn": True,
                    "gnn_class": "SAGE",
                    "features_dim": 64,
                    "net_arch": [],
                    "activation_fn": "ReLU",
                    "dropout": 0.0,
                    "sp_tensor": False,
                    "ortho_init": True,
                }
            )
        )
    )
"""

from typing import Dict, Any, Optional, Union, Type
import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical
from functools import partial

from gymnasium import spaces

# RLlib imports
try:
    from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.utils.annotations import override
    from ray.rllib.utils.nested_dict import NestedDict
except ImportError:
    raise ImportError(
        "RLlib is not installed. Please install it with: pip install 'ray[rllib]'"
    )

# Local imports
from model.custom_policy import CustomGNN, CustomMlp
from model.policy_value import CustomPolicyValueNet


class CustomMaskableTorchRLModule(TorchRLModule):
    """
    RLlib RLModule implementation for the Red-Blue Pebble Game environment.
    
    This module reuses the existing CustomGNN/CustomMlp feature extractors
    and CustomPolicyValueNet for policy/value computation, while providing
    RLlib-compatible interfaces with action mask support.
    """
    
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # These will be initialized in setup()
        self._features_extractor = None
        self._mlp_extractor = None
        self._action_net = None
        self._value_net = None
        
        # Configuration parameters (will be set from model_config)
        self._node_num = None
        self._edge_index = None
        self._addi_features = None
        self._pi_conv_out = None
        self._vf_conv_out = None
        self._gp_vf = None
        self._transformer = None
        self._node_wise = None
        self._ortho_init = True
        self._share_features_extractor = True
        
    @override(TorchRLModule)
    def setup(self):
        """Initialize all network components based on model_config."""
        super().setup()
        
        # Extract configuration from model_config
        model_config = self.config.get("model_config", {})
        
        # Required parameters
        self._node_num = model_config.get("node_num")
        if self._node_num is None:
            raise ValueError("model_config must contain 'node_num'")
            
        self._edge_index = model_config.get("edge_index")
        if self._edge_index is None:
            raise ValueError("model_config must contain 'edge_index'")
        if isinstance(self._edge_index, np.ndarray):
            self._edge_index = th.from_numpy(self._edge_index).long()
        
        # Optional parameters with defaults
        self._addi_features = model_config.get("addi_features", 0)
        self._pi_conv_out = model_config.get("pi_conv_out", 8)
        self._vf_conv_out = model_config.get("vf_conv_out", 2)
        self._gp_vf = model_config.get("gp_vf", False)
        self._transformer = model_config.get("transformer", False)
        self._node_wise = model_config.get("node_wise", True)
        self._ortho_init = model_config.get("ortho_init", True)
        self._share_features_extractor = model_config.get("share_features_extractor", True)
        
        # Feature extractor configuration
        features_dim = model_config.get("features_dim", 64)
        gnn_class = model_config.get("gnn_class", "SAGE")
        node_features = model_config.get("node_features", 4)
        dropout = model_config.get("dropout", 0.0)
        sp_tensor = model_config.get("sp_tensor", False)
        activation_fn_str = model_config.get("activation_fn", "ReLU")
        
        # Parse activation function
        activation_fn = self._get_activation_fn(activation_fn_str)
        
        # Determine which feature extractor to use
        use_gnn = model_config.get("use_gnn", True)
        
        if use_gnn:
            # Use CustomGNN
            # Convert edge_index to numpy for CustomGNN (it expects np.ndarray)
            edge_index_np = self._edge_index.numpy() if isinstance(self._edge_index, th.Tensor) else self._edge_index
            if isinstance(edge_index_np, th.Tensor):
                edge_index_np = edge_index_np.cpu().numpy()
            
            self._features_extractor = CustomGNN(
                observation_space=self.config.observation_space,
                edge_index=edge_index_np,
                node_num=self._node_num,
                node_features=node_features,
                addi_features=self._addi_features,
                gnn_class=gnn_class,
                features_dim=features_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                sp_tensor=sp_tensor,
                device=self.device,
            )
        else:
            # Use CustomMlp
            net_arch = model_config.get("net_arch", [])
            self._features_extractor = CustomMlp(
                observation_space=self.config.observation_space,
                node_num=self._node_num,
                features_dim=features_dim,
                node_wise=self._node_wise,
                net_arch=net_arch,
                activation_fn=activation_fn,
                dropout=dropout,
            )
        
        # Initialize MLP extractor (policy/value head)
        net_arch = model_config.get("net_arch", [])
        # Convert edge_index to numpy for CustomPolicyValueNet (it expects np.ndarray)
        edge_index_np = self._edge_index.numpy() if isinstance(self._edge_index, th.Tensor) else self._edge_index
        if isinstance(edge_index_np, th.Tensor):
            edge_index_np = edge_index_np.cpu().numpy()
        
        self._mlp_extractor = CustomPolicyValueNet(
            feature_dim=features_dim,
            addi_features=self._addi_features,
            node_wise=self._node_wise,
            net_arch=net_arch,
            activation_fn=activation_fn,
            node_num=self._node_num,
            edge_index=edge_index_np,
            pi_conv_out=self._pi_conv_out,
            vf_conv_out=self._vf_conv_out,
            transformer=self._transformer,
            gp_vf=self._gp_vf,
            device=self.device,
        )
        
        # Initialize action and value networks
        latent_dim_pi = self._mlp_extractor.latent_dim_pi
        latent_dim_vf = self._mlp_extractor.latent_dim_vf
        
        if latent_dim_pi == 0:
            self._action_net = None
        else:
            self._action_net = nn.Linear(latent_dim_pi, self.config.action_space.n)
        
        self._value_net = nn.Linear(latent_dim_vf, 1)
        
        # Apply orthogonal initialization if enabled
        if self._ortho_init:
            self._apply_ortho_init()
        
        # Store edge_index for reference (CustomGNN manages its own copy)
        # Note: CustomGNN expects numpy array in __init__, but manages tensor internally
    
    def _get_activation_fn(self, activation_fn_str: str) -> Type[nn.Module]:
        """Convert activation function string to class."""
        activation_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "GELU": nn.GELU,
            "LeakyReLU": nn.LeakyReLU,
        }
        if activation_fn_str not in activation_map:
            raise ValueError(f"Unknown activation function: {activation_fn_str}")
        return activation_map[activation_fn_str]
    
    def _apply_ortho_init(self):
        """Apply orthogonal initialization to network modules."""
        def init_weights(module, gain=1.0):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        module_gains = {
            self._features_extractor: np.sqrt(2),
            self._mlp_extractor: np.sqrt(2),
            self._action_net: 0.01 if self._action_net is not None else None,
            self._value_net: 1.0,
        }
        
        for module, gain in module_gains.items():
            if module is not None and gain is not None:
                module.apply(partial(init_weights, gain=gain))
    
    def _extract_features(self, obs: th.Tensor) -> th.Tensor:
        """Extract features from observations using the feature extractor."""
        # Handle device synchronization for edge_index if using GNN
        if isinstance(self._features_extractor, CustomGNN):
            # CustomGNN handles device synchronization internally in its forward method
            # We just need to ensure the feature extractor's device matches obs device
            if hasattr(self._features_extractor, 'device') and obs.device != self._features_extractor.device:
                self._features_extractor.device = obs.device
                # Update edge_index in feature extractor
                if hasattr(self._features_extractor, 'edge_index'):
                    # Convert to tensor if needed
                    if isinstance(self._features_extractor.edge_index, np.ndarray):
                        self._features_extractor.edge_index = th.from_numpy(
                            self._features_extractor.edge_index
                        ).to(obs.device)
                    else:
                        self._features_extractor.edge_index = self._features_extractor.edge_index.to(obs.device)
        
        return self._features_extractor(obs)
    
    def _apply_action_mask(self, logits: th.Tensor, action_mask: Optional[th.Tensor]) -> th.Tensor:
        """
        Apply action mask to logits by setting invalid actions to negative infinity.
        
        Args:
            logits: Action logits tensor of shape [batch_size, action_space.n]
            action_mask: Action mask tensor of shape [batch_size, action_space.n] or None
            
        Returns:
            Masked logits tensor
        """
        if action_mask is not None:
            # Convert mask to tensor if needed
            if isinstance(action_mask, np.ndarray):
                action_mask = th.from_numpy(action_mask).to(logits.device)
            
            # Ensure mask is boolean or float
            if action_mask.dtype != th.bool:
                action_mask = action_mask.bool()
            
            # Set invalid actions to negative infinity
            logits = logits.masked_fill(~action_mask, float("-inf"))
        
        return logits
    
    def _get_action_dist_inputs(self, latent_pi: th.Tensor, action_mask: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Get action distribution inputs (logits) from latent policy vector.
        
        Args:
            latent_pi: Latent policy vector
            action_mask: Optional action mask
            
        Returns:
            Action logits (possibly masked)
        """
        if self._action_net is not None:
            logits = self._action_net(latent_pi)
        else:
            logits = latent_pi
        
        # Apply action mask
        logits = self._apply_action_mask(logits, action_mask)
        
        return logits
    
    @override(TorchRLModule)
    def _forward_inference(self, batch: NestedDict, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for inference (evaluation).
        
        Args:
            batch: Batch of data containing observations and optionally action masks
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing action distribution inputs and values
        """
        obs = batch[Columns.OBS]
        action_mask = batch.get(Columns.ACTION_MASK, None)
        
        # Extract features
        features = self._extract_features(obs)
        
        # Get policy and value latents
        if self._share_features_extractor:
            latent_pi, latent_vf = self._mlp_extractor(features)
        else:
            # If not sharing, we'd need separate feature extractors
            # For now, assume sharing
            latent_pi, latent_vf = self._mlp_extractor(features)
        
        # Get action distribution inputs
        action_dist_inputs = self._get_action_dist_inputs(latent_pi, action_mask)
        
        # Get values
        values = self._value_net(latent_vf)
        
        return {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.VF_PREDS: values,
        }
    
    @override(TorchRLModule)
    def _forward_exploration(self, batch: NestedDict, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for exploration (sampling actions).
        
        Args:
            batch: Batch of data containing observations and optionally action masks
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing action distribution inputs, values, and action log probs
        """
        obs = batch[Columns.OBS]
        action_mask = batch.get(Columns.ACTION_MASK, None)
        
        # Extract features
        features = self._extract_features(obs)
        
        # Get policy and value latents
        if self._share_features_extractor:
            latent_pi, latent_vf = self._mlp_extractor(features)
        else:
            latent_pi, latent_vf = self._mlp_extractor(features)
        
        # Get action distribution inputs
        action_dist_inputs = self._get_action_dist_inputs(latent_pi, action_mask)
        
        # Get values
        values = self._value_net(latent_vf)
        
        # Sample actions and compute log probs
        dist = Categorical(logits=action_dist_inputs)
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)
        
        return {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.VF_PREDS: values,
            Columns.ACTIONS: actions,
            Columns.ACTION_LOGP: action_log_probs,
        }
    
    @override(TorchRLModule)
    def _forward_train(self, batch: NestedDict, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for training.
        
        Args:
            batch: Batch of data containing observations, actions, and optionally action masks
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing action distribution inputs, values, and action log probs
        """
        obs = batch[Columns.OBS]
        actions = batch.get(Columns.ACTIONS, None)
        action_mask = batch.get(Columns.ACTION_MASK, None)
        
        # Extract features
        features = self._extract_features(obs)
        
        # Get policy and value latents
        if self._share_features_extractor:
            latent_pi, latent_vf = self._mlp_extractor(features)
        else:
            latent_pi, latent_vf = self._mlp_extractor(features)
        
        # Get action distribution inputs
        action_dist_inputs = self._get_action_dist_inputs(latent_pi, action_mask)
        
        # Get values
        values = self._value_net(latent_vf)
        
        # Compute action log probs if actions are provided
        action_log_probs = None
        if actions is not None:
            dist = Categorical(logits=action_dist_inputs)
            action_log_probs = dist.log_prob(actions)
        
        result = {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.VF_PREDS: values,
        }
        
        if action_log_probs is not None:
            result[Columns.ACTION_LOGP] = action_log_probs
        
        return result

