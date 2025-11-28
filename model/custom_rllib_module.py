"""
CustomMlp implementation using pure PyTorch code.
This is a standalone version that doesn't depend on MlpLayer from graph_extractor.
"""

from typing import List, Any, Dict, Optional

from gymnasium import spaces
import numpy as np
import torch as th
from torch import nn
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from model.custom_policy import CustomPolicyValueNet


class AddActionMaskToBatch(ConnectorV2):
    """
    A custom ConnectorV2 that extracts action_mask from episode's info 
    and adds it to the batch.
    
    This connector reads the 'action_mask' field from the environment's info dict
    (set in RedBluePebbleGameEnv._get_info) and adds it as a new column 
    'action_mask' in the batch for the RLModule to use.
    """

    def __init__(
        self,
        input_observation_space=None,
        input_action_space=None,
        *,
        col_name: str = "action_mask",
    ):
        super().__init__(input_observation_space, input_action_space)
        self.col_name = col_name

    def __call__(self, *, episodes, batch, rl_module, explore, shared_data, **kwargs):
        """
        Extract action_mask from each episode's info and add to batch.
        
        The action_mask is stored in the episode's info dict under the key 'action_mask'.
        This is set by RedBluePebbleGameEnv._get_info() method.
        """
        for sa_episode in self.single_agent_episode_iterator(episodes):
            # Get the most recent info from the episode
            # The info dict contains 'action_mask' key set by the environment
            infos = sa_episode.get_infos(indices=-1)
            
            if infos is not None and "action_mask" in infos:
                action_mask = infos["action_mask"]
            else:
                # Fallback: if no action_mask in info, try to get from extra_model_outputs
                # or create a default mask (all actions valid)
                action_mask = np.ones(rl_module.action_space.n, dtype=np.float32)
            
            # Ensure action_mask is a numpy array with correct dtype
            if not isinstance(action_mask, np.ndarray):
                action_mask = np.array(action_mask, dtype=np.float32)
            else:
                action_mask = action_mask.astype(np.float32)
            
            # Add the action_mask to the batch
            self.add_batch_item(
                batch=batch,
                column=self.col_name,
                item_to_add=action_mask,
                single_agent_episode=sa_episode,
            )

        return batch


class CustomMlp(nn.Module):
    """
    Custom MLP feature extractor implemented with pure PyTorch.
    
    Parameters
    ----------
    observation_space : gym.Space
        The observation space of the environment
    node_num : int
        Number of nodes in the graph
    features_dim : int, default=64
        Number of features extracted (output dimension).
        This corresponds to the number of units for the last layer.
    node_wise : bool, default=True
        If True, process each node independently with the same MLP.
        If False, process the entire flattened observation.
    net_arch : list, default=[]
        Architecture of the hidden layers.
        Example: [128, 64] creates two hidden layers with 128 and 64 units.
    activation_fn : nn.Module, default=nn.ReLU
        Activation function to use after each layer
    dropout : float, default=0.0
        Dropout rate applied after each layer (except between last layer and activation)
        
    Notes
    -----
    When node_wise=True:
        - Input shape: [batch, node_num * node_features]
        - Reshapes to: [batch * node_num, node_features]
        - Applies MLP independently to each node
        - Output shape: [batch, node_num * features_dim]
        
    When node_wise=False:
        - Input shape: [batch, total_features]
        - Applies MLP to entire observation
        - Output shape: [batch, features_dim]
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        node_num: int,
        features_dim: int = 64,
        node_wise: bool = True,
        net_arch: List[int] = None,
        activation_fn: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if net_arch is None:
            net_arch = []
            
        self.node_num = node_num
        self.node_wise = node_wise
        self.dropout = dropout
        self.features_dim = features_dim
        
        # Calculate input and output dimensions
        if self.node_wise:
            in_features = observation_space.shape[0] // self.node_num
            out_features = features_dim
        else:
            in_features = observation_space.shape[0]
            out_features = features_dim
        
        # Build MLP network using pure PyTorch
        layers: List[nn.Module] = []
        last_layer_dim = in_features
        
        # Add hidden layers with activation and dropout
        for curr_layer_dim in net_arch:
            layers.append(nn.Linear(last_layer_dim, curr_layer_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_layer_dim = curr_layer_dim
        
        # Add output layer
        layers.append(nn.Linear(last_layer_dim, out_features))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(activation_fn())
        
        # Combine all layers into sequential module
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the MLP.
        
        Parameters
        ----------
        observations : th.Tensor
            Input observations with shape [batch, obs_dim] or [batch, ..., obs_dim]
            
        Returns
        -------
        th.Tensor
            Extracted features
        """
        # Flatten observations if needed
        if observations.dim() > 2:
            observations = observations.reshape(observations.shape[0], -1)
        
        # Expected shape: [batch, node_num * node_features]
        batch_size = observations.shape[0]
        
        if self.node_wise:
            # Reshape to process each node independently
            # [batch, node_num * node_features] -> [batch * node_num, node_features]
            observations = observations.reshape(batch_size * self.node_num, -1)
            
        # Apply MLP
        observations = self.mlp(observations)
        
        if self.node_wise:
            # Reshape back to batch format
            # [batch * node_num, features_dim] -> [batch, node_num * features_dim]
            observations = observations.reshape(batch_size, -1)
            
        return observations


class MyMaskableTorchRLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        # You have access here to the following already set attributes:
        # self.observation_space
        # self.action_space
        # self.inference_only
        # self.model_config  # <- a dict with custom settings

        mlp_features_extractor_config = self.model_config.get("mlp_features_extractor_config", None)

        assert mlp_features_extractor_config is not None, "mlp_features_extractor_config must be provided in model_config"
       
        # Create separate feature extractors for policy and value
        self._extract_features = CustomMlp(self.observation_space, **mlp_features_extractor_config)

        self._mlp_extractor = CustomPolicyValueNet(
            mlp_features_extractor_config["features_dim"],  # super().__init__()->make_features_extractor()->features_extractor.features_dim
            self.model_config["addi_features"],
            self.model_config["node_wise"],
            net_arch=self.model_config["net_arch"],
            activation_fn=self.model_config["activation_fn"],
            node_num=self.model_config["node_num"],
            edge_index=self.model_config["edge_index"],
            pi_conv_out=self.model_config["pi_conv_out"],
            vf_conv_out=self.model_config["vf_conv_out"],
            gp_vf=self.model_config["gp_vf"],
            transformer=self.model_config["transformer"],
            # topk_layer=self.topk_layer,
            # focus_num=self.focus_num,
            # focus_vf=self.focus_vf,
            # layerNorm=self.layerNorm,
            # norm_mode=self.norm_mode,
            # device=self.device,
        )

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Debug: print batch structure
        print("=" * 50)
        print("_forward called")
        print(f"batch type: {type(batch)}")
        print(f"batch keys: {batch.keys() if hasattr(batch, 'keys') else 'N/A'}")
        for key, value in batch.items():
            if isinstance(value, th.Tensor):
                print(f"  {key}: Tensor shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, np.ndarray):
                print(f"  {key}: ndarray shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)} = {value}")
        print(f"kwargs: {kwargs}")
        print("=" * 50)
        
        raise NotImplementedError("Debug: Check the printed batch structure above")

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        pass

    @override(ValueFunctionAPI)
    def compute_values(
        self, 
        batch: Dict[str, Any], 
        embeddings: Optional[Any] = None
    ) -> TensorType:
        pass