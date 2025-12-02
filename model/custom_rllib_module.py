"""
CustomMlp implementation using pure PyTorch code.
This is a standalone version that doesn't depend on MlpLayer from graph_extractor.
"""

from typing import List, Any, Dict, Optional, Union

from gymnasium import spaces
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_sparse import SparseTensor

from model.custom_policy import CustomPolicyValueNet
from model.graph_extractor import GNNExtractor


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

        self._action_net = nn.Linear(self._mlp_extractor.latent_dim_pi, self.action_space.n)
        self._value_net = nn.Linear(self._mlp_extractor.latent_dim_vf, 1)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        """
        Forward pass for inference/exploration.
        
        Computes actions from observations with optional action masking.
        Uses pure PyTorch to implement masked action sampling.
        """
        obs = batch[Columns.OBS]
        action_mask = batch.get("action_mask", None)

        # Extract features and get latent representations
        features = self._extract_features(obs)
        latent_pi, _ = self._mlp_extractor(features)

        # Get action logits from action network
        action_logits = self._action_net(latent_pi)

        # Apply action masking: set invalid actions' logits to -inf
        if action_mask is not None:
            # Ensure action_mask is a tensor with correct dtype
            if not isinstance(action_mask, th.Tensor):
                action_mask = th.tensor(action_mask, dtype=th.float32, device=action_logits.device)
            # Set invalid actions (mask=0) to very large negative value
            HUGE_NEG = th.finfo(action_logits.dtype).min
            action_logits = th.where(
                action_mask.bool(),
                action_logits,
                th.full_like(action_logits, HUGE_NEG),
            )

        # Sample actions from the masked distribution (exploration)
        # Use categorical distribution for discrete action space
        probs = F.softmax(action_logits, dim=-1)
        actions = th.multinomial(probs, num_samples=1).squeeze(-1) # 被选择的节点

        # 必须返回当前动作的对数概率，供采样批次存储
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1).long()).squeeze(-1)

        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.ACTION_LOGP: action_log_probs,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        """
        Forward pass for training.
        
        Computes action distribution, log probabilities, entropy, and values
        for PPO loss computation.
        """
        obs = batch[Columns.OBS]
        actions = batch[Columns.ACTIONS]
        action_mask = batch.get("action_mask", None)

        # Extract features and get latent representations
        features = self._extract_features(obs)
        latent_pi, latent_vf = self._mlp_extractor(features)

        # Get action logits from action network
        action_logits = self._action_net(latent_pi)

        # Apply action masking: set invalid actions' logits to -inf
        if action_mask is not None:
            if not isinstance(action_mask, th.Tensor):
                action_mask = th.tensor(action_mask, dtype=th.float32, device=action_logits.device)
            HUGE_NEG = th.finfo(action_logits.dtype).min
            action_logits = th.where(
                action_mask.bool(),
                action_logits,
                th.full_like(action_logits, HUGE_NEG),
            )

        # Compute log probabilities and entropy
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1).long()).squeeze(-1)
        
        # Compute entropy: -sum(p * log(p))
        # probs = F.softmax(action_logits, dim=-1)
        # entropy = -th.sum(probs * log_probs, dim=-1)

        # Compute values
        values = self._value_net(latent_vf).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.ACTION_LOGP: action_log_probs,
            # Columns.ENTROPY: entropy,
            Columns.VF_PREDS: values,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, 
        batch: Dict[str, Any], 
        embeddings: Optional[Any] = None
    ) -> TensorType:
        """
        Compute value function predictions for given observations.
        """
        obs = batch[Columns.OBS]
        
        # Extract features and get value latent representation
        features = self._extract_features(obs)
        _, latent_vf = self._mlp_extractor(features)
        
        # Compute values
        values = self._value_net(latent_vf).squeeze(-1)
        
        return values
    

class CustomGNN(nn.Module):
    """
    params
    ------
    * observation_space: (gym.Space)
    * node_features: (int) Number of node features
    * features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    * gnn_class: (str) GNN model type, GCN/GAT
    * edge_index: (np.ndarray) The edge index of graph.
    * hidden_features: (int) The hidden features of hidden GCN/GATLayer.
    * blocks: (int) The number of GCN/GATLayer.
    * dropout: (float) The dropout rate of GCN/GATLayer.
    * layerNorm: (bool) Whether to use LayerNorm.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        edge_index: np.ndarray,
        node_num: int,
        node_features: int = 4,  # max(IN, MID, OUT) node features
        addi_features: int = 0,
        gnn_class: str = "SAGE",
        features_dim: int = 64,
        dropout: float = 0.0,
        #  hidden_features: int=16,
        #  blocks: int=0,
        activation_fn: nn.Module = nn.ReLU,
        #  net_arch:list=[],
        #  layerNorm: bool=False,
        #  norm_mode: str='node',
        sp_tensor: bool = False,
        device: Union[th.device, str] = "cpu",
    ):
        super().__init__()
        self.device = device  # model training device, edge_index device
        self.node_num = node_num
        self.edge_index = th.from_numpy(edge_index).to(device)
        self.sp_tensor = sp_tensor
        if self.sp_tensor:
            # self.edge_index_csr for spmm_ext
            # self.edge_index_sparse_t for Pyg
            self.update_sp_edge_index()
            self.adj = None
        else:
            # self.adj
            self.update_edge_index2adj()
            self.edge_index_csr = None
            self.edge_index_sparse_t = None

        self.node_features = node_features
        self.addi_features = addi_features
        self.gnn_class = gnn_class
        if isinstance(gnn_class, str):
            self.gnn_class = self._get_gnn_from_name(gnn_class)

        if addi_features > 0:
            self.lin_addi = nn.Sequential(
                nn.Linear(addi_features, addi_features), nn.LeakyReLU()
            )
        else:
            self.lin_addi = None

        self.gnn = self.gnn_class(
            in_features=node_features,  # node_nums * node_features
            out_features=features_dim,  # pass to policy/value
            activation_fn=activation_fn,
            dropout=dropout,
            sp_tensor=sp_tensor,
        )

    def _get_gnn_from_name(self, type_name: str):
        # gnn_models = {
        #     "GAT": GATExtractor,
        #     "GCN": GCNExtractor,
        #     "SAGE": SAGEExtractor,
        # }
        # gnn_models = {
        #     "SAGE": GNNExtractor,
        # }
        gnn_types = ["GAT", "SAGE"]
        if type_name in gnn_types:
            return GNNExtractor
        else:
            raise ValueError(f"Policy {type_name} unknown")

    def update_edge_index2adj(self, edge_index=None):
        if edge_index is None:
            edge_index = self.edge_index
        if isinstance(edge_index, np.ndarray):
            self.edge_index = th.from_numpy(edge_index).to(self.device)
        self.adj = th.zeros(
            (self.node_num, self.node_num),
            dtype=th.float32,
            device=self.device,
            requires_grad=False,
        )
        # self.adj[edge_index[0], edge_index[1]] = 1.0 # successor -> self
        self.adj[edge_index[1], edge_index[0]] = 1.0  # predecessor -> self
        # self.adj[th.arange(self.node_num), th.arange(self.node_num)] = 1.0 # self -> self
        sum_ = self.adj.sum(dim=1, keepdim=True)
        sum_[sum_ < 1.0] = 1.0  # avoid div zero
        self.adj = self.adj / sum_
        # self.adj[th.arange(self.node_num), th.arange(self.node_num)] = 0.0 # self -> self

    def _get_csr_value(self, edge_index: th.Tensor, device: str = "cpu"):
        sorted_, _ = th.sort(edge_index[1])
        _, counts = th.unique(sorted_, return_counts=True)
        e_value = th.cat(
            [
                th.full(size=(count,), fill_value=1.0 / value, dtype=th.float32)
                for value, count in zip(counts, counts)
            ],
            dim=0,
        ).to(device)
        return e_value

    def edge_index_sparse(self, edge_index: th.Tensor, device: str = "cpu"):
        edge_index_ = edge_index.to(th.int64).to(device)
        _sp = SparseTensor(
            row=edge_index_[1],
            col=edge_index_[0],
            sparse_sizes=(self.node_num, self.node_num),
        )
        return _sp  # _sp.csr()

    def update_sp_edge_index(self, edge_index=None):
        """
        update edge_index to edge_index_sparse_t & edge_index_csr
        """
        if edge_index is None:
            edge_index = self.edge_index
        if isinstance(edge_index, np.ndarray):
            self.edge_index = th.from_numpy(edge_index).to(self.device)
        self.edge_index_sparse_t = self.edge_index_sparse(edge_index, self.device)
        row_off, col_ind, _ = self.edge_index_sparse_t.csr()
        e_v = self._get_csr_value(edge_index, self.device)
        self.edge_index_csr = (
            e_v,
            row_off.to(dtype=th.int32),
            col_ind.to(dtype=th.int32),
            th.arange(edge_index.shape[1], dtype=th.int32, device=self.device),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        assert observations.dim() == 2
        # if self.conv: # weighted-sum of history state
        #     _obs_state = self.conv(_obs_state)
        #     # _obs_state.shape = [batch, 1, node_num, node_features]
        #     obs_addi = self.conv(obs_addi)
        #     # obs_addi.shape = [batch, 1, addi_features, 1]

        # observations.shape=(observations.shape[0], self.node_num * self.node_features + ADD_FEATURE)
        _obs_addi = None
        if self.addi_features > 0:
            _obs_addi = observations[:, 0 : self.addi_features].reshape(
                observations.shape[0], self.addi_features
            )
        # obs_addi.shape = [batch, addi_features]

        _obs_state = observations[:, self.addi_features :].reshape(
            observations.shape[0], self.node_num, self.node_features
        )
        # _obs_state.shape = [batch, node_num, node_features]

        if self.device != observations.device:
            self.device = observations.device
            self.edge_index = self.edge_index.to(observations.device)
            if self.sp_tensor:
                self.update_sp_edge_index()
            else:
                self.update_edge_index2adj()
        if self.gnn_class == "SAGE":
            obs_state: th.Tensor = self.gnn(
                _obs_state, self.edge_index_csr if self.sp_tensor else self.adj
            )  # use spmm_ext
        else:
            obs_state: th.Tensor = self.gnn(
                _obs_state, self.edge_index_sparse_t if self.sp_tensor else self.adj
            )  # use pyg

        obs_state = obs_state.transpose(-1, -2).reshape(obs_state.shape[0], -1)
        # obs_state.shape=[batch, (features_dim, node_nums)]
        obs = obs_state
        if self.addi_features > 0:
            obs_addi = _obs_addi
            # obs_addi = self.lin_addi(_obs_addi)
            obs = th.cat((obs_state, obs_addi), dim=1)
        # obs.shape=[batch, -1]
        return obs


class MyMaskableTorchRLModuleGNN(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        # You have access here to the following already set attributes:
        # self.observation_space
        # self.action_space
        # self.inference_only
        # self.model_config  # <- a dict with custom settings

        gnn_features_extractor_config = self.model_config.get("gnn_features_extractor_config", None)

        assert gnn_features_extractor_config is not None, "gnn_features_extractor_config must be provided in model_config"
       
        # Create separate feature extractors for policy and value
        self._extract_features = CustomGNN(self.observation_space, **gnn_features_extractor_config)

        self._mlp_extractor = CustomPolicyValueNet(
            gnn_features_extractor_config["features_dim"],  # super().__init__()->make_features_extractor()->features_extractor.features_dim
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

        self._action_net = nn.Linear(self._mlp_extractor.latent_dim_pi, self.action_space.n)
        self._value_net = nn.Linear(self._mlp_extractor.latent_dim_vf, 1)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        """
        Forward pass for inference/exploration.
        
        Computes actions from observations with optional action masking.
        Uses pure PyTorch to implement masked action sampling.
        """
        obs = batch[Columns.OBS]
        action_mask = batch.get("action_mask", None)

        # Extract features and get latent representations
        features = self._extract_features(obs)
        latent_pi, _ = self._mlp_extractor(features)

        # Get action logits from action network
        action_logits = self._action_net(latent_pi)

        # Apply action masking: set invalid actions' logits to -inf
        if action_mask is not None:
            # Ensure action_mask is a tensor with correct dtype
            if not isinstance(action_mask, th.Tensor):
                action_mask = th.tensor(action_mask, dtype=th.float32, device=action_logits.device)
            # Set invalid actions (mask=0) to very large negative value
            HUGE_NEG = th.finfo(action_logits.dtype).min
            action_logits = th.where(
                action_mask.bool(),
                action_logits,
                th.full_like(action_logits, HUGE_NEG),
            )

        # Sample actions from the masked distribution (exploration)
        # Use categorical distribution for discrete action space
        probs = F.softmax(action_logits, dim=-1)
        actions = th.multinomial(probs, num_samples=1).squeeze(-1) # 被选择的节点

        # 必须返回当前动作的对数概率，供采样批次存储
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1).long()).squeeze(-1)

        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.ACTION_LOGP: action_log_probs,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        """
        Forward pass for training.
        
        Computes action distribution, log probabilities, entropy, and values
        for PPO loss computation.
        """
        obs = batch[Columns.OBS]
        actions = batch[Columns.ACTIONS]
        action_mask = batch.get("action_mask", None)

        # Extract features and get latent representations
        features = self._extract_features(obs)
        latent_pi, latent_vf = self._mlp_extractor(features)

        # Get action logits from action network
        action_logits = self._action_net(latent_pi)

        # Apply action masking: set invalid actions' logits to -inf
        if action_mask is not None:
            if not isinstance(action_mask, th.Tensor):
                action_mask = th.tensor(action_mask, dtype=th.float32, device=action_logits.device)
            HUGE_NEG = th.finfo(action_logits.dtype).min
            action_logits = th.where(
                action_mask.bool(),
                action_logits,
                th.full_like(action_logits, HUGE_NEG),
            )

        # Compute log probabilities and entropy
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1).long()).squeeze(-1)
        
        # Compute entropy: -sum(p * log(p))
        # probs = F.softmax(action_logits, dim=-1)
        # entropy = -th.sum(probs * log_probs, dim=-1)

        # Compute values
        values = self._value_net(latent_vf).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.ACTION_LOGP: action_log_probs,
            # Columns.ENTROPY: entropy,
            Columns.VF_PREDS: values,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, 
        batch: Dict[str, Any], 
        embeddings: Optional[Any] = None
    ) -> TensorType:
        """
        Compute value function predictions for given observations.
        """
        obs = batch[Columns.OBS]
        
        # Extract features and get value latent representation
        features = self._extract_features(obs)
        _, latent_vf = self._mlp_extractor(features)
        
        # Compute values
        values = self._value_net(latent_vf).squeeze(-1)
        
        return values