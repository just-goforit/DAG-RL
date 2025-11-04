from typing import Callable, Optional, Tuple, Union, Dict

from gymnasium import spaces
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
# from stable_baselines3.common.policies import ActorCriticPolicy

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import Schedule

from model.policy_value import CustomPolicyValueNet
# from model.graph_extractor import GATExtractor, GCNExtractor, SAGEExtractor, 
from model.graph_extractor import MlpLayer, GNNExtractor
from torch_sparse import SparseTensor

from functools import partial

class CustomMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        node_num: int,
        edge_index:np.ndarray,
        addi_features:int, 
        pi_conv_out: int,
        vf_conv_out: int,
        gp_vf: bool,
        transformer:bool,
        # topk_layer: int,
        # focus_num: int,
        # focus_vf: bool,
        # focus_ngn: bool,
        # show_focus: bool,
        # layerNorm: bool,
        # norm_mode: str,
        node_wise: bool,
        *args,
        **kwargs,
    ):
        self.node_num = node_num
        self.edge_index = edge_index
        self.addi_features = addi_features
        self.pi_conv_out = pi_conv_out
        self.vf_conv_out = vf_conv_out
        self.gp_vf = gp_vf
        self.transformer=transformer
        # self.topk_layer = topk_layer
        # self.focus_num = focus_num
        # self.focus_vf = focus_vf
        # self.focus_ngn=focus_ngn
        # self.show_focus = show_focus
        # self.ncloss = ncloss
        # self.ncloss_tau = ncloss_tau
        # self.layerNorm = layerNorm
        # self.norm_mode = norm_mode
        self.node_wise = node_wise
     
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        self.normalize_images = False # avoid bugs

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomPolicyValueNet(
            self.features_dim, # super().__init__()->make_features_extractor()->features_extractor.features_dim
            self.addi_features,
            self.node_wise,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            node_num=self.node_num,
            edge_index=self.edge_index,
            pi_conv_out=self.pi_conv_out,
            vf_conv_out=self.vf_conv_out,
            gp_vf=self.gp_vf,
            transformer=self.transformer,
            # topk_layer=self.topk_layer,
            # focus_num=self.focus_num,
            # focus_vf=self.focus_vf,
            # layerNorm=self.layerNorm,
            # norm_mode=self.norm_mode,
            device=self.device,
        )
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits = latent_pi
        if self.action_net is not None:
            action_logits:th.Tensor = self.action_net(latent_pi)
        # expect focus_logits.shape=[batch, focus_num * action_dim]
        # if self.mlp_extractor.spotlight:
        #     focus_logits = action_logits.reshape(latent_pi.shape[0], self.focus_num, -1)
        #     action_logits = th.zeros((latent_pi.shape[0], self.node_num, focus_logits.shape[-1]), 
        #                              dtype=latent_pi.dtype, 
        #                              device=self.device)
        #     focus_idx = self.mlp_extractor.focus_idx.squeeze(1).squeeze(-1)
        #     # focus_idx.shape=[batch, focus_num]
        #     for i in range(focus_logits.shape[0]):
        #         action_logits[i, focus_idx[i, :], :] = focus_logits[i]
        #     action_logits=action_logits.reshape(action_logits.shape[0], -1)
        
        return self.action_dist.proba_distribution(action_logits=action_logits)
    
    # _predict()->get_distribution()->get_actions()
    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        # print('gnn features:', features.reshape(-1, self.features_extractor.features_dim))
        latent_pi, _ = self.mlp_extractor(features)
        # if self.mlp_extractor.focus_idx is not None:
        #     focus_idx = self.mlp_extractor.focus_idx.squeeze(-1).squeeze(1)
        #     for b in range(focus_idx.shape[0]):
        #         print(f'batch {b} focus:', focus_idx[b])

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution
    
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        act_per_node = self.action_space.n // self.node_num
        # self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, act_per_node * self.focus_num if self.focus_num > 0 else self.action_space.n)
        self.action_net = None if self.mlp_extractor.latent_dim_pi == 0 else nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.n)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            
            if self.action_net is None:
                del module_gains[self.action_net]
                
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        
        # focus_loss = None
        # if self.mlp_extractor.focus_score is not None:
        #     action_masks = action_masks.to(dtype = th.float32).reshape(action_masks.shape[0], self.node_num, -1)
        #     # focus_score shape = [batch_size, 1, focus_num, 1]
        #     focus_score = self.mlp_extractor.focus_score.squeeze(1).squeeze(-1)
        #     focus_score = F.sigmoid(focus_score)
            
        #     idx = self.mlp_extractor.focus_idx.squeeze(1).squeeze(-1)
        #     focus_tar = th.gather(th.sum(action_masks, dim=2), dim=1, index=idx)
        #     delta = focus_score - focus_tar
        #     if self.focus_ngn:
        #         ng_idx = th.zeros_like(focus_tar, dtype=th.float32, device=focus_tar.device)
        #         is_close = th.isclose(focus_tar, ng_idx, rtol=1e-05, atol=1e-08)
        #         delta[~is_close] = 0
        #     focus_loss = th.mean((delta) ** 2)
        return values, log_prob, distribution.entropy()
    
    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation, action_masks).get_actions(deterministic=deterministic)
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :param action_masks: Action masks to apply to the action distribution
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic, action_masks=action_masks)
            
            # if self.show_focus and isinstance(self.mlp_extractor, CustomGNN) and self.focus_num > 0:
            #     focus_idx:th.Tensor = self.mlp_extractor.focus_idx.squeeze(1).squeeze(-1)# focus_idx.shape=[batch, focus_num]
            #     if th.any(focus_idx == actions):
            #         print('focus right')
            #     else:
            #         print('focus:', focus_idx)
            #         print('focus wrong')
            
            # Convert to numpy
            actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions.squeeze(axis=0)

        return actions, None
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)
    
    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

class CustomGNN(BaseFeaturesExtractor):
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
    def __init__(self, 
                 observation_space: spaces.Dict, 
                 edge_index:np.ndarray,
                 node_num:int,
                 node_features:int = 4, # max(IN, MID, OUT) node features
                 addi_features:int = 0, 
                 gnn_class:str = 'SAGE',
                 features_dim: int = 64, 
                 dropout: float=0., 
                #  hidden_features: int=16, 
                #  blocks: int=0,
                 activation_fn:nn.Module=nn.ReLU,
                #  net_arch:list=[],
                #  layerNorm: bool=False,
                #  norm_mode: str='node',
                 sp_tensor:bool=False,
                 device: Union[th.device, str] = "cpu"):
        super().__init__(observation_space, features_dim) 
        self.device = device # model training device, edge_index device
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
            self.lin_addi = nn.Sequential(nn.Linear(addi_features, addi_features), 
                                          nn.LeakyReLU())
        else:
            self.lin_addi = None
        
        self.gnn = self.gnn_class(in_features=node_features, # node_nums * node_features
                                  out_features=features_dim, # pass to policy/value
                                  activation_fn=activation_fn,
                                  dropout=dropout,
                                  sp_tensor=sp_tensor)
        
    def _get_gnn_from_name(self, type_name:str):
        # gnn_models = {
        #     "GAT": GATExtractor,
        #     "GCN": GCNExtractor,
        #     "SAGE": SAGEExtractor,
        # }
        # gnn_models = {
        #     "SAGE": GNNExtractor,
        # }
        gnn_types = ['GAT', 'SAGE']
        if type_name in gnn_types:
            return GNNExtractor
        else:
            raise ValueError(f"Policy {type_name} unknown")
        
    def update_edge_index2adj(self, edge_index=None):
        if edge_index is None:
            edge_index = self.edge_index
        if isinstance(edge_index, np.ndarray):
            self.edge_index = th.from_numpy(edge_index).to(self.device)
        self.adj = th.zeros((self.node_num, self.node_num), dtype=th.float32, device=self.device, requires_grad=False)
        # self.adj[edge_index[0], edge_index[1]] = 1.0 # successor -> self
        self.adj[edge_index[1], edge_index[0]] = 1.0 # predecessor -> self
        # self.adj[th.arange(self.node_num), th.arange(self.node_num)] = 1.0 # self -> self
        sum_ = self.adj.sum(dim=1, keepdim=True)
        sum_[sum_ < 1.0] = 1.0 # avoid div zero
        self.adj = self.adj / sum_
        # self.adj[th.arange(self.node_num), th.arange(self.node_num)] = 0.0 # self -> self
    
    def _get_csr_value(self, edge_index:th.Tensor, device:str='cpu'):
        sorted_, _ = th.sort(edge_index[1])
        _, counts = th.unique(sorted_, return_counts=True)
        e_value = th.cat([th.full(size=(count,), 
                                        fill_value = 1.0/value, 
                                        dtype=th.float32) for value, count in zip(counts, counts)], dim=0).to(device)
        return e_value
    
    def edge_index_sparse(self, edge_index:th.Tensor, device:str='cpu'):
        edge_index_ = edge_index.to(th.int64).to(device)
        _sp = SparseTensor(row=edge_index_[1], col=edge_index_[0], sparse_sizes=(self.node_num, self.node_num))
        return _sp # _sp.csr()
    
    def update_sp_edge_index(self, edge_index=None):
        '''
        update edge_index to edge_index_sparse_t & edge_index_csr
        '''
        if edge_index is None:
            edge_index = self.edge_index
        if isinstance(edge_index, np.ndarray):
            self.edge_index = th.from_numpy(edge_index).to(self.device)
        self.edge_index_sparse_t = self.edge_index_sparse(edge_index, self.device)
        row_off, col_ind, _ = self.edge_index_sparse_t.csr()
        e_v = self._get_csr_value(edge_index, self.device)
        self.edge_index_csr = (e_v, 
                               row_off.to(dtype=th.int32), 
                               col_ind.to(dtype=th.int32), 
                               th.arange(edge_index.shape[1], dtype=th.int32, device=self.device))
        
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
            _obs_addi = observations[:, 0: self.addi_features].reshape(observations.shape[0], self.addi_features)
        # obs_addi.shape = [batch, addi_features]
        
        _obs_state = observations[:, self.addi_features:].reshape(observations.shape[0], self.node_num, self.node_features)
        # _obs_state.shape = [batch, node_num, node_features]

        if self.device != observations.device:
            self.device = observations.device
            self.edge_index = self.edge_index.to(observations.device)
            if self.sp_tensor:
                self.update_sp_edge_index()
            else:
                self.update_edge_index2adj()
        if self.gnn_class == 'SAGE':
            obs_state:th.Tensor = self.gnn(_obs_state, self.edge_index_csr if self.sp_tensor else self.adj) # use spmm_ext
        else:
            obs_state:th.Tensor = self.gnn(_obs_state, self.edge_index_sparse_t if self.sp_tensor else self.adj) # use pyg
            
        obs_state = obs_state.transpose(-1, -2).reshape(obs_state.shape[0], -1)
        # obs_state.shape=[batch, (features_dim, node_nums)]
        obs = obs_state
        if self.addi_features > 0:
            obs_addi = _obs_addi
            # obs_addi = self.lin_addi(_obs_addi)
            obs = th.cat((obs_state, obs_addi), dim=1)
        # obs.shape=[batch, -1]
        return obs

class CustomMlp(BaseFeaturesExtractor):
    """
    params
    ------
    * observation_space: (gym.Space)
    * node_num
    * node_features
    * features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    * net_arch: [(node_features * node_num), net_arch[0]], [net_arch[1], net_arch[2]], ... [net_arch[-1], out_features]
    * activation_fn
    """

    def __init__(self, 
                 observation_space: spaces.Box, 
                 node_num:int,
                #  edge_index:np.ndarray,
                #  node_features:int,
                #  addi_features:int=1,
                 features_dim: int = 64, # out features per node, pass to policy/value
                 node_wise: bool=True,
                #  ncloss: bool=False,
                #  r_pow:int = 2,
                 net_arch:list=[],
                 activation_fn:nn.Module=nn.ReLU,
                 dropout: float=0.,
                #  layerNorm: bool=False,
                #  batchNorm: bool=False
                 ):
        super().__init__(observation_space, features_dim)
        self.node_num = node_num
        self.node_wise = node_wise
        # self.r_pow = r_pow
        self.edge_index = None # th.from_numpy(edge_index).to(device)
        # if ncloss:
        #     # edge_index -> r_adjacency matrix
        #     self.edge_index = th.zeros((node_num, node_num), device=device)
        #     self.edge_index[edge_index[0], edge_index[1]] = 1 # to adjacency matrix
        #     self.edge_index[edge_index[1], edge_index[0]] = 1 # to symmetrize
        #     self.edge_index=self._adj_norm(self.edge_index)
        #     self.edge_index=self._get_r_adj(self.edge_index) # to r_adjacency matrix
        self.mlp = MlpLayer(in_features=observation_space.shape[0] // self.node_num if self.node_wise else observation_space.shape[0],
                            out_features=features_dim if node_wise else features_dim,
                            hidden_arch=list(map(lambda x: x, net_arch)) if node_wise else net_arch,
                            activation_fn=activation_fn,
                            dropout=dropout,
                            #   layerNorm=layerNorm,
                            #   batchNorm=batchNorm
                            )
    
    # def _adj_norm(self, adj:th.Tensor):
    #     """
    #     A -> A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    #     f -(row_norm)> f'
    #     """
    #     adj = adj + th.eye(adj.shape[0], device=adj.device)
    #     row_sum_sqrt = th.pow(th.sum(adj, dim=-1),-0.5).flatten()
    #     deg = th.where(th.isinf(row_sum_sqrt), th.tensor(0.0, device=adj.device), row_sum_sqrt)
    #     deg = th.diag(deg)
    #     adj = deg@adj@deg
    #     return adj
    
    # def _get_r_adj(self, adj:th.Tensor):
    #     adj = adj
    #     for _ in range(self.r_pow-1):
    #         adj = adj@adj
    #     return adj
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.dim() > 2:
            observations = observations.reshape(observations.shape[0], -1)
        # expect observations.shape=[batch, node_num*node_features]
        bs = observations.shape[0]
        if self.node_wise:
            observations = observations.reshape(bs*self.node_num, -1)
        observations = self.mlp(observations)
        if self.node_wise:
            observations = observations.reshape(bs, -1)
        return observations