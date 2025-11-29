import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from model.transformer import TransformerPtrNet
from stable_baselines3.common.utils import get_device
from typing import Type, List, Dict, Union, Tuple, Optional


def ceil2pow(x):
    return pow(2, math.ceil(math.log(x) / math.log(2) + 1))


def floor2pow(x):
    return pow(2, math.ceil(math.log(x) / math.log(2)))


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        pi_feature_dim: int,
        vf_feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = pi_feature_dim
        last_layer_dim_vf = vf_feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class CustomPolicyValueNet(MlpExtractor):
    """
    this is the policy & value net head

    attention
    ---------
    the output features equals to the probability of every action,
    so:
        nodes * actions <==> nodes + actions which output style is better
    """

    def __init__(
        self,
        feature_dim: int,  # get from super->feature_extractor->gnn->feature_dim
        addi_features: int,
        node_wise: bool,  # whether feature_dim is per node
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        node_num: int,
        edge_index: np.ndarray,
        pi_conv_out: int = 8,
        vf_conv_out: int = 2,
        transformer: bool = False,
        # topk_layer:int = 1,
        # focus_num: int = 0, # focus on K nodes through pooling
        gp_vf: bool = False,  # add mean global pooling to vf
        # focus_vf: bool = False, # focus on K nodes through pooling
        # layerNorm: bool = False,
        # norm_mode: str = 'graph',
        device: Union[torch.device, str] = "auto",
    ) -> None:
        self.gp_vf = gp_vf
        self.transformer = transformer
        # pi_dense = (focus_num if focus_num != 0 else node_num) * (pi_conv_out if pi_conv_out != 0 else feature_dim) # action_net(Linear) input
        # vf_dense = ((focus_num+(1 if gp_vf else 0)) if focus_num != 0 and focus_vf else (1 if gp_vf else node_num)) * (vf_conv_out if vf_conv_out != 0 else feature_dim)
        pi_dense = (
            0
            if transformer
            else node_num * (pi_conv_out if pi_conv_out != 0 else feature_dim)
        )
        vf_dense = (1 if gp_vf else node_num) * (
            vf_conv_out if vf_conv_out != 0 else feature_dim
        )
        # action_net(Linear) input
        self.addi_features = addi_features
        if not node_wise:
            # assert focus_num == 0 and 'if not node_wise feature(mlp feature_extractor), focus_num must be 0'
            # assert focus_vf == False and 'if not node_wise feature(mlp feature_extractor), focus_vf must be False'
            assert (
                pi_conv_out == 0
                and "if not node_wise feature(mlp feature_extractor), pi_conv_out must be 0"
            )
            assert (
                vf_conv_out == 0
                and "if not node_wise feature(mlp feature_extractor), vf_conv_out must be 0"
            )
            # assert layerNorm == False and 'if not node_wise feature(mlp feature_extractor), layerNorm must be False'
            assert transformer == False
            pi_dense = vf_dense = feature_dim
        super().__init__(
            pi_dense + addi_features,
            vf_dense + addi_features,
            net_arch,
            activation_fn,
            device,
        )  # Pass remaining arguments to base class
        """
                                                                                                     _dense
                                                                                                        |
                                                                                                        v
                                              +--->conv_pi -> (nm_pi->) relu -> (fc_pi->relu->) -> action_net: [node_num, action_space]
        node_num, feature_dim -> focus_conv ->|
                                              +--->conv_vf -> relu -> (fc_vf->relu->) -> value_net: [1,]
        """
        # act_fn = activation_fn()
        # if isinstance(act_fn, torch.nn.modules.activation.ReLU):
        #     self.act_fn = F.relu
        # elif isinstance(act_fn, torch.nn.modules.activation.Tanh):
        #     self.act_fn = F.tanh
        # elif isinstance(act_fn, torch.nn.modules.activation.GELU):
        #     self.act_fn = F.gelu
        # else:
        #     raise ValueError('Wrong activation function type')

        self.node_num = node_num
        # self.focus_num = focus_num
        # self.focus_vf = focus_vf
        # self.spotlight:List[nn.Module] = [] if focus_num != 0 else None
        # if focus_num != 0:
        #     # more complex spotlight
        #     for _ in range(topk_layer - 1):
        #         self.spotlight.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=1))
        #         self.spotlight.append(nn.LeakyReLU())
        #     self.spotlight.append(nn.Conv2d(feature_dim, 1, kernel_size=1))
        #     # self.spotlight.append(nn.LeakyReLU())
        #     self.spotlight = nn.Sequential(*self.spotlight)

        # conv_layer = 1
        conv_pi: List[nn.Module] = []
        conv_vf: List[nn.Module] = []
        # for _ in range(conv_layer-1):
        #     conv_pi.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=1))
        #     conv_pi.append(activation_fn())
        #     conv_vf.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=1))
        #     conv_vf.append(activation_fn())
        if pi_conv_out > 0:
            conv_pi.append(nn.Conv2d(feature_dim, pi_conv_out, kernel_size=1))
            conv_pi.append(activation_fn())
        if vf_conv_out > 0:
            conv_vf.append(nn.Conv2d(feature_dim, vf_conv_out, kernel_size=1))
            conv_vf.append(activation_fn())

        self.conv_pi = nn.Sequential(*conv_pi) if pi_conv_out != 0 else None
        self.conv_vf = nn.Sequential(*conv_vf) if vf_conv_out != 0 else None
        # if norm_mode == 'graph':
        #     self.nm_pi = LayerNorm(pi_dense, mode='node') if pi_conv_out > 1 and layerNorm else None
        #     # self.nm_vf = LayerNorm(vf_dense, mode='node') if vf_conv_out > 1 and layerNorm else None
        # else:
        #     self.nm_pi = LayerNorm(pi_conv_out, mode='node') if pi_conv_out > 1 and layerNorm else None
        #     # self.nm_vf = LayerNorm(vf_conv_out, mode='node') if vf_conv_out > 1 and layerNorm else None
        # self.norm_mode = norm_mode

        # self.focus_idx = None
        # self.focus_score = None

        appraiser: List = []
        if transformer:
            # appraiser.append(nn.Conv2d(feature_dim, ceil2pow(feature_dim), kernel_size=1))
            # appraiser.append(activation_fn())
            # appraiser.append(nn.Conv2d(ceil2pow(feature_dim), feature_dim, kernel_size=1))
            # appraiser.append(activation_fn())
            self.ptrnet = TransformerPtrNet(
                node_num,
                edge_index,
                n_layers=2,
                d_model=feature_dim,
                ffn_hidden=feature_dim * 2,
                n_head=4,
                drop_prob=0.1,
                layer_norm=True,
                device=device,
            )

            appraiser.append(nn.Linear(feature_dim, 1))
            appraiser.append(nn.Tanh())
            self.appraiser = nn.Sequential(*appraiser)
        else:
            self.ptrnet = None

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.policy_net) > 0:
            features = self.policy_net(features)
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.value_net) > 0:
            features = self.value_net(features)
        return features

    def _forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        ## actor pi
        if self.conv_pi:
            if x.dim() > 2:
                x_pi = x.transpose(1, 2)
            else:
                x_pi = x.reshape(x.shape[0], -1, self.node_num, 1)
            x_pi: torch.Tensor = self.conv_pi(x_pi)
            x_pi = x_pi.transpose(1, 2).reshape(x_pi.shape[0], -1)
            # if self.nm_pi is not None:
            #     # x_pi.shape = [batch_size, pi_dense]
            #     if self.norm_mode == 'node':
            #         x_pi = x_pi.reshape(x_pi.shape[0], -1, self.pi_conv_out)
            #     x_pi = self.nm_pi(x_pi)
            #     x_pi = x_pi.reshape(x_pi.shape[0], -1)
            # x_pi = self.act_fn(x_pi)
        else:
            x_pi = x
        return x_pi

    def global_pooling(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape=[batch_size, features, node_num, ...
        return torch.mean(x, dim=2, keepdim=True)

    def _forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        ## value function
        if self.conv_vf:
            if x.dim() > 2:
                # x_vf = x.transpose(1, 2)
                x_vf = x.unsqueeze(-1)
            else:
                x_vf = x.reshape(x.shape[0], -1, self.node_num, 1)
            x_vf: torch.Tensor = self.conv_vf(x_vf)
            # x_vf.shape=[batch_size, vf_conv_out, node_num, 1]
            x_vf = x_vf.transpose(1, 2).reshape(x_vf.shape[0], -1)
            # if self.nm_vf is not None:
            #     # x_pi.shape = [batch_size, vf_dense]
            #     if self.norm_mode == 'node':
            #         x_vf = x_vf.reshape(x_vf.shape[0], -1, self.vf_conv_out)
            #     x_vf = self.nm_vf(x_vf)
            #     x_vf = x_vf.reshape(x_vf.shape[0], -1)
            # x_vf = self.act_fn(x_vf)
        else:
            x_vf = x.reshape(x.shape[0], -1)
        return x_vf

    def forward(self, x: torch.Tensor):
        # expect x.shape = [batch_size,(features, node_num)]
        focus_v = x[:, : -self.addi_features] if self.addi_features > 0 else x
        focus = x[:, : -self.addi_features] if self.addi_features > 0 else x
        if self.gp_vf:
            # gf = self.global_pooling(focus_v.reshape(x.shape[0], -1, self.node_num))
            focus_v = self.global_pooling(
                focus_v.reshape(x.shape[0], -1, self.node_num)
            )
        elif self.conv_vf is not None:
            focus_v = focus_v.reshape(focus_v.shape[0], -1, self.node_num)
        if self.ptrnet is not None:
            # transformer
            focus = focus.reshape(focus.shape[0], -1, self.node_num).transpose(1, 2)
            if self.ptrnet.emb.pos_emb.encoding.device != focus.device:
                self.ptrnet.emb.pos_emb.update_device(focus.device)
            focus = self.ptrnet(focus)
            focus = self.appraiser(focus)

        # if not vf_only and self.spotlight:
        #     # for focus_vf is not reasonable and deprecated, so if vf_only, we should not cost to focus
        #     # CustomMaskableActorCriticPolicy->predict_values()
        #     g = x[:, : -self.addi_features] if self.addi_features > 0 else x
        #     g = g.reshape(x.shape[0], -1, self.node_num).unsqueeze(-1)
        #     scores:torch.Tensor= self.spotlight(g)
        #     # expect scores.shape = [batch_size, 1, node_num, 1]
        #     if mask is not None and self.focus_num == 1:
        #         # if we have mask, we should choose from mask, more, focus_num must eqs 1
        #         mask = torch.from_numpy(mask).reshape(mask.shape[0], self.node_num, -1)
        #         invalid_mask = (torch.sum(mask, axis=-1) < 1.0).reshape(mask.shape[0], 1, self.node_num, 1)
        #         scores[invalid_mask] = torch.min(scores) - 1.0
        #     self.focus_score, self.focus_idx = torch.topk(scores, self.focus_num, dim=2)
        #     _idx = torch.cat([self.focus_idx] * g.shape[1], dim = 1)
        #     focus = torch.gather(g, dim=2, index=_idx)
        #     # focus.shape = [batch_size, features, focus_num, 1]
        #     focus = focus.transpose(1,2) # avoid no conv raise dim confuse, actually may be useless
        #     # expect focus.shape = [batch_size, focus_num, features, 1]
        # if self.focus_vf:
        #     focus_v = focus
        #     if self.gp_vf:
        #         focus_v = torch.cat((focus_v, gf.transpose(1,2).unsqueeze(-1)), dim=1)
        # elif self.gp_vf:
        #     # only use global pooling for vf
        #     focus_v = gf

        focus = self._forward_actor(focus)  # conv
        focus_v = self._forward_critic(focus_v)  # conv

        if self.addi_features > 0:
            # focus = torch.cat((focus.reshape(x.shape[0], -1), x[:, -self.addi_features].unsqueeze(-1)), dim=1)
            focus_v = torch.cat(
                (
                    focus_v.reshape(x.shape[0], -1),
                    x[:, -self.addi_features].unsqueeze(-1),
                ),
                dim=1,
            )

        return self.forward_actor(focus), self.forward_critic(focus_v)
