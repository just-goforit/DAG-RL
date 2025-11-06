import torch
import torch_sparse
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from typing import List
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import LayerNorm
# from torch_geometric.nn import DeepGCNLayer, GATConv, GCNConv, SAGEConv, 

from typing import List, Optional, Tuple, Union
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing, GATConv, GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm

from spmm_ext import gspmm_src_mul_e_sum

class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
    projected via

    .. math::
        \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
        \mathbf{b})

    as described in Eq. (3) of the paper.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (str or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
            *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`.
            (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        project (bool, optional): If set to :obj:`True`, the layer will apply a
            linear transformation followed by an activation function before
            aggregation (as described in Eq. (3) of the paper).
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
        
    def forward(self, x: Tensor, edge_index: Adj,
                size: Size = None) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b = x.shape[0] # batch size
        n = x.shape[1] # node nums
        original_x = x
        project_x = x
         
        assert isinstance(x, Tensor)

        if self.project and hasattr(self, 'lin'):
            project_x = self.lin(x[0]).relu()
            # x = (self.lin(x[0]).relu(), x[1])
            
        con_project_x = project_x.transpose(0, 1).reshape(n, -1)
        con_x = x.transpose(0, 1).reshape(n, -1)
        
        x: OptPairTensor = (con_project_x, con_x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        
        out = out.reshape(n, b, -1).transpose(0, 1)
        
        out = self.lin_l(out)

        x_r = original_x
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')

class optSAGEConv(nn.Module):
    # use adjecent matrix (zeros filled)
    def __init__(self,
                 in_channels,
                 out_channels,
                 root_weight: bool = True,
                 project: bool = False,
                 normalize: bool = False,
                 bias: bool = True):
        super(optSAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.project = project
        self.normalize = normalize
        self.bias = bias
        
        self.w_aggr = Linear(in_channels, out_channels, bias=bias)
        self.w = Linear(in_channels, out_channels, bias=False)
    
        self.reset_parameters()
        
    def reset_parameters(self):
        self.w_aggr.reset_parameters()
        self.w.reset_parameters()
        
    def forward(self, x:torch.Tensor, edge_index:Union[torch.Tensor, Tuple]):
        # expect x.shape=[B, N, C]
        # expect edge_index.shape=[N, N]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b = x.shape[0]
        n = x.shape[1] # node num
        
        con_x = x.transpose(0, 1).reshape(n, -1)
        
        if isinstance(edge_index, tuple):
            # edge_index = (edge_vale, row_offset, col_indices, edge_indices)
            out = gspmm_src_mul_e_sum(n, con_x, edge_index[0], edge_index[1], edge_index[2], edge_index[3])
        else:
            # adj
            out = edge_index @ con_x # out.shape = [N, B*C]
            
        out = out.reshape(n, b, -1).transpose(0, 1)
        # out.shape = [B, N, C]
        out = self.w_aggr(out)
        root = self.w(x)
        out = out + root
        
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

class customGATConv(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 dropout=0.0,
                 bias=True,
                 negative_slope=0.2,
                 activation=None,
                 add_self_loops=False,
                 normalize=False):
        super(customGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.bias = bias
        self.negative_slope = negative_slope
        self.activation = activation
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.conv = GATConv(in_channels, out_channels // heads, heads, concat, dropout, bias, negative_slope)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv.reset_parameters()
        
    def forward(self, x:torch.Tensor, edge_index:Union[torch.Tensor, SparseTensor]):
        # expect x.shape=[B, N, C]
        # expect edge_index.shape=[N, N]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b = x.shape[0]
        n = x.shape[1] # node num
        
        con_x = x.transpose(0, 1).reshape(n, -1)
        out:torch.Tensor = self.conv(con_x, edge_index)       
        out = out.reshape(n, b, -1).transpose(0, 1)
        
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out    

class GNNExtractor(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 gnn_class: str='SAGE',
                 activation_fn: nn.Module=nn.ReLU,
                 dropout:float=0.0,
                 sp_tensor:bool = False):
        super(GNNExtractor, self).__init__()
        
        avaliable_gnn = ['SAGE', 'GAT']
        if gnn_class not in avaliable_gnn:
            raise ValueError(f'{gnn_class} is supported GNN class, available: {avaliable_gnn}')
        
        self.dropout = dropout
        self.out_features = out_features
        self.sp_tensor = sp_tensor
        
        # self.conv1 = SAGEConv(in_features, out_features)  # batched PyG
        if gnn_class == 'SAGE':
            self.conv1 = optSAGEConv(in_features, out_features) # batched spmm
        else:
            self.conv1 = customGATConv(in_features, out_features)
            
        act_fn = activation_fn()
        if isinstance(act_fn, torch.nn.modules.activation.ReLU):
            self.act_fn = F.relu
        elif isinstance(act_fn, torch.nn.modules.activation.Tanh):
            self.act_fn = F.tanh
        elif isinstance(act_fn, torch.nn.modules.activation.GELU):
            self.act_fn = F.gelu
        else:
            raise ValueError('Wrong activation function type')
        
    def forward(self, x:torch.Tensor, edge_index:Union[torch.Tensor, Tuple, SparseTensor]):
        x = self.conv1(x, edge_index)
        x = self.act_fn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        
class MlpLayer(nn.Module):
    """
    Used to extract features
    """
    def __init__(self, 
                 in_features, 
                 out_features, 
                 hidden_arch:list=[], 
                 activation_fn:nn.Module=nn.GELU,
                 dropout:float=0.,
                #  layerNorm:bool=False,
                #  batchNorm:bool=False
                 ):
        super(MlpLayer, self).__init__()
        _net: List[nn.Module] = []
        last_layer_dim_pi = in_features
        self.dropout = dropout
        for curr_layer_dim in hidden_arch:
            _net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            # if layerNorm:
            #     _net.append(LayerNorm(curr_layer_dim, mode='node')) # actually, it's graph mode
            # if batchNorm:
            #     _net.append(BatchNorm1d(curr_layer_dim)) # batchNorm layer has too many parameters
            _net.append(activation_fn())
            if dropout > 0:
                _net.append(nn.Dropout(dropout))
            last_layer_dim_pi = curr_layer_dim
        _net.append(nn.Linear(last_layer_dim_pi, out_features))
        # if layerNorm:
        #     _net.append(LayerNorm(out_features, mode='node')) # actually, it's graph mode
        # if batchNorm:
        #     _net.append(BatchNorm1d(out_features))
        if dropout > 0:
            _net.append(nn.Dropout(dropout))
        _net.append(activation_fn())
        self._net = nn.Sequential(*_net)
    
    def forward(self, x:torch.Tensor):
        x = self._net(x)
        return x

# class GATExtractor(nn.Module):
#     def __init__(self, 
#                  in_features, 
#                  out_features, 
#                  hidden_features=16, 
#                  block_num:int=0, 
#                  activation_fn: nn.Module=nn.ReLU,
#                  dropout:float=0.5, 
#                  layerNorm:bool=False,
#                  norm_mode:str='graph',
#                  device="cpu"):
#         super(GATExtractor, self).__init__()
#         self.layerNorm = layerNorm
#         self.dropout = dropout
#         self.out_features = out_features
#         self.conv1 = GATConv(in_features, hidden_features)
#         self.blocks = nn.ModuleList([DeepGCNLayer(GATConv(hidden_features, hidden_features),
#                                                   norm=LayerNorm(hidden_features, mode=norm_mode) if layerNorm else None, 
#                                                   act=activation_fn(), 
#                                                   block='res+',
#                                                   dropout=dropout) for _ in range(block_num)])
#         self.nm1 = LayerNorm(hidden_features, mode=norm_mode) if layerNorm else None
#         self.conv2 = GATConv(hidden_features, hidden_features)
#         self.nm2 = LayerNorm(out_features, mode=norm_mode) if layerNorm else None
#         self.device = device
#         act_fn = activation_fn()
#         if isinstance(act_fn, torch.nn.modules.activation.ReLU):
#             self.act_fn = F.relu
#         elif isinstance(act_fn, torch.nn.modules.activation.Tanh):
#             self.act_fn = F.tanh
#         elif isinstance(act_fn, torch.nn.modules.activation.GELU):
#             self.act_fn = F.gelu
#         else:
#             raise ValueError('Wrong activation function type')
            
#     def forward(self, x:torch.Tensor, edge_index):
#         dv = self.conv1.parameters().__next__().device
#         x = x.to(dv)
#         x = self.conv1(x, edge_index)
#         for block in self.blocks:
#             x = block(x, edge_index)
#         if self.nm1:
#             x = self.nm1(x)
#         x = self.act_fn(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         if self.nm2:
#             x = self.nm2(x)
#         x = self.act_fn(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         return x.unsqueeze(0)
 
# class GCNExtractor(nn.Module):
#     """DeepGCN extractor"""    
#     def __init__(self, 
#                  in_features, 
#                  out_features, 
#                  hidden_features=16, 
#                  block_num:int=0,
#                  activation_fn: nn.Module=nn.ReLU,
#                  dropout:float=0.5, 
#                  layerNorm:bool=False, 
#                  norm_mode:str='graph',
#                  device="cpu"):
#         super(GCNExtractor, self).__init__()
#         self.layerNorm = layerNorm
#         self.dropout = dropout
#         self.out_features = out_features
#         self.conv1 = GCNConv(in_features, hidden_features, cached=True)
#         self.blocks = nn.ModuleList([DeepGCNLayer(GCNConv(hidden_features, hidden_features, cached=True), 
#                                                   norm=LayerNorm(hidden_features, mode=norm_mode) if layerNorm else None, 
#                                                   act=activation_fn(), 
#                                                   block='res+',
#                                                   dropout=dropout) for _ in range(block_num)])
#         self.nm1 = LayerNorm(hidden_features, mode=norm_mode) if layerNorm else None
#         self.conv2 = GCNConv(hidden_features, out_features, cached=True)
#         self.nm2 = LayerNorm(out_features, mode=norm_mode) if layerNorm else None
#         self.device = device
#         act_fn = activation_fn()
#         if isinstance(act_fn, torch.nn.modules.activation.ReLU):
#             self.act_fn = F.relu
#         elif isinstance(act_fn, torch.nn.modules.activation.Tanh):
#             self.act_fn = F.tanh
#         elif isinstance(act_fn, torch.nn.modules.activation.GELU):
#             self.act_fn = F.gelu
#         else:
#             raise ValueError('Wrong activation function type')
        
#     def forward(self, x:torch.Tensor, edge_index):
#         dv = self.conv1.parameters().__next__().device
#         x = x.to(dv)
#         x = self.conv1(x, edge_index)
#         for block in self.blocks:
#             x = block(x, edge_index)
#         if self.nm1:
#             x = self.nm1(x)
#         x = self.act_fn(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         if self.nm2:
#             x = self.nm2(x)
#         x = self.act_fn(x) 
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         return x.unsqueeze(0)   

# class SAGEExtractor(nn.Module):
#     def __init__(self, 
#                  in_features, 
#                  out_features, 
#                  hidden_features=32, 
#                  block_num:int=0, 
#                  activation_fn: nn.Module=nn.ReLU,
#                  dropout:float=0.5, 
#                  layerNorm:bool=False, 
#                  norm_mode:str='graph',
#                  device="cpu"):
#         super(SAGEExtractor, self).__init__()
#         self.layerNorm = layerNorm
#         self.dropout = dropout
#         self.out_features = out_features
#         self.conv1 = SAGEConv(in_features, hidden_features)
#         self.blocks = nn.ModuleList([DeepGCNLayer(SAGEConv(hidden_features, hidden_features), 
#                                                   norm=LayerNorm(hidden_features, mode=norm_mode) if layerNorm else None, 
#                                                   act=activation_fn(),
#                                                   block='res+',
#                                                   dropout=dropout) for _ in range(block_num)])
#         self.nm1 = LayerNorm(hidden_features, mode=norm_mode) if layerNorm else None
#         # self.conv2 = SAGEConv(hidden_features, hidden_features)
#         # self.nm2 = LayerNorm(out_features, mode=norm_mode) if layerNorm else None
#         self.device = device
#         act_fn = activation_fn()
#         if isinstance(act_fn, torch.nn.modules.activation.ReLU):
#             self.act_fn = F.relu
#         elif isinstance(act_fn, torch.nn.modules.activation.Tanh):
#             self.act_fn = F.tanh
#         elif isinstance(act_fn, torch.nn.modules.activation.GELU):
#             self.act_fn = F.gelu
#         else:
#             raise ValueError('Wrong activation function type')
        
#     def forward(self, x:torch.Tensor, edge_index):
#         dv = self.conv1.parameters().__next__().device
#         x = x.to(dv)
#         x = self.conv1(x, edge_index)
#         for block in self.blocks:
#             x = block(x, edge_index)
#         if self.nm1:
#             x = self.nm1(x)
#         x = self.act_fn(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         # x = self.conv2(x, edge_index)
#         # if self.nm2:
#         #     x = self.nm2(x)
#         # x = self.act_fn(x) 
#         # x = F.dropout(x, p=self.dropout, training=self.training)
#         return x.unsqueeze(0)   