import torch
import numpy as np
from torch_sparse import SparseTensor
from env.graph import operator_info, naive_Conv2d_DAG

import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from typing import List
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm

from spmm_ext import gspmm_src_mul_e_sum

class SAGEConv(MessagePassing):
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
            # self.lin.reset_parameters()
            self.lin.weight = nn.Parameter(torch.ones_like(self.lin.weight))
        # self.lin_l.reset_parameters()
        self.lin_l.weight = nn.Parameter(torch.ones_like(self.lin_l.weight))
        if self.root_weight:
            self.lin_r.reset_parameters()
            self.lin_r.weight = nn.Parameter(torch.ones_like(self.lin_r.weight))
        
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
        
        self.w_aggr.weight = nn.Parameter(torch.ones_like(self.w_aggr.weight))
        self.w.weight = nn.Parameter(torch.ones_like(self.w.weight))
            
    def forward(self, x:torch.Tensor, edge_index:Union[torch.Tensor, Tuple]):
        # expect x.shape=[B, N, C]
        # expect edge_index.shape=[N, N]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b = x.shape[0] # batch size
        n = x.shape[1] # node num
        
        con_x = x.transpose(0, 1).reshape(n, -1)
        
        if isinstance(edge_index, tuple):
            # edge_index = (edge_vale, row_offset, col_indices, edge_indices)
            out = gspmm_src_mul_e_sum(n, con_x, edge_index[0], edge_index[1], edge_index[2], edge_index[3])
        else:
            # adj
            out = edge_index @ con_x # out.shape = [N, B*C]
            
        out = out.reshape(n, b, -1).transpose(0, 1) # out.shape = [B, N, C]
        out = self.w_aggr(out)
        root = self.w(x)
        out = out + root
        
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out
### 

def update_edge_index2adj(node_num, edge_index, device='cpu'):
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index).to(device)
    adj = torch.zeros((node_num, node_num), dtype=torch.float32, device=device, requires_grad=False)
    adj[edge_index[1], edge_index[0]] = 1.0 # predecessor -> self
    sum_ = adj.sum(dim=1, keepdim=True)
    sum_[sum_ < 1.0] = 1.0
    adj = adj / sum_
    return adj

def edge_index_sparse(node_num:int, edge_index:torch.Tensor, device:str='cpu'):
    edge_index_ = edge_index.to(torch.int64).to(device)
    _sp = SparseTensor(row=edge_index_[1], col=edge_index_[0], sparse_sizes=(node_num, node_num))
    # _sp.csr()
    return _sp

def get_e_v(edge_index:torch.Tensor, device:str='cpu'):
    sorted_, _ = torch.sort(edge_index[1])
    _, counts = torch.unique(sorted_, return_counts=True)
    e_value = torch.cat([torch.full(size=(count,), 
                                    fill_value = 1.0/value, 
                                    dtype=torch.float32) for value, count in zip(counts, counts)], dim=0).to(device)
    return e_value

# op_info = operator_info(batch_size=1, 
#                         in_height=1, 
#                         in_width=1, 
#                         inplanes=1,
#                         kernel_size=1, 
#                         outplanes=1, 
#                         S=64)

def do_test(op_info=None, batch=1):
    if op_info is None:
        op_info = operator_info(batch_size=3, 
                                in_height=12, 
                                in_width=12, 
                                inplanes=3,
                                kernel_size=3, 
                                outplanes=3, 
                                S=64)
    graph = naive_Conv2d_DAG(op_info)

    x = torch.randn((batch, graph.node_num, 3), dtype=torch.float32, device='cuda:0')
    print(graph.node_num, graph.edge_index.shape[1])
    # edge_index = torch.tensor([[0, 0, 1],
    #                            [1, 2, 2]], dtype=torch.int64, device='cuda:0')
    edge_index = torch.from_numpy(graph.edge_index).to(dtype=torch.int64, device='cuda:0')
    adj = update_edge_index2adj(graph.node_num, edge_index, 'cuda:0')
    sp_edge_index = edge_index_sparse(graph.node_num, edge_index, 'cuda:0')
    row_off, col_ind, _ = sp_edge_index.csr()

    edge_index_csr = (get_e_v(edge_index, 'cuda:0'),
                      row_off.to(dtype=torch.int32), 
                      col_ind.to(dtype=torch.int32), 
                      torch.arange(edge_index.shape[1], dtype=torch.int32, device='cuda:0'))

    conv1 = SAGEConv(in_channels=3, out_channels=8, bias=False).to('cuda:0')
    conv2 = optSAGEConv(in_channels=3, out_channels=8, bias=False).to('cuda:0')

    with torch.autograd.profiler.profile() as prof:
        for _ in range(2000):
            # y1 = conv1(x, edge_index)
            # y1 = conv1(x, sp_edge_index)
            y2 = conv2(x, adj)
            # y2 = conv2(x, edge_index_csr)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    ## wrong way !!!
    # y1 = torch.tensor(()).to('cuda:0')
    # for i in range(x.shape[0]):
    #     # x[i].shape=[node_nums, node_features]
    #     y1 = torch.cat((y1, conv1(x[i], edge_index)), dim=0)
    ## wrong way !!!

    # y1 = conv1(x, edge_index)
    y1 = conv1(x, sp_edge_index)
    y2 = conv2(x, adj)
    # y2 = conv2(x, edge_index_csr)
    # result check
    print('Result check: ' + ('T' if torch.allclose(y1, y2, atol=1e-5) == True else 'F'))
    ### 
    
# if __name__ == '__main__':
#     from spmm_ext_src.sageConvTest import do_test
#     do_test(batch = 256)

