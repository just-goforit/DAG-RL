import math 
import numpy as np
import torch
from torch import nn
from pynauty import *
from typing import List

class PositionalEncoding():
    """
    compute sinusoid encoding.
    code from https://github.com/gusdnd852, Hyunwoong, 2019
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

class GraphPositionalEncoding(PositionalEncoding):
    """
    get node positionalEncoding of graph nodes
    """
    def __init__(self, d_model, node_num, edge_index, device):
        super().__init__(d_model, node_num, device)
        
        if edge_index is not None:
            self.canon_labels = self.get_canon_label(node_num, edge_index)
            lable_permutation = torch.tensor(self.canon_labels, dtype=torch.int64)
            self.encoding = torch.index_select(self.encoding, 0, lable_permutation)
            
    def get_canon_label(self, node_num, edge_index: np.ndarray):
        g = Graph(node_num, directed=True)
        for j in range(edge_index.shape[1]):
            g.connect_vertex(edge_index[0,j], edge_index[1,j])
        label = canon_label(g)
        return label
    
    def update_device(self, device):
        self.encoding = self.encoding.to(device)

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None, e=-1e8):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score:torch.Tensor = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        # self.w_q = nn.Linear(d_model, d_model)
        # self.w_k = nn.Linear(d_model, d_model)
        # self.w_v = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, d_model * 3)
        
        self.w_concat = nn.Linear(d_model, d_model)

    # def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask=None):
    def forward(self, x:torch.Tensor, mask=None):
        # 1. dot product with weight matrices
        # q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 1.5 combine to one mm
        x = self.w(x)
        q, k, v = torch.split(x, x.size(-1) // 3, dim=-1)
        
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention_score = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor:torch.Tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor:torch.Tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class LayerNorm(nn.Module):
    def __init__(self, d_model:int, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x:torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model:int, hidden:int, drop_prob:float):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob) if drop_prob > 0 else None

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, ffn_hidden:int, n_head:int, drop_prob:float, layer_norm:bool):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.dropout1 = nn.Dropout(p=drop_prob) if drop_prob > 0 else None
        self.norm1 = LayerNorm(d_model=d_model) if layer_norm else None

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model) if layer_norm else None
        # self.dropout2 = nn.Dropout(p=drop_prob) # has Dropout layer in ffn's last operator

    def forward(self, x:torch.Tensor, src_mask:torch.Tensor):
        # 1. compute self attention
        _x = x
        # x = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.attention(x=x, mask=src_mask)
        
        # 2. add and norm
        if self.dropout1 is not None:
            x = self.dropout1(x)
        x = x + _x
        if self.norm1 is not None:
            x = self.norm1(x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        # x = self.dropout2(x)
        x = x + _x
        if self.norm2 is not None:
            x = self.norm2(x)
        return x

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, d_model:int, max_len:int, drop_prob:float, edge_index:np.ndarray, device:str):
        """
        class for word embedding that included positional information

        :param d_model: dimensions of model
        :param drop_prob: global dtopout prob
        :param edge_index: pass non-None value to enable graph cano
        """
        super(TransformerEmbedding, self).__init__()
        self.pos_emb = GraphPositionalEncoding(d_model, max_len, edge_index, device)
        # self.drop_out = nn.Dropout(p=drop_prob)
        self.scale = math.sqrt(d_model) 

    def forward(self, x:torch.Tensor):
        x = x * self.scale + self.pos_emb.encoding
        return x

class TransformerPtrNet(nn.Module):
    def __init__(self, 
                node_num:int, edge_index:np.ndarray=None, 
                n_layers:int=1, d_model:int=16, ffn_hidden:int=32, n_head:int=2, drop_prob:float=0.1, layer_norm:bool=False, 
                device:str='cpu') -> None:
        super(TransformerPtrNet, self).__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=node_num,
                                        edge_index=edge_index,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  layer_norm=layer_norm)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask=None):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask=None)

        return x

def do_test():
    net = TransformerPtrNet(node_num = 10, n_layers=2, drop_prob=0.1, layer_norm=True)
    x = torch.randn((2, 10,  16), dtype=torch.float32)
    y = net(x)