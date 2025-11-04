import sys
sys.path.append('..')
import numpy as np
from env.env_const import *

def _edges2np(origin_edges:list):
    """
    DAG edges to numpy array

    parameters
    ----------
    * origin_edges: list of tuple style edge

    >>> origin_edges=[(1,2),(3,2),(4,5)]

    return 
    ------
    >>> edges:[[1,3,4],
               [2,2,5]], type=nump.ndarray
    """
    return np.array(origin_edges, dtype=np.int64).T

class operator_info:
    def __init__(self, 
                 batch_size:int=1, in_height:int=16, in_width:int = 16 , inplanes:int = 3, 
                 kernel_size:int = 3, outplanes:int = 1,  
                 stride:int = 1, 
                 S:int=64,
                 env_id:str='dConv2d') -> None:
        self.batch_size = batch_size
        self.in_height = in_height
        self.in_width = in_width
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.kernel_size = kernel_size
        self.stride = stride
        self.S = S
        self.env_id = env_id
    
    def __str__(self) -> str:
        return "batch_size: %d, in_height: %d, in_width: %d, inplanes: %d, outplanes: %d, kernel_size: %d, stride: %d" % \
                (self.batch_size, self.in_height, self.in_width, self.inplanes, self.outplanes, self.kernel_size, self.stride)
    
    def __repr__(self) -> str:
        return self.__str__()

class DAG_Graph:
    """
    DAG -> Graph data
    """
    def __init__(self, g:dict=None, input_num:int=0, output_indeg:int=-1, output_num:list=None, copy=None) -> None:
        if copy is not None:
            self.x = copy.x.copy()
            self.edge_index = copy.edge_index.copy()
            self.node_num = copy.node_num
            self.input_num = copy.input_num
            self.output_num = copy.output_num
            self.mid_num = copy.mid_num
            self.output_indeg = copy.output_indeg
        else :
            assert output_indeg > 0 and output_indeg < 65536 and "output_indeg should be in range(0, 65536)"
            self.node_num = len(g['nodes'])
            self.input_num = input_num
            self.output_num = output_num
            self.mid_num = self.node_num - input_num - output_num
            self.x = np.zeros((self.node_num, max(IN_NODE_FEATURES, OUT_NODE_FEATURES, MID_NODE_FEATURES)), dtype=np.float32)
            self.edge_index = _edges2np(g['edges'])
            self.output_indeg = output_indeg
            # put blue pebble on input nodes
            self.x[:input_num, NODE_MEMED] = 1

    def _change_node_color_by_action(self, node_id, act):
        if act == LOAD or act == COMPUTE:   
            # memory -> cache 
            # cache, cache -> cache
            self.x[node_id, NODE_CACHED] = 1
        elif act == STORE: 
            assert self.x[node_id, NODE_CACHED] == 1 and "can only store cached node"
            # cache -> memory
            self.x[node_id, NODE_MEMED] = 1
            self.x[node_id, NODE_CACHED] = 0
        else: # DELETE
            assert self.x[node_id, NODE_CACHED] == 1 and "can only del cached node"
            self.x[node_id, NODE_CACHED] = 0

    # def _change_node_color(self, node_id, c):
    #     if c == WHITE:
    #         self.x[node_id, NODE_MEMED] = 0
    #         self.x[node_id, NODE_CACHED] = 0
    #     elif c == BLUE:
    #         self.x[node_id, NODE_MEMED] = 1
    #         self.x[node_id, NODE_CACHED] = 0
    #     elif c == RED:
    #         self.x[node_id, NODE_MEMED] = 1
    #         self.x[node_id, NODE_CACHED] = 1
    
    def _get_color(self, node_id):
        _color = WHITE
        # if self.x[node_id, NODE_MEMED] == 1:
        if self.x[node_id, NODE_CACHED] == 1:
            # node features = [0, 1, X] or [1, 1, X]
            _color = RED
        elif self.x[node_id, NODE_MEMED] == 1:
            _color = BLUE
        return _color

def get_edges_naive_plane2plane(nodes_start:int, 
                                kern_start:int,
                                out_start:int,
                                in_height:int, 
                                in_width:int,  
                                kernel_size:int, 
                                stride:int, 
                                new_nodes_start:int):
    """
    get edges between one plane and one kernel naively

    parameters
    ----------
    * nodes_start(int, required): start node index of the plane

    * kern_start(int, required): start node index of the kernel

    * out_start(int, required): output nodes index of the plane
    
    * in_height(int, required): height of the plane

    * in_width(int, required): width of the plane

    * stride(int, required): stride

    * new_nodes_start(int, required): start node index of the new plane

    return
    ------
    edges(list): edges between one plane and one kernels
    >>> [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 1)]
    """
    assert new_nodes_start > 0
    sample_plane = [[i*in_width + j + nodes_start for j in range(0, in_width)] for i in range(0, in_height)]
    kernel_plane = [[i*kernel_size + j + kern_start for j in range(0, kernel_size)] for i in range(0, kernel_size)]
    # print(sample_plane)
    product_edges = []
    product_nodes = []
    cnt = 0
    for i in range((in_height - kernel_size)//stride + 1):
        for j in range((in_width - kernel_size)//stride + 1):
            t_product_nodes=[]
            t_product_edges=[]
            # simulation for each kernel scan
            for k in range(kernel_size):
                for l in range(kernel_size):
                    t_product_nodes.append(new_nodes_start)
                    t_product_edges.append((sample_plane[i*stride+k][j*stride+l], new_nodes_start))
                    t_product_edges.append((kernel_plane[k][l],new_nodes_start))
                    new_nodes_start += 1
            product_nodes.extend(t_product_nodes)
            # just link t_product_nodes to output nodes
            assert len(t_product_nodes) == kernel_size * kernel_size
            for ni in t_product_nodes:
                t_product_edges.append((ni, out_start + cnt))
            cnt += 1
            product_edges.extend(t_product_edges)
            
    return product_nodes, product_edges, new_nodes_start

"""
generate directed conv2d operatr's DAG graph
"""
def naive_Conv2d_DAG(op_info:operator_info, format:str='nvis'):
    """
    naive conv2d algorithm DAG
    
    parameters
    ----------
    * batch_size(int, required[default=64]): batch size

    * in_height(int, required[default=16]): sample height
    
    * in_width(int, required[default=16]): sample width

    * inplanes(int, required[default=3]): input channels

    * kernel_size(int, required[default=3]): kernel size

    * outplanes(int, required[default=32]): output channels

    * stride(int, required[default=1]): stride

    returns
    -------
    g(dict), output_nodes(list), max_input_nodes_index(int)
    >>> g = {'nodes':[1, 2, 3, 4, 5],
    >>>      'edges':[(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 1)]} 
    """
    in_height = op_info.in_height
    in_width = op_info.in_width
    inplanes = op_info.inplanes
    outplanes = op_info.outplanes
    kernel_size = op_info.kernel_size # squared kernel only
    stride = op_info.stride
    batch_size = op_info.batch_size
    # 1. encode the samples and kernels
    """
       /--------------------/           /---/
      /                    / |         /   / |
     /                    /  |        /   /  |
    +-------------------+    |       +---+  /
    |1,2,...            |    |    X  |   | /  
    |                   |    |       +---+/
    |                   |   /       
    |                   |  /
    +-------------------+ /
    """
    """
    ver 1.0:

    o   o o   o o   o o   o  input
     \ /   \ /   \ /   \ /
      o     o     o     o    mid  ———> x
       \   /     /     /      |  
         o      /     /       |   ———> +
          \    /     /        |    |
            o       /        ———   |
             \     /               |
                o            out  ———
               
    ver 1.0:
    o   o o   o  o  o o   o  input
     \ /   \ /   \ /   \ /
      o     o     o     o    mid  ———> x
       \     \   /      /
        —————— o ——————      out  ———> +=
    """
    ## 1
    # get nodes number
    # 1) input
    samples_area = in_height * in_width         # a plane size of sample(img)
    kernels_area = kernel_size * kernel_size    # a plane size of kernel
    samples_vol = samples_area * inplanes       # volume size of a sample
    kernels_vol = kernels_area * inplanes       # volume size of a kernel
    samples_nodes_num = samples_vol * batch_size
    kernels_nodes_num = kernels_vol * outplanes
    inputs_num = samples_nodes_num + kernels_nodes_num
    # 2) output
    out_area = ((in_height - kernel_size)//stride + 1) * ((in_width - kernel_size)//stride + 1) # a plane size of output
    out_vol = out_area * outplanes              # volume size of a output
    outs_num = out_vol * batch_size
    # encode the nodes
    nodes = [i for i in range(inputs_num + outs_num)]
    ## 2
    edges = []
    new_nodes_start = inputs_num + outs_num
    for b in range(batch_size):
        # for each output channel
        for o in range(outplanes):
            # for kernel
            for p in range(inplanes):
                # for each input channel
                addnode, addedge, new_nodes_start = get_edges_naive_plane2plane(nodes_start=b*samples_vol + p*samples_area, 
                                                                                kern_start=o*kernels_vol + p*kernels_area + samples_nodes_num, 
                                                                                out_start=b*out_vol + o*out_area + inputs_num,
                                                                                in_height=in_height, 
                                                                                in_width=in_width, 
                                                                                kernel_size=kernel_size, 
                                                                                stride=stride,
                                                                                new_nodes_start=new_nodes_start)
                
                nodes.extend(addnode)
                edges.extend(addedge)
                
    if format == 'vis':
        G = {'nodes':nodes, 'edges':edges}
    else:
        G = DAG_Graph(g = {'nodes':nodes, 'edges':edges}, 
                      input_num=inputs_num,
                      output_indeg=kernels_vol,
                      output_num=outs_num)
    return G

# deprecated in v2.0
# def get_summary_tree(product_nodes:list=None, new_nodes_start:int=0):
#     """
#     get summary tree
#
#     parameters
#     ----------
#     * product_nodes(list, required): product nodes
#     * new_nodes_start(int, required): start node index
#
#     return
#     ------
#     nodes(list): nodes of the summary tree
#     edges(list): edges of the summary tree
#     new_start_node(int): new start node index
#
#     example
#     -------
#     >>> nodes = [1, 2, 3, 4, 5]
#     get_summary_tree(6, nodes)
#     >>> nodes = [6, 7, 8, 9]
#     >>> edges = [(1, 6), (2, 6), (6, 7), (3, 7), (7, 8), (4, 8), (8, 9), (5, 9)]
#     """
#     assert product_nodes is not None and len(product_nodes) > 1
#     summary_nodes = [i for i in range(new_nodes_start, new_nodes_start + len(product_nodes) - 1)]
#     summary_edges = []
#     summary_edges.append((product_nodes[0], summary_nodes[0]))
#     summary_edges.append((product_nodes[1], summary_nodes[0]))
#     for i in range(2, len(product_nodes)):
#         summary_edges.append((product_nodes[i], summary_nodes[i-1]))
#         summary_edges.append((summary_nodes[i-2], summary_nodes[i-1]))
#     return summary_nodes, summary_edges, summary_nodes[-1] + 1

# deprecated in v2.0
# def get_summary_tree_p2p(planes_output_nodes:list=None, new_nodes_start:int=0):
#     """
#     get summary tree between inplanes
#
#     parameters
#     ----------
#     * planes_output_nodes(list, required): output nodes for each plane(shape like:[[plane1's out],[plane2's out],...,[planek's out]])
#     * new_nodes_start(int, required): start node index
#
#     >>>     +------+
#     >>>   +------+z|
#     >>> +------+y|-+
#     >>> |     x|-+
#     >>> +------+
#
#     returns
#     -------
#     * a tuple: (addnodes, addedges, new_nodes_start, output_nodes)
#     * addnodes(list): nodes of the summary tree without planes_output_nodes
#     * addedges(list): edges of the summary tree
#     * new_nodes_start(int): new start node index
#     * output_nodes(list): output nodes of these planes
#
#     >>> +------+
#     >>> |     s|
#     >>> +------+
#     """
#     assert planes_output_nodes is not None and len(planes_output_nodes) > 1
#     summary_nodes = []
#     summary_edges = []
#     output_nodes = []
#     count = len(planes_output_nodes[0])
#     for i in range(count):
#         tmp_input = [plane[i] for plane in planes_output_nodes]
#         add_nodes, add_edges, new_nodes_start= get_summary_tree(tmp_input, new_nodes_start)
#         summary_nodes.extend(add_nodes)
#         summary_edges.extend(add_edges)
#         output_nodes.append(add_nodes[-1])
#
#     return summary_nodes, summary_edges, new_nodes_start, output_nodes