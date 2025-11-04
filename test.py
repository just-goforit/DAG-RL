# from pynauty import *
# import numpy as np
# # create a graph
# g = Graph(7, directed=True)
# edge_index = np.array([[0,1,2,3,4,5],
#                        [4,4,5,5,6,6]])
# for j in range(edge_index.shape[1]):
#     g.connect_vertex(edge_index[0,j], edge_index[1,j])
# label = canon_label(g)
# print(label)

from model.transformer import do_test
do_test()