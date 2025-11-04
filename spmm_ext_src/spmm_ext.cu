/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cu
 * \brief SPMM C APIs and definitions.
 */
// #include "cuda_common.h"
// #include "spmm.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

namespace dgl {

// using namespace cuda;

namespace aten {

// edata is (E, 1)
__global__ void u_mul_e_sum_kernel(const int num_nodes, const int feat_dim,
                                   const float *ufeature, const float *efeature,
                                   const int *node_index, const int *edge_index,
                                   const int *node_pointer,
                                   float *__restrict__ next_layer) {
  int node_id = blockDim.y * blockIdx.x + threadIdx.y;
  int feat_offset = blockIdx.y * blockDim.x + threadIdx.x;
  if (node_id >= num_nodes)
    return;
  if (feat_offset >= feat_dim)
    return;
  int offset = node_pointer[node_id];
  int end = node_pointer[node_id + 1];
  float local = 0.0f;
  int target;
  for (int i = offset; i < end; i++) {
    const int eid = __ldg(edge_index + i);
    const int cid = __ldg(node_index + i);
    target = cid * feat_dim + feat_offset;
    local += ufeature[target] * efeature[eid];
  }
  next_layer[node_id * feat_dim + feat_offset] = local;
}

/*
  Every warp process a tile of tile_r (# of nodes) * tile_c (# of features)
  blockDim.x = 32
  tile_r = blockDim.y
  tile_c * gridDim.y >= feat_dim
  tile_c = 32 * factor
*/
template <int tile_r, int tile_c, int factor>
__global__ void u_mul_e_sum_kernel_neat(
    const int num_nodes, const int feat_dim, const float *ufeature,
    const float *efeature, const int *node_index, const int *edge_index,
    const int *node_pointer, float *__restrict__ next_layer) {
  int node_id = tile_r * blockIdx.x + threadIdx.y;
  if (node_id >= num_nodes)
    return;
  int offset = node_pointer[node_id];
  int degree = node_pointer[node_id + 1] - offset;
  int sm_offset = threadIdx.y << 5;
  __shared__ int neighbor_local[tile_r << 5];
  __shared__ float factor_local[tile_r << 5];
  float local[factor];
#pragma unroll
  for (int i = 0; i < factor; i++) {
    local[i] = 0;
  }

  // Tree reduction might be useful for certain graphs
  int feat_id = blockIdx.y * tile_c + threadIdx.x;
  for (int i = 0; i < degree / 32; i++) {
    neighbor_local[sm_offset + threadIdx.x] =
        node_index[offset + i * 32 + threadIdx.x] * feat_dim;
    factor_local[sm_offset + threadIdx.x] =
        efeature[edge_index[offset + i * 32 + threadIdx.x]];
#pragma unroll
    for (int j = 0; j < 32; j++) {
      // const int eid = __ldg(edge_index + offset + i * 32 + j);
      // float local_factor = efeature[eid];
      int local_target = neighbor_local[sm_offset + j] + feat_id;
      float local_factor = factor_local[sm_offset + j];
#pragma unroll
      for (int k = 0; k < factor; k++) {
        local[k] += ufeature[local_target + (k << 5)] * local_factor;
      }
    }
  }

  if (threadIdx.x < degree % 32) {
    neighbor_local[sm_offset + threadIdx.x] =
        node_index[offset + degree - (degree % 32) + threadIdx.x] * feat_dim;
    factor_local[sm_offset + threadIdx.x] =
        efeature[edge_index[offset + degree - (degree % 32) + threadIdx.x]];
  }
  __syncwarp();
  for (int i = 0; i < degree % 32; i++) {
    float local_factor = factor_local[sm_offset + i];
    // const int eid = __ldg(edge_index + offset + degree - (degree % 32) + i);
    // float local_factor = efeature[eid];
    int local_target = neighbor_local[sm_offset + i] + feat_id;
#pragma unroll
    for (int k = 0; k < factor; k++) {
      local[k] += ufeature[local_target + (k << 5)] * local_factor;
    }
  }
#pragma unroll
  for (int i = 0; i < factor; i++) {
    next_layer[node_id * feat_dim + feat_id + i * 32] = local[i];
  }
}

} // namespace aten
} // namespace dgl

// Todo: dedup and simplify the interfaces
torch::Tensor src_mul_e_sum(int64_t num_nodes,
                            torch::Tensor ufeature, 
                            torch::Tensor efeature,
                            // std::vector<int64_t> dims, bool keep_dim,
                            // at::optional<at::ScalarType> dtype,
                            torch::Tensor in_pointer,
                            torch::Tensor in_node_indices,
                            torch::Tensor in_edge_indices) {
  int feat_dim = ufeature.dim() == 1 ? 1 : ufeature.size(1);
  auto out = torch::zeros({num_nodes, feat_dim},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
  if (feat_dim % 64 == 0) {
    dim3 blocks(num_nodes, feat_dim / 64, 1);
    dim3 threads(32, 1, 1);
    dgl::aten::u_mul_e_sum_kernel_neat<1, 64, 2><<<blocks, threads>>>(
        num_nodes, feat_dim, ufeature.data_ptr<float>(),
        efeature.data_ptr<float>(), in_node_indices.data_ptr<int>(),
        in_edge_indices.data_ptr<int>(),
        in_pointer.data_ptr<int>(), out.data_ptr<float>());
  } else if (feat_dim % 32 == 0) {
    dim3 blocks((num_nodes + 3) / 4, feat_dim / 32, 1);
    dim3 threads(32, 4, 1);
    dgl::aten::u_mul_e_sum_kernel_neat<4, 32, 1><<<blocks, threads>>>(
        num_nodes, feat_dim, ufeature.data_ptr<float>(),
        efeature.data_ptr<float>(), in_node_indices.data_ptr<int>(),
        in_edge_indices.data_ptr<int>(),
        in_pointer.data_ptr<int>(), out.data_ptr<float>());
  } else {
    int tx = feat_dim < 32 ? feat_dim : 32;
    dim3 blocks((num_nodes + 15) / 16, (feat_dim + 31) / 32, 1);
    dim3 threads(tx, 16, 1);
    dgl::aten::u_mul_e_sum_kernel<<<blocks, threads>>>(
        num_nodes, feat_dim, ufeature.data_ptr<float>(),
        efeature.data_ptr<float>(), in_node_indices.data_ptr<int>(),
        in_edge_indices.data_ptr<int>(),
        in_pointer.data_ptr<int>(), out.data_ptr<float>());
  }
  return out;
}

// static auto registry =
//     torch::RegisterOperators()
//         .op("my_ops::gspmm_src_mul_e_sum(int num_node, Tensor x, Tensor y, "
//             "Tensor row_off, Tensor col_ind, Tensor edge_ind) -> Tensor z",
//             &src_mul_e_sum);

PYBIND11_MODULE(spmm_ext, m) {
    m.def("gspmm_src_mul_e_sum",
          &src_mul_e_sum,
          "spmm warpper");
}