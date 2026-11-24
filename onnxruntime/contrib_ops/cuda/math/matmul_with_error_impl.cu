// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_with_error_impl.h"
#include "contrib_ops/cuda/math/with_error_device.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename CudaT>
__global__ void MatMulWithErrorKernel(
    const CudaT* A, const CudaT* B, const CudaT* Y, Dot2ErrorType<CudaT>* E, Dot2ErrorType<CudaT>* R,
    int64_t total, int64_t M, int64_t N, int64_t K,
    bool trans_a, bool trans_b, const int64_t* a_off, const int64_t* b_off) {
  using Acc = Dot2ErrorType<CudaT>;

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int64_t j = idx % N;
  const int64_t i = (idx / N) % M;
  const int64_t b = idx / (N * M);

  const CudaT* a_ptr = A + (a_off ? a_off[b] : 0);
  const CudaT* b_ptr = B + (b_off ? b_off[b] : 0);

  Dot2State<Acc> d;
  for (int64_t k = 0; k < K; ++k) {
    d.AddProduct(ToAcc(a_ptr[trans_a ? k * M + i : i * K + k]), ToAcc(b_ptr[trans_b ? j * K + k : k * N + j]));
  }
  E[idx] = FinishError<Acc>(d.s, d.c, ToAcc(Y[idx]));
  if (R != nullptr) R[idx] = Rn<Acc>::Add(d.s, d.c);
}

}  // namespace

template <typename CudaT>
void MatMulWithErrorImpl(
    cudaStream_t stream, const CudaT* A, const CudaT* B, const CudaT* Y, Dot2ErrorType<CudaT>* E,
    Dot2ErrorType<CudaT>* R, int64_t batch, int64_t M, int64_t N, int64_t K, bool trans_a, bool trans_b,
    const int64_t* a_off, const int64_t* b_off) {
  const int64_t total = batch * M * N;
  if (total == 0) return;
  constexpr int kThreads = 256;
  const int64_t blocks = (total + kThreads - 1) / kThreads;
  MatMulWithErrorKernel<CudaT><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      A, B, Y, E, R, total, M, N, K, trans_a, trans_b, a_off, b_off);
}

#define SPECIALIZED_IMPL(CudaT)                                                                              \
  template void MatMulWithErrorImpl<CudaT>(cudaStream_t, const CudaT*, const CudaT*, const CudaT*,           \
                                           Dot2ErrorType<CudaT>*, Dot2ErrorType<CudaT>*,       \
                                           int64_t, int64_t, int64_t, int64_t, bool, bool,   \
                                           const int64_t*, const int64_t*);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
