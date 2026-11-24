// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/reduce_sum_with_error_impl.h"
#include "contrib_ops/cuda/math/with_error_device.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename CudaT>
__global__ void ReduceSumWithErrorKernel(const CudaT* X, const CudaT* Y, Dot2ErrorType<CudaT>* E,
                                         int64_t out_count, ReduceSumWithErrorLayout lay) {
  using Acc = Dot2ErrorType<CudaT>;

  int64_t o = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (o >= out_count) return;

  // Base offset of this output's reduction group in the input.
  int64_t base = 0;
  int64_t rem = o;
  for (int a = lay.n_kept - 1; a >= 0; --a) {
    base += (rem % lay.kept_dims[a]) * lay.kept_strides[a];
    rem /= lay.kept_dims[a];
  }

  int64_t red_count = 1;
  for (int a = 0; a < lay.n_reduced; ++a) red_count *= lay.reduced_dims[a];

  Dot2State<Acc> d;
  int64_t counter[kReduceWithErrorMaxRank] = {};
  int64_t offset = base;
  for (int64_t r = 0; r < red_count; ++r) {
    d.Add(ToAcc(X[offset]), Acc(0));
    // advance the odometer over the reduced axes, last axis fastest (row-major order)
    for (int a = lay.n_reduced - 1; a >= 0; --a) {
      offset += lay.reduced_strides[a];
      if (++counter[a] < lay.reduced_dims[a]) break;
      offset -= lay.reduced_dims[a] * lay.reduced_strides[a];
      counter[a] = 0;
    }
  }
  E[o] = FinishError<Acc>(d.s, d.c, ToAcc(Y[o]));
}

}  // namespace

template <typename CudaT>
void ReduceSumWithErrorImpl(cudaStream_t stream, const CudaT* X, const CudaT* Y, Dot2ErrorType<CudaT>* E,
                            int64_t out_count, const ReduceSumWithErrorLayout& layout) {
  if (out_count == 0) return;
  constexpr int kThreads = 256;
  const int64_t blocks = (out_count + kThreads - 1) / kThreads;
  ReduceSumWithErrorKernel<CudaT><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      X, Y, E, out_count, layout);
}

#define SPECIALIZED_IMPL(CudaT)                                                                         \
  template void ReduceSumWithErrorImpl<CudaT>(cudaStream_t, const CudaT*, const CudaT*,                 \
                                              Dot2ErrorType<CudaT>*, int64_t, const ReduceSumWithErrorLayout&);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
