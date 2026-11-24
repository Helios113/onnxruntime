// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cuda/math/with_error_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kReduceWithErrorMaxRank = 8;

// Layout of a reduction over a row-major input: the kept axes (in order) index the output, the reduced axes (in
// order, last axis fastest) are summed. Strides are input strides in elements.
struct ReduceSumWithErrorLayout {
  int n_kept = 0;
  int n_reduced = 0;
  int64_t kept_dims[kReduceWithErrorMaxRank] = {};
  int64_t kept_strides[kReduceWithErrorMaxRank] = {};
  int64_t reduced_dims[kReduceWithErrorMaxRank] = {};
  int64_t reduced_strides[kReduceWithErrorMaxRank] = {};
};

// E[o] = (s - Y[o]) + c with (s, c) the compensated (TwoSum) sum of the reduced elements of the input in row-major
// order of the reduced axes; E is NaN if Y[o] is not finite. out_count is the number of outputs.
template <typename CudaT>
void ReduceSumWithErrorImpl(
    cudaStream_t stream,
    const CudaT* X,
    const CudaT* Y,
    Dot2ErrorType<CudaT>* E,
    int64_t out_count,
    const ReduceSumWithErrorLayout& layout);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
