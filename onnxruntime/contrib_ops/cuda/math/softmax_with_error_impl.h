// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cuda/math/with_error_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Softmax of X along an axis viewed as [outer, n, inner], and E, the error propagated through the chain
// x - max, exp, sum, division (see SoftmaxWithError in contrib_ops/cpu/softmax_with_error.cc). E is NaN where Y is not
// finite.
// B bounds the rest, mainly the rounding of exp, for which a relative error rho is assumed (2 ulp for float, 1 ulp for
// double: the documented CUDA maximum): B = 2 rho y + 4 rho |E|, NaN where E is.
template <typename CudaT>
void SoftmaxWithErrorImpl(
    cudaStream_t stream,
    const CudaT* X,
    CudaT* Y,
    Dot2ErrorType<CudaT>* E,
    Dot2ErrorType<CudaT>* B,
    int64_t outer,
    int64_t n,
    int64_t inner);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
