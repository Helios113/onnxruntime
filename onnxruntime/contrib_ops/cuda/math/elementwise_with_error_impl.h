// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cuda/math/with_error_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kElementwiseWithErrorMaxRank = 8;

enum class ElementwiseWithErrorOp { Add, Sub, Mul, Div, Sqrt };

// Broadcast layout: output dims plus each input's element strides in the output's index space (0 along broadcast dims).
struct ElementwiseWithErrorLayout {
  int rank = 0;
  int64_t out_dims[kElementwiseWithErrorMaxRank] = {};
  int64_t a_strides[kElementwiseWithErrorMaxRank] = {};
  int64_t b_strides[kElementwiseWithErrorMaxRank] = {};
};

// C = A op B (correctly rounded, in the accumulation type, then rounded to CudaT) and E, the rounding error of C:
// exact(A op B) ~= C + E, from an error-free transformation (TwoSum / fma remainder). B is unused for Sqrt.
// E is NaN where C is not finite.
template <typename CudaT>
void ElementwiseWithErrorImpl(
    cudaStream_t stream,
    ElementwiseWithErrorOp op,
    const CudaT* A,
    const CudaT* B,
    CudaT* C,
    Dot2ErrorType<CudaT>* E,
    int64_t total,
    const ElementwiseWithErrorLayout& layout);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
