// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cuda/math/with_error_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// E for Y = alpha * op(A) @ op(B) + beta * C (ONNX Gemm), where Y is the stock Gemm result.
//   dot  = Dot2 over k (k = 0..K-1 in order)  -> (s, c)
//   v    = alpha*(s + c) + beta*C             -> TwoProd(alpha, s) and TwoProd(beta, C), joined by a TwoSum
//   E    = (v_hi - Y) + corrections           (NaN if Y is not finite)
// so that the exact result is approximately Y + E. alpha / beta are the values the stock kernel used (already
// converted to T). C is optional: pass c == nullptr to ignore it (as the stock op does for beta == 0 or no C).
// C is broadcast to [M, N]: element (i,j) is C[(c_rows == 1 ? 0 : i) * c_cols + (c_cols == 1 ? 0 : j)].
template <typename CudaT>
void GemmWithErrorImpl(
    cudaStream_t stream,
    const CudaT* A,
    const CudaT* B,
    const CudaT* C,
    const CudaT* Y,
    Dot2ErrorType<CudaT>* E,
    int64_t M,
    int64_t N,
    int64_t K,
    bool trans_a,
    bool trans_b,
    Dot2ErrorType<CudaT> alpha,
    Dot2ErrorType<CudaT> beta,
    int64_t c_rows,
    int64_t c_cols);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
