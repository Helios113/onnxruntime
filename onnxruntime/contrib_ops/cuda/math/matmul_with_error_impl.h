// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cuda/math/with_error_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Computes E[b,i,j] = (s - Y[b,i,j]) + c, where (s, c) is the Dot2 (Ogita-Rump-Oishi) result of
// sum_k A[b,i,k] * B[b,k,j], accumulated with k = 0..K-1 in order. Hence exact(A @ B) ~= Y + E.
// If Y[b,i,j] is not finite, E is NaN. If R is not nullptr it receives the compensated product s + c (rounded once).
//
// A and B are row-major. Element (i,k) of A is at a_off[b] + (trans_a ? k * M + i : i * K + k);
// element (k,j) of B is at b_off[b] + (trans_b ? j * K + k : k * N + j). Y and E are contiguous
// [batch, M, N]. a_off / b_off may be nullptr when batch == 1 (offset 0).
template <typename CudaT>
void MatMulWithErrorImpl(
    cudaStream_t stream,
    const CudaT* A,
    const CudaT* B,
    const CudaT* Y,
    Dot2ErrorType<CudaT>* E,
    Dot2ErrorType<CudaT>* R,
    int64_t batch,
    int64_t M,
    int64_t N,
    int64_t K,
    bool trans_a,
    bool trans_b,
    const int64_t* a_off,
    const int64_t* b_off);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
