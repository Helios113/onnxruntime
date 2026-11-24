// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/gemm_with_error_impl.h"
#include "contrib_ops/cuda/math/with_error_device.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename CudaT>
__global__ void GemmWithErrorKernel(
    const CudaT* A, const CudaT* B, const CudaT* C, const CudaT* Y, Dot2ErrorType<CudaT>* E,
    int64_t total, int64_t M, int64_t N, int64_t K, bool trans_a, bool trans_b,
    Dot2ErrorType<CudaT> alpha, Dot2ErrorType<CudaT> beta, int64_t c_rows, int64_t c_cols) {
  using Acc = Dot2ErrorType<CudaT>;
  using R = Rn<Acc>;

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int64_t j = idx % N;
  const int64_t i = idx / N;

  Dot2State<Acc> d;
  for (int64_t k = 0; k < K; ++k) {
    d.AddProduct(ToAcc(A[trans_a ? k * M + i : i * K + k]), ToAcc(B[trans_b ? j * K + k : k * N + j]));
  }

  // alpha * (s + c): TwoProd(alpha, s) is exact; alpha * c is already a second-order term.
  const Acc p1 = R::Mul(alpha, d.s);
  const Acc e1 = R::Fma(alpha, d.s, -p1);
  Acc hi = p1;
  Acc corr = R::Add(e1, R::Mul(alpha, d.c));

  if (C != nullptr) {
    const Acc cv = ToAcc(C[(c_rows == 1 ? 0 : i) * c_cols + (c_cols == 1 ? 0 : j)]);
    const Acc p2 = R::Mul(beta, cv);
    const Acc e2 = R::Fma(beta, cv, -p2);
    // TwoSum(p1, p2)
    hi = R::Add(p1, p2);
    const Acc bb = R::Sub(hi, p1);
    const Acc q = R::Add(R::Sub(p1, R::Sub(hi, bb)), R::Sub(p2, bb));
    corr = R::Add(corr, R::Add(q, e2));
  }

  E[idx] = FinishError<Acc>(hi, corr, ToAcc(Y[idx]));
}

}  // namespace

template <typename CudaT>
void GemmWithErrorImpl(
    cudaStream_t stream, const CudaT* A, const CudaT* B, const CudaT* C, const CudaT* Y, Dot2ErrorType<CudaT>* E,
    int64_t M, int64_t N, int64_t K, bool trans_a, bool trans_b,
    Dot2ErrorType<CudaT> alpha, Dot2ErrorType<CudaT> beta, int64_t c_rows, int64_t c_cols) {
  const int64_t total = M * N;
  if (total == 0) return;
  constexpr int kThreads = 256;
  const int64_t blocks = (total + kThreads - 1) / kThreads;
  GemmWithErrorKernel<CudaT><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      A, B, C, Y, E, total, M, N, K, trans_a, trans_b, alpha, beta, c_rows, c_cols);
}

#define SPECIALIZED_IMPL(CudaT)                                                                                  \
  template void GemmWithErrorImpl<CudaT>(cudaStream_t, const CudaT*, const CudaT*, const CudaT*, const CudaT*,   \
                                         Dot2ErrorType<CudaT>*, int64_t, int64_t, int64_t, bool, bool,           \
                                         Dot2ErrorType<CudaT>, Dot2ErrorType<CudaT>, int64_t, int64_t);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
