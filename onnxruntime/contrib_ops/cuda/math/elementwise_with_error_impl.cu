// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/elementwise_with_error_impl.h"
#include "contrib_ops/cuda/math/with_error_device.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

__device__ __forceinline__ float RnSqrt(float a) { return __fsqrt_rn(a); }
__device__ __forceinline__ double RnSqrt(double a) { return __dsqrt_rn(a); }

// r = op(a, b) in Acc, err = its rounding error (exact result ~= r + err). Same formulas as the CPU kernels.
template <ElementwiseWithErrorOp OP, typename Acc>
__device__ __forceinline__ void Apply(Acc a, Acc b, Acc& r, Acc& err) {
  using R = Rn<Acc>;
  if constexpr (OP == ElementwiseWithErrorOp::Add || OP == ElementwiseWithErrorOp::Sub) {
    const Acc bv = OP == ElementwiseWithErrorOp::Sub ? -b : b;
    r = R::Add(a, bv);  // TwoSum (Knuth)
    const Acc bb = R::Sub(r, a);
    err = R::Add(R::Sub(a, R::Sub(r, bb)), R::Sub(bv, bb));
  } else if constexpr (OP == ElementwiseWithErrorOp::Mul) {
    r = R::Mul(a, b);
    err = R::Fma(a, b, -r);
  } else if constexpr (OP == ElementwiseWithErrorOp::Div) {
    r = RnDiv(a, b);
    err = RnDiv(R::Fma(-r, b, a), b);
  } else {
    r = RnSqrt(a);
    err = r == Acc(0) ? Acc(0) : RnDiv(R::Fma(-r, r, a), R::Add(r, r));
  }
}

template <typename CudaT, ElementwiseWithErrorOp OP>
__global__ void ElementwiseWithErrorKernel(const CudaT* A, const CudaT* B, CudaT* C, Dot2ErrorType<CudaT>* E,
                                           int64_t total, ElementwiseWithErrorLayout lay) {
  using Acc = Dot2ErrorType<CudaT>;

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int64_t off_a = 0;
  int64_t off_b = 0;
  int64_t rem = idx;
  for (int a = lay.rank - 1; a >= 0; --a) {
    const int64_t coord = rem % lay.out_dims[a];
    rem /= lay.out_dims[a];
    off_a += coord * lay.a_strides[a];
    off_b += coord * lay.b_strides[a];
  }

  const Acc av = ToAcc(A[off_a]);
  const Acc bv = OP == ElementwiseWithErrorOp::Sqrt ? Acc(0) : ToAcc(B[off_b]);
  Acc r;
  Acc err;
  Apply<OP, Acc>(av, bv, r, err);
  const CudaT c = FromAccDev<CudaT, Acc>(r);
  C[idx] = c;
  E[idx] = FinishError<Acc>(r, err, ToAcc(c));
}

template <typename CudaT, ElementwiseWithErrorOp OP>
void Launch(cudaStream_t stream, const CudaT* A, const CudaT* B, CudaT* C, Dot2ErrorType<CudaT>* E, int64_t total,
            const ElementwiseWithErrorLayout& layout) {
  constexpr int kThreads = 256;
  const int64_t blocks = (total + kThreads - 1) / kThreads;
  ElementwiseWithErrorKernel<CudaT, OP><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      A, B, C, E, total, layout);
}

}  // namespace

template <typename CudaT>
void ElementwiseWithErrorImpl(cudaStream_t stream, ElementwiseWithErrorOp op, const CudaT* A, const CudaT* B, CudaT* C,
                              Dot2ErrorType<CudaT>* E, int64_t total, const ElementwiseWithErrorLayout& layout) {
  if (total == 0) return;
  switch (op) {
    case ElementwiseWithErrorOp::Add:
      Launch<CudaT, ElementwiseWithErrorOp::Add>(stream, A, B, C, E, total, layout);
      break;
    case ElementwiseWithErrorOp::Sub:
      Launch<CudaT, ElementwiseWithErrorOp::Sub>(stream, A, B, C, E, total, layout);
      break;
    case ElementwiseWithErrorOp::Mul:
      Launch<CudaT, ElementwiseWithErrorOp::Mul>(stream, A, B, C, E, total, layout);
      break;
    case ElementwiseWithErrorOp::Div:
      Launch<CudaT, ElementwiseWithErrorOp::Div>(stream, A, B, C, E, total, layout);
      break;
    case ElementwiseWithErrorOp::Sqrt:
      Launch<CudaT, ElementwiseWithErrorOp::Sqrt>(stream, A, B, C, E, total, layout);
      break;
  }
}

#define SPECIALIZED_IMPL(CudaT)                                                                                   \
  template void ElementwiseWithErrorImpl<CudaT>(cudaStream_t, ElementwiseWithErrorOp, const CudaT*, const CudaT*, \
                                                CudaT*, Dot2ErrorType<CudaT>*, int64_t,                          \
                                                const ElementwiseWithErrorLayout&);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
