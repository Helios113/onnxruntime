// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/softmax_with_error_impl.h"
#include "contrib_ops/cuda/math/with_error_device.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float DevExp(float v) { return expf(v); }
__device__ __forceinline__ double DevExp(double v) { return exp(v); }

// Assumed relative error of DevExp: the documented maximum of 2 ulp (float) / 1 ulp (double).
template <typename Acc>
constexpr Acc kExpRelErr = std::is_same_v<Acc, double> ? Acc(0x1p-52) : Acc(0x1p-22);

// t = x - m with its TwoSum error dt, u = exp(t), eps = u * dt (first-order propagation through exp).
template <typename Acc>
__device__ __forceinline__ void ExpTerm(Acc x, Acc m, Acc& u, Acc& eps) {
  using R = Rn<Acc>;
  const Acc nm = -m;
  const Acc t = R::Add(x, nm);  // TwoSum (Knuth)
  const Acc bb = R::Sub(t, x);
  Acc dt = R::Add(R::Sub(x, R::Sub(t, bb)), R::Sub(nm, bb));
  if (!isfinite(t)) dt = 0;
  u = DevExp(t);
  eps = R::Mul(u, dt);
}

// One block per softmax vector. Pass 1: max. Pass 2: compensated sum of exp(x - max) and the propagated error sum.
// Pass 3: recompute exp(x - max) (deterministic, same value as pass 2) and write Y, E.
template <typename CudaT>
__global__ void SoftmaxWithErrorKernel(const CudaT* X, CudaT* Y, Dot2ErrorType<CudaT>* E, Dot2ErrorType<CudaT>* B,
                                       int64_t outer, int64_t n, int64_t inner) {
  using Acc = Dot2ErrorType<CudaT>;
  using R = Rn<Acc>;
  __shared__ Acc sh_a[kThreads];
  __shared__ Acc sh_b[kThreads];
  __shared__ Acc sh_c[kThreads];

  const int tid = threadIdx.x;
  const int64_t vectors = outer * inner;
  for (int64_t v = blockIdx.x; v < vectors; v += gridDim.x) {
    const int64_t base = (v / inner) * n * inner + (v % inner);

    Acc m = -INFINITY;
    bool have = false;
    for (int64_t i = tid; i < n; i += kThreads) {
      const Acc xi = ToAcc(X[base + i * inner]);
      m = (!have || xi > m || isnan(xi)) ? xi : m;
      have = true;
    }
    // Threads without elements hold -inf and never win over a real value; NaN wins everywhere.
    sh_a[tid] = m;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const Acc o = sh_a[tid + stride];
        sh_a[tid] = (o > sh_a[tid] || isnan(o)) ? o : sh_a[tid];
      }
      __syncthreads();
    }
    m = sh_a[0];
    __syncthreads();

    Dot2State<Acc> sum;
    Acc eps_sum = 0;
    for (int64_t i = tid; i < n; i += kThreads) {
      Acc u, eps;
      ExpTerm<Acc>(ToAcc(X[base + i * inner]), m, u, eps);
      sum.Add(u, 0);
      eps_sum = R::Add(eps_sum, eps);
    }
    sh_a[tid] = sum.s;
    sh_b[tid] = sum.c;
    sh_c[tid] = eps_sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        Dot2State<Acc> acc;
        acc.s = sh_a[tid];
        acc.c = sh_b[tid];
        acc.Add(sh_a[tid + stride], sh_b[tid + stride]);
        sh_a[tid] = acc.s;
        sh_b[tid] = acc.c;
        sh_c[tid] = R::Add(sh_c[tid], sh_c[tid + stride]);
      }
      __syncthreads();
    }
    const Acc s = sh_a[0];
    const Acc ds = R::Add(sh_b[0], sh_c[0]);
    __syncthreads();

    for (int64_t i = tid; i < n; i += kThreads) {
      Acc u, eps;
      ExpTerm<Acc>(ToAcc(X[base + i * inner]), m, u, eps);
      const Acc yi = RnDiv(u, s);
      const Acc r = RnDiv(R::Fma(-yi, s, u), s);
      const Acc err = R::Add(r, RnDiv(R::Sub(eps, R::Mul(yi, ds)), s));
      const CudaT y = FromAccDev<CudaT, Acc>(yi);
      Y[base + i * inner] = y;
      const Acc ei = FinishError<Acc>(yi, err, ToAcc(y));
      E[base + i * inner] = ei;
      constexpr Acc rho = kExpRelErr<Acc>;
      B[base + i * inner] = isnan(ei) ? ei : R::Add(R::Mul(rho, R::Add(yi, yi)), R::Mul(4 * rho, fabs(ei)));
    }
  }
}

}  // namespace

template <typename CudaT>
void SoftmaxWithErrorImpl(cudaStream_t stream, const CudaT* X, CudaT* Y, Dot2ErrorType<CudaT>* E,
                          Dot2ErrorType<CudaT>* B, int64_t outer, int64_t n, int64_t inner) {
  const int64_t vectors = outer * inner;
  if (vectors == 0 || n == 0) return;
  const unsigned int blocks = static_cast<unsigned int>(vectors < 65535 ? vectors : 65535);
  SoftmaxWithErrorKernel<CudaT><<<blocks, kThreads, 0, stream>>>(X, Y, E, B, outer, n, inner);
}

#define SPECIALIZED_IMPL(CudaT)                                                                                   \
  template void SoftmaxWithErrorImpl<CudaT>(cudaStream_t, const CudaT*, CudaT*, Dot2ErrorType<CudaT>*,          \
                                            Dot2ErrorType<CudaT>*, int64_t, int64_t, int64_t);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
