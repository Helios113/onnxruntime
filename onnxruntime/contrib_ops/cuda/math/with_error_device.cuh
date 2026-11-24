// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <math.h>

#include "contrib_ops/cuda/math/with_error_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"

// Device building blocks shared by the *WithError kernels (MatMul, Gemm, ReduceSum).
// Explicit round-to-nearest intrinsics everywhere: they can be neither contracted nor reordered by the compiler,
// so the result does not depend on build flags.

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename Acc>
struct Rn;

template <>
struct Rn<float> {
  __device__ static float Mul(float a, float b) { return __fmul_rn(a, b); }
  __device__ static float Fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
  __device__ static float Add(float a, float b) { return __fadd_rn(a, b); }
  __device__ static float Sub(float a, float b) { return __fsub_rn(a, b); }
};

template <>
struct Rn<double> {
  __device__ static double Mul(double a, double b) { return __dmul_rn(a, b); }
  __device__ static double Fma(double a, double b, double c) { return __fma_rn(a, b, c); }
  __device__ static double Add(double a, double b) { return __dadd_rn(a, b); }
  __device__ static double Sub(double a, double b) { return __dsub_rn(a, b); }
};

__device__ __forceinline__ float ToAcc(float v) { return v; }
__device__ __forceinline__ double ToAcc(double v) { return v; }
__device__ __forceinline__ float ToAcc(half v) { return __half2float(v); }
__device__ __forceinline__ float ToAcc(BFloat16 v) { return v.ToFloat(); }

__device__ __forceinline__ float RnDiv(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ double RnDiv(double a, double b) { return __ddiv_rn(a, b); }

template <typename CudaT, typename Acc>
__device__ __forceinline__ CudaT FromAccDev(Acc v);
template <>
__device__ __forceinline__ float FromAccDev<float, float>(float v) { return v; }
template <>
__device__ __forceinline__ double FromAccDev<double, double>(double v) { return v; }
template <>
__device__ __forceinline__ half FromAccDev<half, float>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ BFloat16 FromAccDev<BFloat16, float>(float v) { return BFloat16(v); }

template <typename Acc>
__device__ __forceinline__ Acc QuietNaN();
template <>
__device__ __forceinline__ float QuietNaN<float>() { return __int_as_float(0x7fc00000); }
template <>
__device__ __forceinline__ double QuietNaN<double>() { return __longlong_as_double(0x7ff8000000000000LL); }

// Running compensated sum (Dot2 / Neumaier with TwoSum): (s, c) with s + c ~ the exact sum of the added terms.
template <typename Acc>
struct Dot2State {
  using R = Rn<Acc>;
  Acc s = 0;
  Acc c = 0;

  // Add a term p whose rounding error is e (e = 0 when p is exact), i.e. the exact term is p + e.
  __device__ __forceinline__ void Add(Acc p, Acc e) {
    const Acc t = R::Add(s, p);  // TwoSum (Knuth)
    const Acc bb = R::Sub(t, s);
    const Acc q = R::Add(R::Sub(s, R::Sub(t, bb)), R::Sub(p, bb));
    s = t;
    c = R::Add(c, R::Add(q, e));
  }

  // Add the product a*b (TwoProd through the hardware FMA: exact).
  __device__ __forceinline__ void AddProduct(Acc a, Acc b) {
    const Acc p = R::Mul(a, b);
    const Acc e = R::Fma(a, b, -p);
    Add(p, e);
  }
};

// E = (t - y) + corr where y is the op's actual result. A non-finite y (e.g. an fp16 result that overflowed to inf)
// has no meaningful error: return NaN.
template <typename Acc>
__device__ __forceinline__ Acc FinishError(Acc t, Acc corr, Acc y) {
  if (!isfinite(y)) return QuietNaN<Acc>();
  return Rn<Acc>::Add(Rn<Acc>::Sub(t, y), corr);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
