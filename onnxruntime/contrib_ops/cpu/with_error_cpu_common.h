// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// The error terms rely on the exact sequence of rounded operations below (TwoProd through std::fma, TwoSum, ...):
// the compiler must not contract a*b+c into an FMA or reassociate. -ffast-math is not used for this target; the
// pragmas additionally switch contraction off for every file that includes this header (include it first).
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

#include <cmath>
#include <limits>
#include <type_traits>

#include "core/framework/float16.h"

// CPU building blocks shared by the *WithError kernels (Gemm, ReduceSum). They mirror
// contrib_ops/cuda/math/with_error_device.cuh so that E can be compared across providers.

namespace onnxruntime {
namespace contrib {
namespace with_error_cpu {

// Accumulation / error-output type: double for double, float for everything else (fp16/bf16 are widened to fp32,
// where the product of two such values is exact).
template <typename T>
using ErrorType = std::conditional_t<std::is_same_v<T, double>, double, float>;

template <typename T>
constexpr bool kHasStockCpuKernel = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
inline ErrorType<T> ToAcc(T v) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return v;
  } else {
    return v.ToFloat();
  }
}

template <typename T>
inline T FromAcc(ErrorType<T> v) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return v;
  } else {
    return T(v);
  }
}

// The scalar the kernel actually multiplies with: attributes are float, but a fp16/bf16 kernel converts them to T
// first. Mirror that so E is relative to the same alpha / beta Y was computed with.
template <typename T>
inline ErrorType<T> EffectiveScalar(float v) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return static_cast<ErrorType<T>>(v);
  } else {
    return T(v).ToFloat();
  }
}

// Running compensated sum (Dot2 / Neumaier with TwoSum): (s, c) with s + c ~ the exact sum of the added terms.
template <typename Acc>
struct Dot2State {
  Acc s = 0;
  Acc c = 0;

  // Add a term p whose rounding error is e (e = 0 when p is exact), i.e. the exact term is p + e.
  void Add(Acc p, Acc e) {
    const Acc t = s + p;  // TwoSum (Knuth)
    const Acc bb = t - s;
    const Acc q = (s - (t - bb)) + (p - bb);
    s = t;
    c = c + (q + e);
  }

  // Add the product a*b (TwoProd through std::fma: exact).
  void AddProduct(Acc a, Acc b) {
    const Acc p = a * b;
    const Acc e = std::fma(a, b, -p);
    Add(p, e);
  }
};

// E = (t - y) + corr where y is the op's actual result. A non-finite y (e.g. an fp16 result that overflowed to inf)
// has no meaningful error: return NaN.
template <typename Acc>
inline Acc FinishError(Acc t, Acc corr, Acc y) {
  if (!std::isfinite(y)) return std::numeric_limits<Acc>::quiet_NaN();
  return (t - y) + corr;
}

}  // namespace with_error_cpu
}  // namespace contrib
}  // namespace onnxruntime
