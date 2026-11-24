// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>

#include "core/framework/float16.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Accumulation / error-output type of the *WithError ops: double for double inputs, float for everything else
// (fp16/bf16 inputs are widened to fp32, where the products of two such values are exact).
template <typename CudaT>
using Dot2ErrorType = std::conditional_t<std::is_same_v<CudaT, double>, double, float>;

// The scalar a kernel actually multiplies with: attributes are float, but the stock kernels convert them to T first
// (so an fp16 alpha is rounded to fp16). Mirror that so E is relative to the same alpha the stock op used.
template <typename T>
inline Dot2ErrorType<T> EffectiveScalar(float v) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return static_cast<Dot2ErrorType<T>>(v);
  } else {
    return T(v).ToFloat();
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
