// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/math/elementwise_with_error_impl.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Add / Sub / Mul / Div (numpy broadcasting) / Sqrt with the rounding error of the result as a second output.
template <typename T, ElementwiseWithErrorOp OP>
class ElementwiseWithError final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit ElementwiseWithError(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
