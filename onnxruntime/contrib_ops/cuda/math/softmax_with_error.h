// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/math/softmax_with_error_impl.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Softmax along an axis with the error propagated through max, exp, sum and division as a second output.
template <typename T>
class SoftmaxWithError final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit SoftmaxWithError(const OpKernelInfo& info) : CudaKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", -1);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
