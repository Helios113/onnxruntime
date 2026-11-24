// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/reduction/reduction_ops.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Y is produced by the stock CUDA ReduceSum (held as a member and invoked, so Y is bit-identical to it);
// E is the rounding error of Y from a compensated TwoSum pass over the same elements, so exact sum ~= Y + E.
template <typename T>
class ReduceSumWithError final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit ReduceSumWithError(const OpKernelInfo& info)
      : CudaKernel(info),
        reduce_sum_(info),
        noop_with_empty_axes_{info.GetAttrOrDefault<int64_t>("noop_with_empty_axes", 0) != 0} {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ::onnxruntime::cuda::ReduceSum<T> reduce_sum_;
  const bool noop_with_empty_axes_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
