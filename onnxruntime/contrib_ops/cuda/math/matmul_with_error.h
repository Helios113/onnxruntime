// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Y = A @ B via cuBLAS (identical to the CUDA MatMul), plus E, the rounding error of Y computed with a
// Dot2 kernel, so that exact(A @ B) ~= Y + E.
template <typename T>
class MatMulWithError final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulWithError(const OpKernelInfo& info)
      : CudaKernel(info),
        trans_A_{info.GetAttrOrDefault<int64_t>("transA", 0) != 0},
        trans_B_{info.GetAttrOrDefault<int64_t>("transB", 0) != 0} {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  const bool trans_A_;
  const bool trans_B_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
