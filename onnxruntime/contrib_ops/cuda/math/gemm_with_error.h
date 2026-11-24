// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/math/gemm.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Y is produced by the stock CUDA Gemm kernel (held as a member and invoked, so Y is bit-identical to it);
// E is the rounding error of Y, so that the exact result is approximately Y + E.
template <typename T>
class GemmWithError final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit GemmWithError(const OpKernelInfo& info)
      : CudaKernel(info),
        gemm_(info),
        trans_A_{info.GetAttrOrDefault<int64_t>("transA", 0) != 0},
        trans_B_{info.GetAttrOrDefault<int64_t>("transB", 0) != 0},
        alpha_{info.GetAttrOrDefault<float>("alpha", 1.0f)},
        beta_{info.GetAttrOrDefault<float>("beta", 1.0f)} {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ::onnxruntime::cuda::Gemm<T> gemm_;
  const bool trans_A_;
  const bool trans_B_;
  const float alpha_;
  const float beta_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
