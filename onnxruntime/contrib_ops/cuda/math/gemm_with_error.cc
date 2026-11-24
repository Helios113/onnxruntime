// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/gemm_with_error.h"

#include "contrib_ops/cuda/math/gemm_with_error_impl.h"
#include "core/providers/cpu/math/gemm_helper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                         \
      GemmWithError,                                                                                     \
      kMSDomain,                                                                                         \
      1,                                                                                                 \
      T,                                                                                                 \
      kCudaExecutionProvider,                                                                            \
      (*KernelDefBuilder::Create())                                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                                         \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<Dot2ErrorType<ToCudaType<T>::MappedType>>()), \
      GemmWithError<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status GemmWithError<T>::ComputeInternal(OpKernelContext* ctx) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  using ET = Dot2ErrorType<CudaT>;

  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const Tensor* C = ctx->Input<Tensor>(2);  // optional

  GemmHelper helper(A->Shape(), trans_A_, B->Shape(), trans_B_, C != nullptr ? C->Shape() : TensorShape({}));
  ORT_RETURN_IF_ERROR(helper.State());
  const int64_t M = helper.M();
  const int64_t N = helper.N();
  const int64_t K = helper.K();

  // 1. Y from the stock Gemm kernel (also allocates output 0).
  ORT_RETURN_IF_ERROR(gemm_.ComputeInternal(ctx));
  const Tensor* Y = ctx->Output<Tensor>(0);

  Tensor* E = ctx->Output(1, {M, N});
  if (Y->Shape().Size() == 0) return Status::OK();

  // The stock op ignores C when beta == 0.
  const bool use_c = C != nullptr && beta_ != 0.0f;
  int64_t c_rows = 1;
  int64_t c_cols = 1;
  if (use_c) {
    const auto& cs = C->Shape();
    if (cs.NumDimensions() == 2) {
      c_rows = cs[0];
      c_cols = cs[1];
    } else if (cs.NumDimensions() == 1) {
      c_cols = cs[0];
    }
  }

  // 2. E on the same stream, after the GEMM.
  GemmWithErrorImpl<CudaT>(
      Stream(ctx),
      reinterpret_cast<const CudaT*>(A->Data<T>()),
      reinterpret_cast<const CudaT*>(B->Data<T>()),
      use_c ? reinterpret_cast<const CudaT*>(C->Data<T>()) : nullptr,
      reinterpret_cast<const CudaT*>(Y->Data<T>()),
      reinterpret_cast<ET*>(E->MutableDataRaw()),
      M, N, K, trans_A_, trans_B_,
      EffectiveScalar<T>(alpha_), EffectiveScalar<T>(beta_), c_rows, c_cols);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
