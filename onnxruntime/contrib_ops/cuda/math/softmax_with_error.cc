// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/softmax_with_error.h"

#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                         \
      SoftmaxWithError,                                                                                  \
      kMSDomain,                                                                                         \
      1,                                                                                                 \
      T,                                                                                                 \
      kCudaExecutionProvider,                                                                            \
      (*KernelDefBuilder::Create())                                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                                         \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<Dot2ErrorType<ToCudaType<T>::MappedType>>()), \
      SoftmaxWithError<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status SoftmaxWithError<T>::ComputeInternal(OpKernelContext* ctx) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  using ET = Dot2ErrorType<CudaT>;

  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = ctx->Output(0, shape);
  Tensor* E = ctx->Output(1, shape);
  Tensor* B = ctx->Output(2, shape);
  if (shape.Size() == 0) return Status::OK();

  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, static_cast<int64_t>(shape.NumDimensions())));
  SoftmaxWithErrorImpl<CudaT>(
      Stream(ctx),
      reinterpret_cast<const CudaT*>(X->Data<T>()),
      reinterpret_cast<CudaT*>(Y->MutableData<T>()),
      reinterpret_cast<ET*>(E->MutableDataRaw()),
      reinterpret_cast<ET*>(B->MutableDataRaw()),
      shape.SizeToDimension(axis), shape[axis], shape.SizeFromDimension(axis + 1));

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
