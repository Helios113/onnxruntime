// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/elementwise_with_error.h"

#include <vector>

#include "contrib_ops/cpu/with_error_broadcast.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(OP_NAME, OP, T)                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                         \
      OP_NAME,                                                                                           \
      kMSDomain,                                                                                         \
      1,                                                                                                 \
      T,                                                                                                 \
      kCudaExecutionProvider,                                                                            \
      (*KernelDefBuilder::Create())                                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                                         \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<Dot2ErrorType<ToCudaType<T>::MappedType>>()), \
      ElementwiseWithError<T, OP>);

#define REGISTER_OP(OP_NAME, OP)                \
  REGISTER_KERNEL_TYPED(OP_NAME, OP, float)     \
  REGISTER_KERNEL_TYPED(OP_NAME, OP, double)    \
  REGISTER_KERNEL_TYPED(OP_NAME, OP, MLFloat16) \
  REGISTER_KERNEL_TYPED(OP_NAME, OP, BFloat16)

REGISTER_OP(AddWithError, ElementwiseWithErrorOp::Add)
REGISTER_OP(SubWithError, ElementwiseWithErrorOp::Sub)
REGISTER_OP(MulWithError, ElementwiseWithErrorOp::Mul)
REGISTER_OP(DivWithError, ElementwiseWithErrorOp::Div)
REGISTER_OP(SqrtWithError, ElementwiseWithErrorOp::Sqrt)

template <typename T, ElementwiseWithErrorOp OP>
Status ElementwiseWithError<T, OP>::ComputeInternal(OpKernelContext* ctx) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  using ET = Dot2ErrorType<CudaT>;
  constexpr bool unary = OP == ElementwiseWithErrorOp::Sqrt;

  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = unary ? nullptr : ctx->Input<Tensor>(1);

  contrib::with_error_cpu::BroadcastLayout bl;
  if (unary) {
    bl.out_dims.assign(A->Shape().GetDims().begin(), A->Shape().GetDims().end());
    bl.a_strides.assign(bl.out_dims.size(), 0);
    bl.b_strides.assign(bl.out_dims.size(), 0);
    int64_t stride = 1;
    for (size_t a = bl.out_dims.size(); a-- > 0;) {
      bl.a_strides[a] = stride;
      stride *= bl.out_dims[a];
    }
  } else {
    ORT_RETURN_IF_ERROR(
        contrib::with_error_cpu::ComputeBroadcastLayout(A->Shape().GetDims(), B->Shape().GetDims(), bl));
  }
  ORT_RETURN_IF(bl.out_dims.size() > static_cast<size_t>(kElementwiseWithErrorMaxRank), "ElementwiseWithError supports at most ",
                kElementwiseWithErrorMaxRank, " dimensions, got ", bl.out_dims.size());

  ElementwiseWithErrorLayout layout;
  layout.rank = static_cast<int>(bl.out_dims.size());
  for (int a = 0; a < layout.rank; ++a) {
    const auto ua = static_cast<size_t>(a);
    layout.out_dims[a] = bl.out_dims[ua];
    layout.a_strides[a] = bl.a_strides[ua];
    layout.b_strides[a] = bl.b_strides[ua];
  }

  const TensorShape out_shape(bl.out_dims);
  Tensor* C = ctx->Output(0, out_shape);
  Tensor* E = ctx->Output(1, out_shape);
  const int64_t total = out_shape.Size();
  if (total == 0) return Status::OK();

  ElementwiseWithErrorImpl<CudaT>(
      Stream(ctx), OP,
      reinterpret_cast<const CudaT*>(A->Data<T>()),
      unary ? nullptr : reinterpret_cast<const CudaT*>(B->Data<T>()),
      reinterpret_cast<CudaT*>(C->MutableData<T>()),
      reinterpret_cast<ET*>(E->MutableDataRaw()),
      total, layout);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
