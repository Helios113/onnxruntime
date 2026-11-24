// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_with_error.h"

#include <optional>

#include "contrib_ops/cuda/math/matmul_with_error_impl.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/math/matmul.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                        \
      MatMulWithError,                                                                                  \
      kMSDomain,                                                                                        \
      1,                                                                                                \
      T,                                                                                                \
      kCudaExecutionProvider,                                                                           \
      (*KernelDefBuilder::Create())                                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                                        \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<Dot2ErrorType<ToCudaType<T>::MappedType>>()), \
      MatMulWithError<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status MatMulWithError<T>::ComputeInternal(OpKernelContext* ctx) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  using ET = Dot2ErrorType<CudaT>;

  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);

  // Transpose flags are ignored for rank-1 inputs, as in MatMul.
  const bool trans_a = A->Shape().NumDimensions() == 1 ? false : trans_A_;
  const bool trans_b = B->Shape().NumDimensions() == 1 ? false : trans_B_;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape(), trans_a, trans_b, false, false, false));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  Tensor* E = ctx->Output(1, helper.OutputShape());
  Tensor* R = ctx->Output(2, helper.OutputShape());  // optional compensated product; nullptr if not requested
  if (Y->Shape().Size() == 0) return Status::OK();

  cudaStream_t stream = Stream(ctx);

  if (helper.K() == 0) {
    // Empty reduction: Y = 0 and the error is 0.
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), stream));
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(E->MutableDataRaw(), 0, E->SizeInBytes(), stream));
    if (R != nullptr) CUDA_RETURN_IF_ERROR(cudaMemsetAsync(R->MutableDataRaw(), 0, R->SizeInBytes(), stream));
    return Status::OK();
  }

  // 1. Y = A @ B with the regular cuBLAS path (E depends on Y, so this must come first).
  ORT_RETURN_IF_ERROR(FuncMatMul<T>(this, ctx, A, B, 1.0f, trans_a, trans_b, false, false, Y));

  // 2. E from the Dot2 kernel, same stream so it is ordered after the GEMM.
  // Derive the batch count from the output shape: the helper only fills its offsets on request.
  const size_t batch = static_cast<size_t>(Y->Shape().Size() / (helper.M() * helper.N()));
  const int64_t* a_off = nullptr;
  const int64_t* b_off = nullptr;
  // Only allocate the offset buffers when there is a batch: a zero-sized CudaAsyncBuffer throws "alloc failed".
  std::optional<CudaAsyncBuffer<int64_t>> a_offsets;
  std::optional<CudaAsyncBuffer<int64_t>> b_offsets;
  if (batch > 1) {
    a_offsets.emplace(this, batch);
    b_offsets.emplace(this, batch);
    helper.FillOffsets();
    for (size_t i = 0; i < batch; ++i) {
      a_offsets->CpuPtr()[i] = static_cast<int64_t>(helper.LeftOffsets()[i]);
      b_offsets->CpuPtr()[i] = static_cast<int64_t>(helper.RightOffsets()[i]);
    }
    ORT_RETURN_IF_ERROR(a_offsets->CopyToGpu(ctx->GetComputeStream()));
    ORT_RETURN_IF_ERROR(b_offsets->CopyToGpu(ctx->GetComputeStream()));
    a_off = a_offsets->GpuPtr();
    b_off = b_offsets->GpuPtr();
  }

  MatMulWithErrorImpl<CudaT>(
      stream,
      reinterpret_cast<const CudaT*>(A->Data<T>()),
      reinterpret_cast<const CudaT*>(B->Data<T>()),
      reinterpret_cast<const CudaT*>(Y->Data<T>()),
      reinterpret_cast<ET*>(E->MutableDataRaw()),
      R != nullptr ? reinterpret_cast<ET*>(R->MutableDataRaw()) : nullptr,
      static_cast<int64_t>(batch), helper.M(), helper.N(), helper.K(),
      trans_a, trans_b, a_off, b_off);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
