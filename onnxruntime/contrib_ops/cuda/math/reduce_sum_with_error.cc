// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/reduce_sum_with_error.h"

#include <algorithm>

#include "contrib_ops/cuda/math/reduce_sum_with_error_impl.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                         \
      ReduceSumWithError,                                                                                \
      kMSDomain,                                                                                         \
      1,                                                                                                 \
      T,                                                                                                 \
      kCudaExecutionProvider,                                                                            \
      (*KernelDefBuilder::Create())                                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                                         \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<Dot2ErrorType<ToCudaType<T>::MappedType>>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 1),                                                       \
      ReduceSumWithError<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status ReduceSumWithError<T>::ComputeInternal(OpKernelContext* ctx) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  using ET = Dot2ErrorType<CudaT>;

  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* axes_tensor = ctx->InputCount() >= 2 ? ctx->Input<Tensor>(1) : nullptr;  // optional

  // 1. Y from the stock ReduceSum (also allocates output 0).
  ORT_RETURN_IF_ERROR(reduce_sum_.ComputeInternal(ctx));
  const Tensor* Y = ctx->Output<Tensor>(0);
  Tensor* E = ctx->Output(1, Y->Shape());
  const int64_t out_count = Y->Shape().Size();
  if (out_count == 0) return Status::OK();

  cudaStream_t stream = Stream(ctx);

  // Which axes are reduced (same rules as the stock op: empty axes + noop_with_empty_axes -> identity).
  const auto& dims = X->Shape().GetDims();
  const int64_t rank = static_cast<int64_t>(dims.size());
  ORT_RETURN_IF(rank > kReduceWithErrorMaxRank, "ReduceSumWithError supports at most ", kReduceWithErrorMaxRank,
                " dimensions, got ", rank);

  std::vector<int64_t> axes;
  if (axes_tensor != nullptr) {
    ORT_RETURN_IF(axes_tensor->Shape().NumDimensions() != 1, "An axes tensor must be a vector tensor.");
    const auto* data = axes_tensor->Data<int64_t>();
    axes.assign(data, data + axes_tensor->Shape()[0]);
  }
  if (axes.empty() && noop_with_empty_axes_) {
    // Y == X, nothing is rounded.
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(E->MutableDataRaw(), 0, E->SizeInBytes(), stream));
    return Status::OK();
  }
  std::vector<bool> reduced(static_cast<size_t>(rank), axes.empty());
  for (int64_t a : axes) {
    reduced[static_cast<size_t>(HandleNegativeAxis(a, rank))] = true;
  }

  std::vector<int64_t> strides(static_cast<size_t>(rank), 1);
  for (int64_t a = rank - 2; a >= 0; --a) {
    strides[static_cast<size_t>(a)] = strides[static_cast<size_t>(a) + 1] * dims[static_cast<size_t>(a) + 1];
  }

  ReduceSumWithErrorLayout layout;
  for (int64_t a = 0; a < rank; ++a) {
    const auto ua = static_cast<size_t>(a);
    if (reduced[ua]) {
      layout.reduced_dims[layout.n_reduced] = dims[ua];
      layout.reduced_strides[layout.n_reduced++] = strides[ua];
    } else {
      layout.kept_dims[layout.n_kept] = dims[ua];
      layout.kept_strides[layout.n_kept++] = strides[ua];
    }
  }

  // 2. E on the same stream, after the reduction.
  ReduceSumWithErrorImpl<CudaT>(
      stream,
      reinterpret_cast<const CudaT*>(X->Data<T>()),
      reinterpret_cast<const CudaT*>(Y->Data<T>()),
      reinterpret_cast<ET*>(E->MutableDataRaw()),
      out_count, layout);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
