// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/with_error_cpu_common.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

namespace onnxruntime {
namespace contrib {

using namespace with_error_cpu;

// Y = ReduceSum(data, axes), plus E, the rounding error of Y: the exact sum ~= Y + E.
// float / double: Y comes from the stock CPU ReduceSum kernel (held and invoked, so Y is bit-identical to it).
// float16 / bfloat16: there is no stock CPU kernel, so Y is computed here: the reduced elements are summed in
// row-major order in fp32 and rounded once to T.
// E is (s - Y) + c with (s, c) the compensated (TwoSum) sum of the reduced elements in row-major order of the reduced
// axes. Same semantics as the CUDA kernel.
template <typename T>
class ReduceSumWithError final : public OpKernel {
 public:
  explicit ReduceSumWithError(const OpKernelInfo& info)
      : OpKernel(info),
        keepdims_{info.GetAttrOrDefault<int64_t>("keepdims", 1) != 0},
        noop_with_empty_axes_{info.GetAttrOrDefault<int64_t>("noop_with_empty_axes", 0) != 0} {
    if constexpr (kHasStockCpuKernel<T>) {
      reduce_sum_ = std::make_unique<ReduceSum<T>>(info);
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  std::unique_ptr<OpKernel> reduce_sum_;  // stock kernel, float / double only
  const bool keepdims_;
  const bool noop_with_empty_axes_;
};

#define REGISTER_KERNEL_TYPED(T)                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
      ReduceSumWithError,                                                    \
      kMSDomain,                                                             \
      1,                                                                     \
      T,                                                                     \
      kCpuExecutionProvider,                                                 \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())             \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<ErrorType<T>>()), \
      ReduceSumWithError<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status ReduceSumWithError<T>::Compute(OpKernelContext* ctx) const {
  using Acc = ErrorType<T>;
  constexpr bool stock = kHasStockCpuKernel<T>;

  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* axes_tensor = ctx->InputCount() >= 2 ? ctx->Input<Tensor>(1) : nullptr;  // optional

  const auto& dims = X->Shape().GetDims();
  const int64_t rank = static_cast<int64_t>(dims.size());

  std::vector<int64_t> axes;
  if (axes_tensor != nullptr) {
    ORT_RETURN_IF(axes_tensor->Shape().NumDimensions() != 1, "An axes tensor must be a vector tensor.");
    const auto* data = axes_tensor->Data<int64_t>();
    axes.assign(data, data + axes_tensor->Shape()[0]);
  }
  const bool noop = axes.empty() && noop_with_empty_axes_;

  // Which axes are reduced (same rules as the stock op: empty axes + noop_with_empty_axes -> identity).
  std::vector<bool> reduced(static_cast<size_t>(rank), axes.empty() && !noop);
  for (int64_t a : axes) {
    reduced[static_cast<size_t>(HandleNegativeAxis(a, rank))] = true;
  }

  // 1. Y with the stock kernel (also allocates output 0), or allocate it for the types without one.
  if constexpr (stock) {
    ORT_RETURN_IF_ERROR(reduce_sum_->Compute(ctx));
  } else {
    TensorShapeVector out_dims;
    for (int64_t a = 0; a < rank; ++a) {
      if (!reduced[static_cast<size_t>(a)]) {
        out_dims.push_back(dims[static_cast<size_t>(a)]);
      } else if (keepdims_) {
        out_dims.push_back(1);
      }
    }
    ctx->Output(0, TensorShape(out_dims));
  }
  Tensor* Y = ctx->Output<Tensor>(0);
  Tensor* E = ctx->Output(1, Y->Shape());
  const int64_t out_count = Y->Shape().Size();
  if (out_count == 0) return Status::OK();

  Acc* e_data = E->MutableData<Acc>();
  const T* x_data = X->Data<T>();
  T* y_data = Y->MutableData<T>();

  if (noop) {
    // Y == X, nothing is rounded.
    if constexpr (!stock) std::memcpy(y_data, x_data, X->SizeInBytes());
    std::fill(e_data, e_data + out_count, Acc(0));
    return Status::OK();
  }

  std::vector<int64_t> strides(static_cast<size_t>(rank), 1);
  for (int64_t a = rank - 2; a >= 0; --a) {
    strides[static_cast<size_t>(a)] = strides[static_cast<size_t>(a) + 1] * dims[static_cast<size_t>(a) + 1];
  }

  std::vector<int64_t> kept_dims, kept_strides, red_dims, red_strides;
  for (int64_t a = 0; a < rank; ++a) {
    const auto ua = static_cast<size_t>(a);
    if (reduced[ua]) {
      red_dims.push_back(dims[ua]);
      red_strides.push_back(strides[ua]);
    } else {
      kept_dims.push_back(dims[ua]);
      kept_strides.push_back(strides[ua]);
    }
  }
  int64_t red_count = 1;
  for (int64_t d : red_dims) red_count *= d;
  const int n_kept = static_cast<int>(kept_dims.size());
  const int n_red = static_cast<int>(red_dims.size());

  // 2. E (and Y for fp16 / bf16), one output element per task.
  const TensorOpCost cost{static_cast<double>(red_count) * sizeof(T), static_cast<double>(sizeof(Acc)),
                          static_cast<double>(red_count) * 8.0};
  concurrency::ThreadPool::TryParallelFor(
      ctx->GetOperatorThreadPool(), static_cast<std::ptrdiff_t>(out_count), cost,
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        std::vector<int64_t> counter(static_cast<size_t>(n_red));
        for (std::ptrdiff_t o = first; o < last; ++o) {
          // Base offset of this output's reduction group in the input.
          int64_t base = 0;
          int64_t rem = o;
          for (int a = n_kept - 1; a >= 0; --a) {
            base += (rem % kept_dims[static_cast<size_t>(a)]) * kept_strides[static_cast<size_t>(a)];
            rem /= kept_dims[static_cast<size_t>(a)];
          }

          Dot2State<Acc> d;
          Acc plain = 0;  // row-major plain sum, only used to produce Y for the types without a stock kernel
          std::fill(counter.begin(), counter.end(), 0);
          int64_t offset = base;
          for (int64_t r = 0; r < red_count; ++r) {
            const Acc v = ToAcc(x_data[offset]);
            d.Add(v, Acc(0));
            if constexpr (!stock) plain = plain + v;
            // advance the odometer over the reduced axes, last axis fastest (row-major order)
            for (int a = n_red - 1; a >= 0; --a) {
              const auto ua = static_cast<size_t>(a);
              offset += red_strides[ua];
              if (++counter[ua] < red_dims[ua]) break;
              offset -= red_dims[ua] * red_strides[ua];
              counter[ua] = 0;
            }
          }
          if constexpr (!stock) y_data[o] = FromAcc<T>(plain);
          e_data[o] = FinishError<Acc>(d.s, d.c, ToAcc(y_data[o]));
        }
      });

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
