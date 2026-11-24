// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/with_error_cpu_common.h"

#include <vector>

#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

using namespace with_error_cpu;

// Softmax along `axis` with the error propagated through the chain
//   t_i = x_i - m,   u_i = exp(t_i),   s = sum_j u_j,   y_i = u_i / s
// computed in the accumulation type (double for double, float otherwise). Per element:
//   subtraction  t_i + dt_i = x_i - m exactly (TwoSum), dt_i carried through exp to first order: eps_i = u_i dt_i
//   sum          s + c = sum_j u_j exactly (compensated TwoSum accumulation); ds = c + sum_j eps_j
//   division     u_i / s = y_i + r_i, r_i = fma(-y_i, s, u_i) / s
// and, to first order, softmax_i of the exact chain - y_i = r_i + (eps_i - y_i ds) / s. The max involves no rounding,
// the rounding of exp itself is not modelled (the computed u_i is taken as exact). For fp16 / bf16, Y is y rounded once
// and E = (y - Y) + that error. E is NaN where Y is not finite. Same semantics as the CUDA kernel.
// B bounds the rest, mainly the rounding of exp, for which a relative error of kExpRelErr is assumed (documented
// maximum, 1 ulp, of the usual libm; not measured): B = 2 rho y + 4 rho |E|, NaN where E is.
template <typename Acc>
constexpr Acc kExpRelErr = std::is_same_v<Acc, double> ? Acc(0x1p-52) : Acc(0x1p-23);

template <typename T>
class SoftmaxWithError final : public OpKernel {
 public:
  explicit SoftmaxWithError(const OpKernelInfo& info) : OpKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", -1);
  }

  Status Compute(OpKernelContext* ctx) const override {
    using Acc = ErrorType<T>;

    const Tensor* X = ctx->Input<Tensor>(0);
    const TensorShape& shape = X->Shape();
    Tensor* Y = ctx->Output(0, shape);
    Tensor* E = ctx->Output(1, shape);
    Tensor* B = ctx->Output(2, shape);
    if (shape.Size() == 0) return Status::OK();

    const size_t axis = onnxruntime::narrow<size_t>(HandleNegativeAxis(axis_, static_cast<int64_t>(shape.NumDimensions())));
    const int64_t n = shape[axis];
    const int64_t outer = shape.SizeToDimension(axis);
    const int64_t inner = shape.SizeFromDimension(axis + 1);

    const T* x_data = X->Data<T>();
    T* y_data = Y->MutableData<T>();
    Acc* e_data = E->MutableData<Acc>();
    Acc* b_data = B->MutableData<Acc>();
    constexpr Acc rho = kExpRelErr<Acc>;

    const TensorOpCost cost{static_cast<double>(n) * sizeof(T), static_cast<double>(n) * (sizeof(T) + sizeof(Acc)),
                            static_cast<double>(n) * 60.0};
    concurrency::ThreadPool::TryParallelFor(
        ctx->GetOperatorThreadPool(), static_cast<std::ptrdiff_t>(outer * inner), cost,
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          std::vector<Acc> u(static_cast<size_t>(n));
          std::vector<Acc> eps(static_cast<size_t>(n));
          for (std::ptrdiff_t v = first; v < last; ++v) {
            const int64_t base = (v / inner) * n * inner + (v % inner);

            Acc m = ToAcc(x_data[base]);
            for (int64_t i = 1; i < n; ++i) {
              const Acc xi = ToAcc(x_data[base + i * inner]);
              m = (xi > m || std::isnan(xi)) ? xi : m;
            }

            Dot2State<Acc> sum;
            Acc eps_sum = 0;
            for (int64_t i = 0; i < n; ++i) {
              const Acc xi = ToAcc(x_data[base + i * inner]);
              const Acc nm = -m;
              const Acc t = xi + nm;  // TwoSum (Knuth)
              const Acc bb = t - xi;
              Acc dt = (xi - (t - bb)) + (nm - bb);
              if (!std::isfinite(t)) dt = 0;
              const Acc ui = std::exp(t);
              const Acc ei = ui * dt;
              u[static_cast<size_t>(i)] = ui;
              eps[static_cast<size_t>(i)] = ei;
              sum.Add(ui, 0);
              eps_sum += ei;
            }
            const Acc s = sum.s;
            const Acc ds = sum.c + eps_sum;

            for (int64_t i = 0; i < n; ++i) {
              const Acc ui = u[static_cast<size_t>(i)];
              const Acc yi = ui / s;
              const Acc r = std::fma(-yi, s, ui) / s;
              const Acc err = r + (eps[static_cast<size_t>(i)] - yi * ds) / s;
              const T y = FromAcc<T>(yi);
              y_data[base + i * inner] = y;
              const Acc ei = FinishError<Acc>(yi, err, ToAcc(y));
              e_data[base + i * inner] = ei;
              b_data[base + i * inner] = std::isnan(ei) ? ei : (rho * (yi + yi) + 4 * rho * std::fabs(ei));
            }
          }
        });

    return Status::OK();
  }

 private:
  int64_t axis_;
};

#define REGISTER_KERNEL_TYPED(T)                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
      SoftmaxWithError,                                                      \
      kMSDomain,                                                             \
      1,                                                                     \
      T,                                                                     \
      kCpuExecutionProvider,                                                 \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())             \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<ErrorType<T>>()), \
      SoftmaxWithError<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

}  // namespace contrib
}  // namespace onnxruntime
