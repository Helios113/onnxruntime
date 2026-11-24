// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/with_error_cpu_common.h"

#include <vector>

#include "contrib_ops/cpu/with_error_broadcast.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

using namespace with_error_cpu;

// Add / Sub / Mul / Div (numpy broadcasting) and Sqrt, each returning its result C (Y for Sqrt) plus E, the rounding
// error of C: exact(op) ~= C + E. IEEE +, -, *, /, sqrt are correctly rounded, so the result is computed here (it is
// bit-identical to the stock kernels, and float16 / bfloat16 are computed in fp32 and rounded once, which is exact
// rounding for these operations). The error comes from an error-free transformation in the accumulation type
// (double for double, float otherwise):
//   Add / Sub: TwoSum (Knuth)                        exact: r + err
//   Mul:       err = fma(a, b, -r)                   exact: r + err
//   Div:       err = fma(-r, b, a) / b               exact: r + err  (the division of the exact remainder is rounded)
//   Sqrt:      err = fma(-r, r, x) / (2 r)           exact: r + err  (first order)
// E = (r - C) + err, where r - C is the rounding to fp16 / bf16 when T is one of them. E is NaN where C is not finite.
// Same semantics as the CUDA kernels.
struct AddOp {
  static constexpr bool kUnary = false;
  template <typename Acc>
  static void Apply(Acc a, Acc b, Acc& r, Acc& err) {
    r = a + b;
    const Acc bb = r - a;
    err = (a - (r - bb)) + (b - bb);
  }
};

struct SubOp {
  static constexpr bool kUnary = false;
  template <typename Acc>
  static void Apply(Acc a, Acc b, Acc& r, Acc& err) {
    AddOp::Apply<Acc>(a, -b, r, err);
  }
};

struct MulOp {
  static constexpr bool kUnary = false;
  template <typename Acc>
  static void Apply(Acc a, Acc b, Acc& r, Acc& err) {
    r = a * b;
    err = std::fma(a, b, -r);
  }
};

struct DivOp {
  static constexpr bool kUnary = false;
  template <typename Acc>
  static void Apply(Acc a, Acc b, Acc& r, Acc& err) {
    r = a / b;
    err = std::fma(-r, b, a) / b;
  }
};

struct SqrtOp {
  static constexpr bool kUnary = true;
  template <typename Acc>
  static void Apply(Acc a, Acc /*unused*/, Acc& r, Acc& err) {
    r = std::sqrt(a);
    err = r == Acc(0) ? Acc(0) : std::fma(-r, r, a) / (r + r);
  }
};

template <typename T, typename Op>
class ElementwiseWithError final : public OpKernel {
 public:
  explicit ElementwiseWithError(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    using Acc = ErrorType<T>;

    const Tensor* A = ctx->Input<Tensor>(0);
    const Tensor* B = Op::kUnary ? nullptr : ctx->Input<Tensor>(1);

    BroadcastLayout layout;
    if (Op::kUnary) {
      layout.out_dims.assign(A->Shape().GetDims().begin(), A->Shape().GetDims().end());
      layout.a_strides.assign(layout.out_dims.size(), 0);
      layout.b_strides.assign(layout.out_dims.size(), 0);
      int64_t stride = 1;
      for (size_t a = layout.out_dims.size(); a-- > 0;) {
        layout.a_strides[a] = stride;
        stride *= layout.out_dims[a];
      }
    } else {
      ORT_RETURN_IF_ERROR(ComputeBroadcastLayout(A->Shape().GetDims(), B->Shape().GetDims(), layout));
    }

    const TensorShape out_shape(layout.out_dims);
    Tensor* C = ctx->Output(0, out_shape);
    Tensor* E = ctx->Output(1, out_shape);
    const int64_t total = out_shape.Size();
    if (total == 0) return Status::OK();

    const T* a_data = A->Data<T>();
    const T* b_data = Op::kUnary ? nullptr : B->Data<T>();
    T* c_data = C->MutableData<T>();
    Acc* e_data = E->MutableData<Acc>();
    const int rank = static_cast<int>(layout.out_dims.size());

    const TensorOpCost cost{2.0 * sizeof(T), static_cast<double>(sizeof(T) + sizeof(Acc)), 30.0};
    concurrency::ThreadPool::TryParallelFor(
        ctx->GetOperatorThreadPool(), static_cast<std::ptrdiff_t>(total), cost,
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          for (std::ptrdiff_t idx = first; idx < last; ++idx) {
            int64_t off_a = 0;
            int64_t off_b = 0;
            int64_t rem = idx;
            for (int a = rank - 1; a >= 0; --a) {
              const auto ua = static_cast<size_t>(a);
              const int64_t coord = rem % layout.out_dims[ua];
              rem /= layout.out_dims[ua];
              off_a += coord * layout.a_strides[ua];
              off_b += coord * layout.b_strides[ua];
            }
            const Acc a = ToAcc(a_data[off_a]);
            const Acc b = Op::kUnary ? Acc(0) : ToAcc(b_data[off_b]);
            Acc r;
            Acc err;
            Op::template Apply<Acc>(a, b, r, err);
            const T c = FromAcc<T>(r);
            c_data[idx] = c;
            e_data[idx] = FinishError<Acc>(r, err, ToAcc(c));
          }
        });

    return Status::OK();
  }
};

#define REGISTER_KERNEL_TYPED(OP_NAME, OP, T)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
      OP_NAME,                                                               \
      kMSDomain,                                                             \
      1,                                                                     \
      T,                                                                     \
      kCpuExecutionProvider,                                                 \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())             \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<ErrorType<T>>()), \
      ElementwiseWithError<T, OP>);

#define REGISTER_OP(OP_NAME, OP)                \
  REGISTER_KERNEL_TYPED(OP_NAME, OP, float)     \
  REGISTER_KERNEL_TYPED(OP_NAME, OP, double)    \
  REGISTER_KERNEL_TYPED(OP_NAME, OP, MLFloat16) \
  REGISTER_KERNEL_TYPED(OP_NAME, OP, BFloat16)

REGISTER_OP(AddWithError, AddOp)
REGISTER_OP(SubWithError, SubOp)
REGISTER_OP(MulWithError, MulOp)
REGISTER_OP(DivWithError, DivOp)
REGISTER_OP(SqrtWithError, SqrtOp)

}  // namespace contrib
}  // namespace onnxruntime
