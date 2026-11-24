// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/with_error_cpu_common.h"

#include <memory>

#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/math/gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"

namespace onnxruntime {
namespace contrib {

using namespace with_error_cpu;

// Y = alpha * A' * B' + beta * C, plus E, the rounding error of Y: exact result ~= Y + E.
// float / double: Y comes from the stock CPU Gemm kernel (held and invoked, so Y is bit-identical to it).
// float16 / bfloat16: there is no reliable stock CPU kernel, so Y is computed here: the products are summed in k order
// in fp32, scaled by alpha, added to beta * C (each step rounded to fp32), and rounded once to T.
// E is (hi - Y) + corr from a compensated evaluation of alpha * (A' * B') + beta * C (Dot2 over k in order, TwoProd for
// the alpha / beta scalings, TwoSum for the final addition). Same semantics as the CUDA kernel.
template <typename T>
class GemmWithError final : public OpKernel {
 public:
  explicit GemmWithError(const OpKernelInfo& info)
      : OpKernel(info),
        trans_a_{info.GetAttrOrDefault<int64_t>("transA", 0) != 0},
        trans_b_{info.GetAttrOrDefault<int64_t>("transB", 0) != 0},
        alpha_{info.GetAttrOrDefault<float>("alpha", 1.0f)},
        beta_{info.GetAttrOrDefault<float>("beta", 1.0f)} {
    if constexpr (kHasStockCpuKernel<T>) {
      gemm_ = std::make_unique<Gemm<T>>(info);
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  std::unique_ptr<OpKernel> gemm_;  // stock kernel, float / double only
  const bool trans_a_;
  const bool trans_b_;
  const float alpha_;
  const float beta_;
};

#define REGISTER_KERNEL_TYPED(T)                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                  \
      GemmWithError,                                                              \
      kMSDomain,                                                                  \
      1,                                                                          \
      T,                                                                          \
      kCpuExecutionProvider,                                                      \
      KernelDefBuilder()                                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                  \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<ErrorType<T>>()),      \
      GemmWithError<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status GemmWithError<T>::Compute(OpKernelContext* ctx) const {
  using Acc = ErrorType<T>;

  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const Tensor* C = ctx->Input<Tensor>(2);  // optional

  GemmHelper helper(A->Shape(), trans_a_, B->Shape(), trans_b_, C != nullptr ? C->Shape() : TensorShape({}));
  ORT_RETURN_IF_ERROR(helper.State());
  const int64_t M = helper.M();
  const int64_t N = helper.N();
  const int64_t K = helper.K();

  // 1. Y with the stock kernel (also allocates output 0), or allocate it for the types without one.
  constexpr bool stock = kHasStockCpuKernel<T>;
  if constexpr (stock) {
    ORT_RETURN_IF_ERROR(gemm_->Compute(ctx));
  } else {
    ctx->Output(0, {M, N});
  }
  Tensor* Y = ctx->Output<Tensor>(0);
  Tensor* E = ctx->Output(1, {M, N});
  const int64_t total = M * N;
  if (total == 0) return Status::OK();

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

  const T* a_data = A->Data<T>();
  const T* b_data = B->Data<T>();
  const T* c_data = use_c ? C->Data<T>() : nullptr;
  T* y_data = Y->MutableData<T>();
  Acc* e_data = E->MutableData<Acc>();
  const bool trans_a = trans_a_;
  const bool trans_b = trans_b_;
  const Acc alpha = EffectiveScalar<T>(alpha_);
  const Acc beta = EffectiveScalar<T>(beta_);

  const TensorOpCost cost{static_cast<double>(K) * 2.0 * sizeof(T), static_cast<double>(sizeof(Acc)),
                          static_cast<double>(K) * 20.0};
  concurrency::ThreadPool::TryParallelFor(
      ctx->GetOperatorThreadPool(), static_cast<std::ptrdiff_t>(total), cost,
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        for (std::ptrdiff_t idx = first; idx < last; ++idx) {
          const int64_t j = idx % N;
          const int64_t i = idx / N;

          Dot2State<Acc> d;
          Acc plain = 0;  // k-ordered plain sum, only used to produce Y for the types without a stock kernel
          for (int64_t k = 0; k < K; ++k) {
            const Acc a = ToAcc(a_data[trans_a ? k * M + i : i * K + k]);
            const Acc b = ToAcc(b_data[trans_b ? j * K + k : k * N + j]);
            d.AddProduct(a, b);
            if constexpr (!stock) plain = plain + a * b;
          }

          // alpha * (s + c): TwoProd(alpha, s) is exact; alpha * c is already a second-order term.
          const Acc p1 = alpha * d.s;
          const Acc e1 = std::fma(alpha, d.s, -p1);
          Acc hi = p1;
          Acc corr = e1 + alpha * d.c;
          Acc cv = 0;

          if (use_c) {
            cv = ToAcc(c_data[(c_rows == 1 ? 0 : i) * c_cols + (c_cols == 1 ? 0 : j)]);
            const Acc p2 = beta * cv;
            const Acc e2 = std::fma(beta, cv, -p2);
            // TwoSum(p1, p2)
            hi = p1 + p2;
            const Acc bb = hi - p1;
            const Acc q = (p1 - (hi - bb)) + (p2 - bb);
            corr = corr + (q + e2);
          }

          if constexpr (!stock) {
            Acc y = alpha * plain;
            if (use_c) y = y + beta * cv;
            y_data[idx] = FromAcc<T>(y);
          }
          e_data[idx] = FinishError<Acc>(hi, corr, ToAcc(y_data[idx]));
        }
      });

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
