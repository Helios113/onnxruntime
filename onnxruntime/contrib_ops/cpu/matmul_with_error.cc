// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The error term relies on the exact sequence of rounded operations below (TwoProd through std::fma, TwoSum, ...):
// the compiler must not contract a*b+c into an FMA or reassociate. -ffast-math is not used for this target; the
// pragmas additionally switch contraction off for this file.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

#include <cmath>
#include <limits>

#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/math/matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace contrib {

// Y = A @ B from the stock CPU MatMul kernel (held as a member and invoked, so Y is bit-identical to it), plus E,
// the rounding error of Y from a Dot2 pass over k = 0..K-1 in order, so that exact(A @ B) ~= Y + E.
// E is NaN where Y is not finite. Same semantics as the CUDA kernel, so E can be compared across providers.
template <typename T>
class MatMulWithError final : public OpKernel {
 public:
  explicit MatMulWithError(const OpKernelInfo& info)
      : OpKernel(info),
        matmul_(info),
        trans_a_{info.GetAttrOrDefault<int64_t>("transA", 0) != 0},
        trans_b_{info.GetAttrOrDefault<int64_t>("transB", 0) != 0} {}

  Status Compute(OpKernelContext* ctx) const override;

 private:
  MatMul<T> matmul_;
  const bool trans_a_;
  const bool trans_b_;
};

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMulWithError,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("E", DataTypeImpl::GetTensorType<T>()), \
      MatMulWithError<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

template <typename T>
Status MatMulWithError<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);

  const bool trans_a = A->Shape().NumDimensions() == 1 ? false : trans_a_;
  const bool trans_b = B->Shape().NumDimensions() == 1 ? false : trans_b_;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape(), trans_a, trans_b, false, false, /*fill_offsets*/ true));

  // 1. Y with the stock kernel (also allocates output 0). The stock CPU MatMul<double> is the generic Eigen kernel,
  // which does not implement transA / transB (only MatMul<float> does), so for transposed double inputs Y is
  // computed here with a plain k-ordered sum instead.
  const bool own_y = std::is_same_v<T, double> && (trans_a || trans_b);
  if (own_y) {
    ctx->Output(0, helper.OutputShape());
  } else {
    ORT_RETURN_IF_ERROR(matmul_.Compute(ctx));
  }
  Tensor* Y = ctx->Output<Tensor>(0);
  Tensor* E = ctx->Output(1, Y->Shape());

  const int64_t M = helper.M();
  const int64_t N = helper.N();
  const int64_t K = helper.K();
  const int64_t batch = static_cast<int64_t>(helper.OutputOffsets().size());
  const int64_t total = batch * M * N;
  if (total == 0) return Status::OK();

  const T* a_data = A->Data<T>();
  const T* b_data = B->Data<T>();
  T* y_mut = Y->MutableData<T>();
  T* e_data = E->MutableData<T>();
  const auto& left_offsets = helper.LeftOffsets();
  const auto& right_offsets = helper.RightOffsets();

  const TensorOpCost cost{static_cast<double>(K) * 2.0 * sizeof(T), static_cast<double>(sizeof(T)),
                          static_cast<double>(K) * 20.0};
  concurrency::ThreadPool::TryParallelFor(
      ctx->GetOperatorThreadPool(), static_cast<std::ptrdiff_t>(total), cost,
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        for (std::ptrdiff_t idx = first; idx < last; ++idx) {
          const int64_t j = idx % N;
          const int64_t i = (idx / N) % M;
          const int64_t b = idx / (N * M);
          const T* a_ptr = a_data + left_offsets[static_cast<size_t>(b)];
          const T* b_ptr = b_data + right_offsets[static_cast<size_t>(b)];

          T s = 0;
          T c = 0;
          for (int64_t k = 0; k < K; ++k) {
            const T a = a_ptr[trans_a ? k * M + i : i * K + k];
            const T bv = b_ptr[trans_b ? j * K + k : k * N + j];
            const T p = a * bv;
            const T e = std::fma(a, bv, -p);  // TwoProd: exact
            const T t = s + p;                // TwoSum (Knuth)
            const T bb = t - s;
            const T q = (s - (t - bb)) + (p - bb);
            s = t;
            c = c + (q + e);
          }
          if (own_y) {
            T acc = 0;
            for (int64_t k = 0; k < K; ++k) {
              const T prod = a_ptr[trans_a ? k * M + i : i * K + k] * b_ptr[trans_b ? j * K + k : k * N + j];
              acc = acc + prod;
            }
            y_mut[idx] = acc;
          }
          const T y = y_mut[idx];
          e_data[idx] = std::isfinite(y) ? ((s - y) + c) : std::numeric_limits<T>::quiet_NaN();
        }
      });

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
