// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/common/common.h"
#include <gsl/gsl>

namespace onnxruntime {
namespace contrib {
namespace with_error_cpu {

// numpy-style (multidirectional) broadcast of two shapes: the output dims plus, for each input, its element strides
// in the output's index space (0 along broadcast dims). Host-only; shared by the CPU and CUDA elementwise kernels.
struct BroadcastLayout {
  std::vector<int64_t> out_dims;
  std::vector<int64_t> a_strides;
  std::vector<int64_t> b_strides;
};

inline Status ComputeBroadcastLayout(gsl::span<const int64_t> a_dims, gsl::span<const int64_t> b_dims,
                                     BroadcastLayout& layout) {
  const size_t rank = std::max(a_dims.size(), b_dims.size());
  layout.out_dims.assign(rank, 1);
  layout.a_strides.assign(rank, 0);
  layout.b_strides.assign(rank, 0);

  int64_t a_stride = 1;
  int64_t b_stride = 1;
  for (size_t i = 0; i < rank; ++i) {
    const size_t pos = rank - 1 - i;  // walk from the last dim
    const int64_t ad = i < a_dims.size() ? a_dims[a_dims.size() - 1 - i] : 1;
    const int64_t bd = i < b_dims.size() ? b_dims[b_dims.size() - 1 - i] : 1;
    ORT_RETURN_IF(ad != bd && ad != 1 && bd != 1, "Incompatible dimensions for broadcasting: ", ad, " vs ", bd);
    layout.out_dims[pos] = (ad == 1) ? bd : ad;
    layout.a_strides[pos] = (ad == 1) ? 0 : a_stride;
    layout.b_strides[pos] = (bd == 1) ? 0 : b_stride;
    a_stride *= ad;
    b_stride *= bd;
  }
  return Status::OK();
}

}  // namespace with_error_cpu
}  // namespace contrib
}  // namespace onnxruntime
