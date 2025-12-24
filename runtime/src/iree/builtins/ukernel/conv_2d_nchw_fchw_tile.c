// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/conv_2d_nchw_fchw_internal.h"

static void iree_uk_conv_tile_generic_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, const void* IREE_UK_RESTRICT filter_tile_ptr,
    iree_uk_index_t in_size_c, iree_uk_index_t out_size_c,
    iree_uk_index_t n, iree_uk_index_t oc,
    iree_uk_index_t oh, iree_uk_index_t ow,
    iree_uk_index_t in_size_h, iree_uk_index_t in_size_w,
    iree_uk_index_t filter_size_h, iree_uk_index_t filter_size_w,
    iree_uk_index_t out_size_h, iree_uk_index_t out_size_w,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,
    iree_uk_index_t in_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t filter_stride0, iree_uk_index_t filter_stride1,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    iree_uk_index_t in_type_size, iree_uk_index_t filter_type_size,
    iree_uk_index_t out_type_size) {
  float* out_ptr = (float*)((char*)out_tile_ptr + 0);
  float sum = 0.0f;
  for (iree_uk_index_t ic = 0; ic < in_size_c; ic++) {
    for (iree_uk_index_t kh = 0; kh < filter_size_h; kh++) {
      for (iree_uk_index_t kw = 0; kw < filter_size_w; kw++) {
        iree_uk_index_t ih = oh + kh;
        iree_uk_index_t iw = ow + kw;
        if (ih >= 0 && ih < in_size_h &&
            iw >= 0 && iw < in_size_w) {
          iree_uk_index_t in_idx = n * in_stride0 +
                                  ic * in_stride1 +
                                  ih * in_size_w +
                                  iw;
          iree_uk_index_t filter_idx = oc * filter_stride0 +
                                      ic * filter_stride1 +
                                      kh * filter_size_w +
                                      kw;
          float* in_ptr = (float*)((char*)in_tile_ptr + in_idx * in_type_size);
          float* filter_ptr = (float*)((char*)filter_tile_ptr + filter_idx * filter_type_size);
          sum += (*in_ptr) * (*filter_ptr);
        }
      }
    }
  }
  *out_ptr = sum;
}

iree_uk_conv_2d_nchw_fchw_tile_func_t iree_uk_conv_2d_nchw_fchw_select_tile_func(
    const iree_uk_conv_params_t* params) {
  iree_uk_conv_2d_nchw_fchw_tile_func_t arch_tile_func =
      iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(params);
  if (arch_tile_func) return arch_tile_func;
  return iree_uk_conv_tile_generic_direct;
}
