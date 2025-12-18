// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/conv_2d_nchw_fchw_internal.h"

static void iree_uk_conv_tile_generic_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, const void* IREE_UK_RESTRICT kernel_tile_ptr,
    iree_uk_index_t in_size0, iree_uk_index_t in_size1,
    iree_uk_index_t kernel_size0, iree_uk_index_t kernel_size1,
    iree_uk_index_t outer_size0, iree_uk_index_t outer_size1,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,
    iree_uk_index_t elem_size) {
  float* out_ptr_l1 = out_tile_ptr;
  const float* in_ptr_l1 = in_tile_ptr;
  const float* kernel_ptr = kernel_tile_ptr;
  for (iree_uk_index_t i0 = 0; i0 < in_size0 - kernel_size0 + 1; ++i0) {
    for (iree_uk_index_t i1 = 0; i1 < in_size1 - kernel_size1 + 1; ++i1) {
      const float* IREE_UK_RESTRICT in_ptr = in_ptr_l1;
      float* IREE_UK_RESTRICT out_ptr = out_ptr_l1;
      iree_uk_index_t out_idx = i1;
      for (iree_uk_index_t kh = 0; kh < kernel_size0; ++kh) {
        for (iree_uk_index_t kw = 0; kw < kernel_size1; ++kw) {
          iree_uk_index_t lhs_idx = i1 + kh * kernel_size0 + kw;
          iree_uk_index_t rhs_idx = kh * kernel_size0 + kw;
          out_ptr[out_idx] += in_ptr[lhs_idx] * kernel_ptr[rhs_idx];
        }
      }
      out_ptr_l1 += outer_size1;
      in_ptr_l1 += in_size1;
    }
  }
}

static iree_uk_conv_tile_func_t iree_uk_conv_2d_nchw_fchw_select_tile_func_generic(
    const iree_uk_conv_params_t* params) {
  return iree_uk_conv_tile_generic_direct;
}

iree_uk_conv_tile_func_t iree_uk_conv_2d_nchw_fchw_select_tile_func(
    const iree_uk_conv_params_t* params) {
  iree_uk_conv_tile_func_t arch_tile_func =
      iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(params);
  if (arch_tile_func) return arch_tile_func;
  return iree_uk_conv_2d_nchw_fchw_select_tile_func_generic(params);
}
