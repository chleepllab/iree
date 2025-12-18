// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/general_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/conv_2d_nchw_fchw_riscv_64_internal.h"

void iree_uk_conv_tile_generic_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, const void* IREE_UK_RESTRICT kernel_ptr,
    iree_uk_index_t in_size0, iree_uk_index_t in_size1,
    iree_uk_index_t kernel_size0, iree_uk_index_t kernel_size1,
    iree_uk_index_t outer_size0, iree_uk_index_t outer_size1,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,
    iree_uk_index_t elem_size) {
  /*const char* IREE_UK_RESTRICT in_ptr_l1 = in_tile_ptr;
  char* IREE_UK_RESTRICT out_ptr_l1 = out_tile_ptr;
  for (iree_uk_index_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
    const char* IREE_UK_RESTRICT in_ptr = in_ptr_l1;
    char* IREE_UK_RESTRICT out_ptr = out_ptr_l1;
    for (iree_uk_index_t tile_i0 = 0; tile_i0 < tile_size0; ++tile_i0) {
      iree_uk_memcpy_riscv_64(out_ptr, in_ptr, tile_size1 * elem_size, elem_size);
      out_ptr += tile_size1 * elem_size;
      in_ptr += in_stride0 * elem_size;
    }
    out_ptr_l1 += out_stride1 * elem_size;
    in_ptr_l1 += tile_size1 * elem_size;
  }*/
}
