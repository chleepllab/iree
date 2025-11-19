// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/unpack_riscv_64_internal.h"

iree_uk_unpack_tile_func_t iree_uk_unpack_select_tile_func_arch(
    const iree_uk_unpack_params_t* params) {
  // For now, RISC-V implementation returns null, falling back to generic
  // implementation. This can be extended with RISC-V specific optimizations
  // using RISC-V vector extensions (RVV) or other RISC-V specific features.
  iree_uk_unpack_type_t unpack_type = iree_uk_unpack_type(params->flags);
  int esize = iree_uk_type_size(iree_uk_unpack_out_type(unpack_type));
  bool transpose = params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER;
  // Unpack is currently only used in practice with esize==4 and non-transpose.
  /*if (esize != 4 || transpose) return 0;
  if (params->in_size2 == 8 && params->in_size3 == 8) {
    return iree_uk_unpack_tile_8x8_x32_riscv_64_direct;
  }*/
  if (esize == 4 && params->in_size2 == 8 && params->in_size3 == 8) {
    return transpose ? 0 : iree_uk_unpack_tile_8x8_x32_riscv_64_direct;
  } else if (esize == 4 && params->in_size2 == 8 && params->in_size3 == 1) {
    return transpose ? iree_uk_unpack_tile_8x1_x32_riscv_64_transpose
                     : iree_uk_unpack_tile_8x1_x32_riscv_64_direct;
    //return transpose ? 0 : iree_uk_unpack_tile_8x1_x32_riscv_64_direct;
    //return transpose ? iree_uk_unpack_tile_8x1_x32_riscv_64_transpose : 0;
  } else if (esize == 1 && params->in_size2 == 8 && params->in_size3 == 1) {
    return transpose ? iree_uk_unpack_tile_8x1_x8_riscv_64_transpose
                     : iree_uk_unpack_tile_8x1_x8_riscv_64_direct;
    //return transpose ? 0 : iree_uk_unpack_tile_8x1_x8_riscv_64_direct;
  } else if (esize == 1 && params->in_size2 == 8 && params->in_size3 == 4) {
    return transpose ? iree_uk_unpack_tile_8x4_x8_riscv_64_transpose
                     : iree_uk_unpack_tile_8x4_x8_riscv_64_direct;
    //return transpose ? 0 : iree_uk_unpack_tile_8x4_x8_riscv_64_direct;
    //return transpose ? iree_uk_unpack_tile_8x4_x8_riscv_64_transpose : 0;
  } else if (esize == 1 && params->in_size2 == 8 && params->in_size3 == 8) {
    return transpose ? iree_uk_unpack_tile_8x8_x8_riscv_64_transpose
                     : iree_uk_unpack_tile_8x8_x8_riscv_64_direct;
    //return transpose ? 0 : iree_uk_unpack_tile_8x8_x8_riscv_64_direct;
    //return transpose ? iree_uk_unpack_tile_8x8_x8_riscv_64_transpose : 0;
  }
  if (esize == 4 || esize == 1) {
    return transpose ? iree_uk_unpack_tile_generic_riscv_64_transpose
                     : iree_uk_unpack_tile_generic_riscv_64_direct;
    //return transpose ? 0 : iree_uk_unpack_tile_generic_riscv_64_direct;
    //return transpose ? iree_uk_unpack_tile_generic_riscv_64_transpose : 0;
  }
  return 0;
} 
