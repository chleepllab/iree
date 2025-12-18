// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/conv_2d_nchw_fchw_riscv_64_internal.h"

iree_uk_conv_tile_func_t iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(
    const iree_uk_conv_params_t* params) {
  // For now, RISC-V implementation returns null, falling back to generic
  // implementation. This can be extended with RISC-V specific optimizations
  // using RISC-V vector extensions (RVV) or other RISC-V specific features.
  //iree_uk_conv_type_t conv_type = iree_uk_conv_type(params->flags);
  //int esize = iree_uk_type_size(iree_uk_conv_out_type(pack_type));
  //bool transpose = params->flags & IREE_UK_FLAG_CONV_TRANSPOSE_INNER;
  return 0;
}
