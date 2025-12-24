// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/conv_2d_nchw_fchw_x86_64_internal.h"

iree_uk_conv_2d_nchw_fchw_tile_func_t iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(
    const iree_uk_conv_params_t* params) {
  return 0;
}
