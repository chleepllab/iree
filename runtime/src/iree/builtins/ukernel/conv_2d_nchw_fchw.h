// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_CONV_H_
#define IREE_BUILTINS_UKERNEL_CONV_H_

#include "iree/builtins/ukernel/common.h"

// `conv` microkernel for 2D convolution with NCHW input and FCHW filter layout.
IREE_UK_EXPORT void iree_uk_conv_2d_nchw_fchw(
    const void* lhs_buffer, iree_uk_index_t lhs_offset,
    const void* rhs_buffer, iree_uk_index_t rhs_offset,
    void* out_buffer, iree_uk_index_t out_offset,
    iree_uk_index_t batch_count, iree_uk_index_t input_channel_count,
    iree_uk_index_t output_channel_count,
    iree_uk_index_t input_height, iree_uk_index_t input_width,
    iree_uk_index_t kernel_height, iree_uk_index_t kernel_width,
    iree_uk_index_t output_height, iree_uk_index_t output_width,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,
    iree_uk_uint32_t flags, const iree_uk_uint64_t* cpu_data);

#endif  // IREE_BUILTINS_UKERNEL_CONV_H_
