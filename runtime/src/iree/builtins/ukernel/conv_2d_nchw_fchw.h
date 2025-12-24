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
    const void* in_buffer, iree_uk_index_t in_offset,
    iree_uk_index_t in_stride0, iree_uk_index_t in_stride1,
    const void* filter_buffer, iree_uk_index_t filter_offset,
    iree_uk_index_t filter_stride0, iree_uk_index_t filter_stride1,
    void* out_buffer, iree_uk_index_t out_offset,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    iree_uk_index_t out_size_n,
    iree_uk_index_t in_size_c, iree_uk_index_t out_size_c,
    iree_uk_index_t in_size_h, iree_uk_index_t in_size_w,
    iree_uk_index_t filter_size_h, iree_uk_index_t filter_size_w,
    iree_uk_index_t out_size_h, iree_uk_index_t out_size_w,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,
    iree_uk_uint32_t flags, const iree_uk_uint64_t* cpu_data);

#endif  // IREE_BUILTINS_UKERNEL_CONV_H_
