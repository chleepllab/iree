// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/conv_2d_nchw_fchw.h"

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/conv_2d_nchw_fchw_internal.h"

static void iree_uk_conv_validate(const iree_uk_conv_params_t* params) {
#ifdef IREE_UK_ENABLE_ASSERTS
  const iree_uk_uint32_t allflags =
      IREE_UK_FLAG_CONV_TYPE_MASK | IREE_UK_FLAG_CONV_ACCUMULATE |
      IREE_UK_FLAG_CONV_SKIP_INTERMEDIATE_ROUNDINGS |
      IREE_UK_FLAG_CONV_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION;
  IREE_UK_ASSERT(!(params->flags & ~allflags));
  //iree_uk_uint32_t flags_type = params->flags & IREE_UK_FLAG_CONV_TYPE_MASK;
  //IREE_UK_ASSERT(flags_type < IREE_UK_FLAG_CONV_TYPE_END);
  
  IREE_UK_ASSERT(params->batch_count > 0);
  IREE_UK_ASSERT(params->input_channel_count > 0);
  IREE_UK_ASSERT(params->output_channel_count > 0);
  IREE_UK_ASSERT(params->input_height > 0);
  IREE_UK_ASSERT(params->input_width > 0);
  IREE_UK_ASSERT(params->kernel_height > 0);
  IREE_UK_ASSERT(params->kernel_width > 0);
  IREE_UK_ASSERT(params->output_height > 0);
  IREE_UK_ASSERT(params->output_width > 0);
  
  iree_uk_index_t expected_output_height = 
      params->input_height - params->kernel_height + 1;
  iree_uk_index_t expected_output_width = 
      params->input_width - params->kernel_width + 1;
  
  IREE_UK_ASSERT(params->output_height == expected_output_height);
  IREE_UK_ASSERT(params->output_width == expected_output_width);
#endif  // IREE_UK_ENABLE_ASSERTS
}

static bool iree_uk_conv_early(const iree_uk_conv_params_t* params) {
  return (params->batch_count == 0 || params->output_channel_count == 0 ||
          params->output_height == 0 || params->output_width == 0);
}

static void iree_uk_conv_using_tile_func(const iree_uk_conv_params_t* params,
                                          iree_uk_conv_tile_func_t tile_func) {
  //iree_uk_index_t batch = 1;
  //iree_uk_index_t input_channel = 1;
  //iree_uk_index_t output_channel = 1;
  /*iree_uk_index_t input_height = params->input_height;
  iree_uk_index_t input_width = params->input_width;
  iree_uk_index_t kernel_height = params->kernel_height;
  iree_uk_index_t kernel_width = params->kernel_width;
  iree_uk_index_t output_height = params->output_height;
  iree_uk_index_t output_width = params->output_width;
  iree_uk_index_t tile_size0 = params->tile_size0;
  iree_uk_index_t tile_size1 = params->tile_size0;

  iree_uk_conv_type_t conv_type = iree_uk_conv_type(params->flags);
  iree_uk_type_t lhs_type = iree_uk_conv_lhs_type(conv_type);
  iree_uk_index_t elem_size = iree_uk_type_size(lhs_type);
  
  const char* in_buf =
      (const char*)params->lhs_buffer + (params->lhs_offset * elem_size);
  const char* kernel_buf = (char*)params->rhs_buffer + (params->rhs_offset * elem_size);
  char* out_buf = (char*)params->out_buffer + (params->out_offset * elem_size);
  iree_uk_index_t lpad = tile_size0 / 2;
  iree_uk_index_t rpad = tile_size0 / 2;
  iree_uk_index_t out_stride0 = input_height - tile_size0 + 1;
  iree_uk_index_t out_stride1 = input_width - tile_size1 + 1;
  iree_uk_index_t dim0 = 0;
  for (iree_uk_index_t i0 = 0; i0 < input_height; i0 += tile_size0) {
    iree_uk_index_t i0_start = (i0 - lpad > 0 ? i0 - lpad : 0);
    iree_uk_index_t i0_end = (i0 + tile_size0 + rpad < input_height ? i0 + tile_size0 + rpad : input_height - 1);
    iree_uk_index_t i0_size = i0_end - i0_start;
    iree_uk_index_t dim1 = 0;
    for (iree_uk_index_t i1 = 0; i1 < input_width; i1 += tile_size1) {
      iree_uk_index_t i1_start = (i1 - lpad > 0 ? i1 - lpad : 0);
      iree_uk_index_t i1_end = (i1 + tile_size1 + rpad < input_width ? i1 + tile_size1 + rpad : input_width - 1);
      iree_uk_index_t i1_size = i1_end - i1_start;
      tile_func(out_buf + dim0 * tile_size0 + dim1, in_buf + i0 * tile_size1 + i1, kernel_buf, i0_size, i1_size,
		kernel_height, kernel_width, output_height, output_width, tile_size0, tile_size1, elem_size);
      dim1 += out_stride1;
    }
    dim0 += out_stride0;
  }*/
  /*for (iree_uk_int32_t n = 0; n < batch; ++n) {
    char* out_tile = out_tile_row;
    const char* rhs_panel = rhs_panel_start;
    for (iree_uk_int32_t oc = 0; oc < output_channel; +oc) {
      for (iree_uk_int32_t ic = 0; ic < input_channel; ++ic) {
        tile_func(out_tile, lhs_panel, rhs_panel, params);
        out_tile += out_tile_size;
        rhs_panel += rhs_panel_stride;
      }
    }
  }*/
}

void iree_uk_conv_p(const iree_uk_conv_params_t* params) {
  iree_uk_conv_validate(params);

  // Maybe handle this conv "early"
  if (iree_uk_conv_early(params)) return;

  // Select a target-specific tile_func
  iree_uk_conv_tile_func_t tile_func =
      iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(params);
  //iree_uk_conv_using_tile_func(params, tile_func);
  if (params->batch_count == 1) {
    ((float*)(params->out_buffer))[0] = 100000.0;
  }
}

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
    iree_uk_uint32_t flags, const iree_uk_uint64_t* cpu_data) {
  iree_uk_conv_params_t params = {
      .lhs_buffer = lhs_buffer,
      .lhs_offset = lhs_offset,
      .rhs_buffer = rhs_buffer,
      .rhs_offset = rhs_offset,
      .out_buffer = out_buffer,
      .out_offset = out_offset,
      .batch_count = batch_count,
      .input_channel_count = input_channel_count,
      .output_channel_count = output_channel_count,
      .input_height = input_height,
      .input_width = input_width,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .output_height = output_height,
      .output_width = output_width,
      .tile_size0 = tile_size0,
      .tile_size1 = tile_size1,
      .flags = flags,
      .cpu_data = cpu_data
  };
  
  iree_uk_conv_p(&params);
}
