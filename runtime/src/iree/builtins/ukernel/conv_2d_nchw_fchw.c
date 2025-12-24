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
  
  IREE_UK_ASSERT(params->out_size_n > 0);
  IREE_UK_ASSERT(params->in_size_c > 0);
  IREE_UK_ASSERT(params->out_size_c > 0);
  IREE_UK_ASSERT(params->in_size_h > 0);
  IREE_UK_ASSERT(params->in_size_w > 0);
  IREE_UK_ASSERT(params->filter_size_h > 0);
  IREE_UK_ASSERT(params->filter_size_w > 0);
  IREE_UK_ASSERT(params->out_size_h > 0);
  IREE_UK_ASSERT(params->out_size_w > 0);
  
  iree_uk_index_t expected_output_height = 
      params->in_size_h - params->filter_size_h + 1;
  iree_uk_index_t expected_output_width = 
      params->in_size_w - params->filter_size_w + 1;
  
  IREE_UK_ASSERT(params->out_size_h == expected_output_height);
  IREE_UK_ASSERT(params->out_size_w == expected_output_width);
#endif  // IREE_UK_ENABLE_ASSERTS
}

static bool iree_uk_conv_early(const iree_uk_conv_params_t* params) {
  return (params->out_size_n == 0 || params->out_size_c == 0 ||
          params->out_size_h == 0 || params->out_size_w == 0);
}

static void iree_conv_2d_nchw_fchw_using_tile_func_generic(
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

static void iree_uk_conv_using_tile_func(const iree_uk_conv_params_t* params,
                                          iree_uk_conv_2d_nchw_fchw_tile_func_t tile_func) {
  iree_uk_conv_type_t conv_type = iree_uk_conv_type(params->flags);
  iree_uk_type_t in_type = iree_uk_conv_in_type(conv_type);
  iree_uk_type_t filter_type = iree_uk_conv_filter_type(conv_type);
  iree_uk_type_t out_type = iree_uk_conv_out_type(conv_type);

  iree_uk_index_t in_type_size = iree_uk_type_size(in_type);
  iree_uk_index_t filter_type_size = iree_uk_type_size(filter_type);
  iree_uk_index_t out_type_size = iree_uk_type_size(out_type);
  for (iree_uk_index_t n = 0; n < params->out_size_n; n++) {
    for (iree_uk_index_t oc = 0; oc < params->out_size_c; oc++) {
      for (iree_uk_index_t oh = 0; oh < params->out_size_h; oh++) {
        for (iree_uk_index_t ow = 0; ow < params->out_size_w; ow++) {
          iree_uk_index_t out_idx = n * params->out_stride0 +
                                    oc * params->out_stride1 +
                                    oh * params->out_size_w +
                                    ow;
          char* out_ptr = (char*)params->out_buffer + out_idx * out_type_size;
          tile_func(out_ptr, params->in_buffer, params->filter_buffer,
                    params->in_size_c, params->out_size_c,
                    n, oc,
                    oh, ow,
		    params->in_size_h, params->in_size_w,
                    params->filter_size_h, params->filter_size_w,
                    params->out_size_h, params->out_size_w,
                    params->tile_size0, params->tile_size1,
                    params->in_stride0, params->in_stride1,
                    params->filter_stride0, params->filter_stride1,
                    params->out_stride0, params->out_stride1,
                    in_type_size, filter_type_size, out_type_size);
          /*float* out_ptr = (float*)((char*)params->out_buffer + out_idx * out_type_size);
	  float sum = 0.0f;
          for (iree_uk_index_t ic = 0; ic < params->in_size_c; ic++) {
            for (iree_uk_index_t kh = 0; kh < params->filter_size_h; kh++) {
              for (iree_uk_index_t kw = 0; kw < params->filter_size_w; kw++) {
                iree_uk_index_t ih = oh + kh;
                iree_uk_index_t iw = ow + kw;
                if (ih >= 0 && ih < params->in_size_h &&
                    iw >= 0 && iw < params->in_size_w) {
                  iree_uk_index_t in_idx = n * params->in_stride0 +
                                          ic * params->in_stride1 +
                                          ih * params->in_size_w +
                                          iw;
                  iree_uk_index_t filter_idx = oc * params->filter_stride0 +
                                              ic * params->filter_stride1 +
                                              kh * params->filter_size_w +
                                              kw;
                  float* in_ptr = (float*)((char*)params->in_buffer + in_idx * in_type_size);
                  float* filter_ptr = (float*)((char*)params->filter_buffer + filter_idx * filter_type_size);
                  sum += (*in_ptr) * (*filter_ptr);
                }
              }
            }
          }
          *out_ptr = sum;*/
        }
      }
    }
  }
}

void iree_uk_conv_p(const iree_uk_conv_params_t* params) {
  iree_uk_conv_validate(params);

  // Maybe handle this conv "early"
  if (iree_uk_conv_early(params)) return;

  iree_uk_conv_2d_nchw_fchw_tile_func_t tile_func = iree_uk_conv_2d_nchw_fchw_select_tile_func(params);
  iree_uk_conv_using_tile_func(params, tile_func);
  //if (1) {
  //  ((float*)(params->out_buffer))[0] = 100000.0;
  //}
  //((float*)(params->out_buffer))[0] = 200000.0;
}

iree_uk_uint32_t iree_uk_conv_2d_nchw_fchw_info_p(const iree_uk_conv_params_t* params) {
  iree_uk_uint32_t result = 0;
  if (iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(params)) {
    result |= IREE_UK_FLAG_CONV_INFO_HAVE_ARCHITECTURE_SPECIFIC_TILE_FUNCTION;
  }
  return result;
}

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
    iree_uk_uint32_t flags, const iree_uk_uint64_t* cpu_data) {
  iree_uk_conv_params_t params = {
      .in_buffer = in_buffer,
      .in_offset = in_offset,
      .in_stride0 = in_stride0,
      .in_stride1 = in_stride1,
      .filter_buffer = filter_buffer,
      .filter_offset = filter_offset,
      .filter_stride0 = filter_stride0,
      .filter_stride1 = filter_stride1,
      .out_buffer = out_buffer,
      .out_offset = out_offset,
      .out_stride0 = out_stride0,
      .out_stride1 = out_stride1,
      .out_size_n = out_size_n,
      .in_size_c = in_size_c,
      .out_size_c = out_size_c,
      .in_size_h = in_size_h,
      .in_size_w = in_size_w,
      .filter_size_h = filter_size_h,
      .filter_size_w = filter_size_w,
      .out_size_h = out_size_h,
      .out_size_w = out_size_w,
      .tile_size0 = tile_size0,
      .tile_size1 = tile_size1,
      .flags = flags,
      .cpu_data = cpu_data
  };
  
  iree_uk_conv_p(&params);
}

IREE_UK_EXPORT iree_uk_uint32_t
iree_uk_conv_2d_nchw_fchw_info(iree_uk_int32_t tile_size0, iree_uk_int32_t tile_size1,
                  iree_uk_uint32_t flags, const iree_uk_uint64_t* cpu_data) {
  iree_uk_conv_params_t params = {
      .filter_size_h = tile_size0,
      .filter_size_w = tile_size1,
      .tile_size0 = tile_size0,
      .tile_size1 = tile_size1,
      .flags = flags,
      .cpu_data = cpu_data
  };
  return iree_uk_conv_2d_nchw_fchw_info_p(&params);
}
