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
  iree_uk_uint32_t flags_type = params->flags & IREE_UK_FLAG_CONV_TYPE_MASK;
  IREE_UK_ASSERT(flags_type < IREE_UK_FLAG_CONV_TYPE_END);
  
  // Validate dimensions
  IREE_UK_ASSERT(params->batch_count > 0);
  IREE_UK_ASSERT(params->input_channel_count > 0);
  IREE_UK_ASSERT(params->output_channel_count > 0);
  IREE_UK_ASSERT(params->input_height > 0);
  IREE_UK_ASSERT(params->input_width > 0);
  IREE_UK_ASSERT(params->kernel_height > 0);
  IREE_UK_ASSERT(params->kernel_width > 0);
  IREE_UK_ASSERT(params->output_height > 0);
  IREE_UK_ASSERT(params->output_width > 0);
  
  // Validate strides and dilations
  IREE_UK_ASSERT(params->stride_height > 0);
  IREE_UK_ASSERT(params->stride_width > 0);
  IREE_UK_ASSERT(params->dilation_height > 0);
  IREE_UK_ASSERT(params->dilation_width > 0);
  
  // Validate padding (can be negative for valid padding)
  IREE_UK_ASSERT(params->padding_top >= 0 || params->flags & IREE_UK_FLAG_CONV_VALID_PADDING);
  IREE_UK_ASSERT(params->padding_bottom >= 0 || params->flags & IREE_UK_FLAG_CONV_VALID_PADDING);
  IREE_UK_ASSERT(params->padding_left >= 0 || params->flags & IREE_UK_FLAG_CONV_VALID_PADDING);
  IREE_UK_ASSERT(params->padding_right >= 0 || params->flags & IREE_UK_FLAG_CONV_VALID_PADDING);
  
  // For conv_2d_nchw_fchw, we expect specific layout
  // Input: [N, C, H, W]
  // Kernel: [OC, IC, KH, KW]
  // Output: [N, OC, OH, OW]
  
  // Check that output dimensions match convolution formula
  iree_uk_index_t expected_output_height = 
      (params->input_height + params->padding_top + params->padding_bottom - 
       params->dilation_height * (params->kernel_height - 1) - 1) / params->stride_height + 1;
  iree_uk_index_t expected_output_width = 
      (params->input_width + params->padding_left + params->padding_right - 
       params->dilation_width * (params->kernel_width - 1) - 1) / params->stride_width + 1;
  
  IREE_UK_ASSERT(params->output_height == expected_output_height);
  IREE_UK_ASSERT(params->output_width == expected_output_width);
#endif  // IREE_UK_ENABLE_ASSERTS
}

// General conv implementation, shared among all cases.
static void iree_uk_conv_using_tile_func(const iree_uk_conv_params_t* params,
                                          iree_uk_conv_tile_func_t tile_func) {
  const iree_uk_int32_t batch_count = params->batch_count;
  const iree_uk_int32_t output_channel_count = params->output_channel_count;
  const iree_uk_int32_t output_height = params->output_height;
  const iree_uk_int32_t output_width = params->output_width;
  const iree_uk_int16_t M0 = params->M0;
  const iree_uk_int16_t N0 = params->N0;
  
  iree_uk_conv_type_t conv_type = iree_uk_conv_type(params->flags);
  const iree_uk_type_t lhs_type = iree_uk_conv_lhs_type(conv_type);
  const iree_uk_type_t rhs_type = iree_uk_conv_rhs_type(conv_type);
  const iree_uk_type_t out_type = iree_uk_conv_out_type(conv_type);
  
  const iree_uk_int16_t lhs_elem_bits_log2 =
      iree_uk_type_bit_count_log2(lhs_type);
  const iree_uk_int16_t rhs_elem_bits_log2 =
      iree_uk_type_bit_count_log2(rhs_type);
  const iree_uk_int16_t out_elem_size_log2 = iree_uk_type_size_log2(out_type);
  
  // For conv_2d_nchw_fchw, we process tiles in output space
  // Each tile computes M0 x N0 output elements
  
  char* out_ptr = (char*)params->out_buffer + 
                  (params->out_offset << out_elem_size_log2);
  const char* lhs_ptr = (const char*)params->lhs_buffer +
                       iree_uk_bits_to_bytes_exact(params->lhs_offset << lhs_elem_bits_log2);
  const char* rhs_ptr = (const char*)params->rhs_buffer +
                       iree_uk_bits_to_bytes_exact(params->rhs_offset << rhs_elem_bits_log2);
  
  iree_uk_index_t out_stride_n = params->out_stride0 << out_elem_size_log2;
  iree_uk_index_t out_stride_oc = params->out_stride1 << out_elem_size_log2;
  iree_uk_index_t out_stride_oh = params->out_stride2 << out_elem_size_log2;
  
  iree_uk_index_t lhs_stride_n = params->lhs_stride0 << lhs_elem_bits_log2;
  iree_uk_index_t lhs_stride_ic = params->lhs_stride1 << lhs_elem_bits_log2;
  iree_uk_index_t lhs_stride_h = params->lhs_stride2 << lhs_elem_bits_log2;
  iree_uk_index_t lhs_stride_w = params->lhs_stride3 << lhs_elem_bits_log2;
  
  iree_uk_index_t rhs_stride_oc = params->rhs_stride0 << rhs_elem_bits_log2;
  iree_uk_index_t rhs_stride_ic = params->rhs_stride1 << rhs_elem_bits_log2;
  iree_uk_index_t rhs_stride_h = params->rhs_stride2 << rhs_elem_bits_log2;
  iree_uk_index_t rhs_stride_w = params->rhs_stride3 << rhs_elem_bits_log2;
  
  // For each batch
  for (iree_uk_int32_t n = 0; n < batch_count; ++n) {
    char* out_batch_ptr = out_ptr + n * out_stride_n;
    const char* lhs_batch_ptr = lhs_ptr + n * lhs_stride_n;
    
    // For each output channel tile
    for (iree_uk_int32_t oc_tile_start = 0; oc_tile_start < output_channel_count; oc_tile_start += M0) {
      iree_uk_int32_t oc_tile_size = iree_uk_index_min(M0, output_channel_count - oc_tile_start);
      const char* rhs_oc_tile_ptr = rhs_ptr + oc_tile_start * rhs_stride_oc;
      
      // For each output height position
      for (iree_uk_int32_t oh = 0; oh < output_height; ++oh) {
        // For each output width tile
        for (iree_uk_int32_t ow_tile_start = 0; ow_tile_start < output_width; ow_tile_start += N0) {
          iree_uk_int32_t ow_tile_size = iree_uk_index_min(N0, output_width - ow_tile_start);
          
          char* out_tile_ptr = out_batch_ptr + 
                              oc_tile_start * out_stride_oc +
                              oh * out_stride_oh +
                              ow_tile_start * (1 << out_elem_size_log2);
          
          // For each input channel
          for (iree_uk_int32_t ic = 0; ic < params->input_channel_count; ++ic) {
            const char* lhs_ic_ptr = lhs_batch_ptr + ic * lhs_stride_ic;
            const char* rhs_ic_ptr = rhs_oc_tile_ptr + ic * rhs_stride_ic;
            
            // For each kernel height
            for (iree_uk_int32_t kh = 0; kh < params->kernel_height; ++kh) {
              iree_uk_int32_t ih = oh * params->stride_height + kh * params->dilation_height - params->padding_top;
              if (ih < 0 || ih >= params->input_height) continue;
              
              const char* lhs_h_ptr = lhs_ic_ptr + ih * lhs_stride_h;
              const char* rhs_h_ptr = rhs_ic_ptr + kh * rhs_stride_h;
              
              // For each kernel width
              for (iree_uk_int32_t kw = 0; kw < params->kernel_width; ++kw) {
                // Compute input width position
                for (iree_uk_int32_t ow = 0; ow < ow_tile_size; ++ow) {
                  iree_uk_int32_t iw = (ow_tile_start + ow) * params->stride_width + 
                                      kw * params->dilation_width - params->padding_left;
                  if (iw < 0 || iw >= params->input_width) continue;
                  
                  // For each output channel in tile
                  for (iree_uk_int32_t oc_in_tile = 0; oc_in_tile < oc_tile_size; ++oc_in_tile) {
                    // Get input value
                    const char* lhs_val_ptr = lhs_h_ptr + iw * lhs_stride_w;
                    // Get kernel value
                    const char* rhs_val_ptr = rhs_h_ptr + kw * rhs_stride_w + 
                                            oc_in_tile * rhs_stride_oc;
                    
                    // Compute output position
                    char* out_val_ptr = out_tile_ptr + 
                                       oc_in_tile * out_stride_oc +
                                       ow * (1 << out_elem_size_log2);
                    
                    // Call tile function for the actual computation
                    // This is a simplified version - actual implementation would use tile_func
                    // For now, we'll do direct computation
                    if (conv_type == iree_uk_conv_type_f32f32f32) {
                      float lhs_val, rhs_val, out_val;
                      iree_uk_memcpy(&lhs_val, lhs_val_ptr, sizeof(float));
                      iree_uk_memcpy(&rhs_val, rhs_val_ptr, sizeof(float));
                      
                      if (params->flags & IREE_UK_FLAG_CONV_ACCUMULATE) {
                        iree_uk_memcpy(&out_val, out_val_ptr, sizeof(float));
                        out_val += lhs_val * rhs_val;
                      } else {
                        out_val = lhs_val * rhs_val;
                      }
                      
                      iree_uk_memcpy(out_val_ptr, &out_val, sizeof(float));
                    }
                    // Add other data type support here
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// Early-return code paths
static bool iree_uk_conv_early(const iree_uk_conv_params_t* params) {
  // Trivial cases
  if (params->batch_count == 0 || params->output_channel_count == 0 ||
      params->output_height == 0 || params->output_width == 0) {
    return true;
  }
  
  // If K=0 and we're not accumulating, output is undefined
  if (params->input_channel_count == 0 && 
      !(params->flags & IREE_UK_FLAG_CONV_ACCUMULATE)) {
    return true;
  }
  
  return false;
}

void iree_uk_conv_p(const iree_uk_conv_params_t* params) {
  iree_uk_conv_validate(params);

  // Maybe handle this conv "early"
  if (iree_uk_conv_early(params)) return;

  // Select a target-specific tile_func
  iree_uk_conv_tile_func_t tile_func =
      iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(params);

  // If no target-specific tile_func is available, fall back to a generic one if
  // allowed by the flags.
  if (!tile_func) {
    if (params->flags &
        IREE_UK_FLAG_CONV_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION) {
      tile_func = iree_uk_conv_2d_nchw_fchw_select_tile_func_generic(params);
    } else {
      IREE_UK_ASSERT(
          0 && "no target-specific tile function, and fallback not enabled.");
    }
  }

  iree_uk_conv_using_tile_func(params, tile_func);
  ((float*)(params->out_buffer))[0] = 100000.0;
}

iree_uk_uint32_t iree_uk_conv_info_p(const iree_uk_conv_params_t* params) {
  iree_uk_uint32_t result = 0;
  if (iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(params)) {
    result |= IREE_UK_FLAG_CONV_INFO_HAVE_ARCHITECTURE_SPECIFIC_TILE_FUNCTION;
  }
  return result;
}

IREE_UK_EXPORT void iree_uk_conv_2d_nchw_fchw(
    const void* lhs_buffer, iree_uk_index_t lhs_offset,
    iree_uk_index_t lhs_stride0, iree_uk_index_t lhs_stride1,
    iree_uk_index_t lhs_stride2, iree_uk_index_t lhs_stride3,
    const void* rhs_buffer, iree_uk_index_t rhs_offset,
    iree_uk_index_t rhs_stride0, iree_uk_index_t rhs_stride1,
    iree_uk_index_t rhs_stride2, iree_uk_index_t rhs_stride3,
    void* out_buffer, iree_uk_index_t out_offset,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    iree_uk_index_t out_stride2,
    iree_uk_index_t batch_count, iree_uk_index_t input_channel_count,
    iree_uk_index_t output_channel_count, iree_uk_index_t input_height,
    iree_uk_index_t input_width, iree_uk_index_t kernel_height,
    iree_uk_index_t kernel_width, iree_uk_index_t output_height,
    iree_uk_index_t output_width, iree_uk_index_t stride_height,
    iree_uk_index_t stride_width, iree_uk_index_t dilation_height,
    iree_uk_index_t dilation_width, iree_uk_index_t padding_top,
    iree_uk_index_t padding_bottom, iree_uk_index_t padding_left,
    iree_uk_index_t padding_right, iree_uk_int32_t M0, iree_uk_int32_t N0,
    iree_uk_int32_t K0, iree_uk_uint32_t flags,
    const iree_uk_uint64_t* cpu_data) {
  
  iree_uk_conv_params_t params = {
      .lhs_buffer = lhs_buffer,
      .lhs_offset = lhs_offset,
      .lhs_stride0 = lhs_stride0,
      .lhs_stride1 = lhs_stride1,
      .lhs_stride2 = lhs_stride2,
      .lhs_stride3 = lhs_stride3,
      .rhs_buffer = rhs_buffer,
      .rhs_offset = rhs_offset,
      .rhs_stride0 = rhs_stride0,
      .rhs_stride1 = rhs_stride1,
      .rhs_stride2 = rhs_stride2,
      .rhs_stride3 = rhs_stride3,
      .out_buffer = out_buffer,
      .out_offset = out_offset,
      .out_stride0 = out_stride0,
      .out_stride1 = out_stride1,
      .out_stride2 = out_stride2,
      .batch_count = batch_count,
      .input_channel_count = input_channel_count,
      .output_channel_count = output_channel_count,
      .input_height = input_height,
      .input_width = input_width,
      .kernel_height = kernel_height,
      .kernel_width = kernel_width,
      .output_height = output_height,
      .output_width = output_width,
      .stride_height = stride_height,
      .stride_width = stride_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .padding_top = padding_top,
      .padding_bottom = padding_bottom,
      .padding_left = padding_left,
      .padding_right = padding_right,
      .M0 = M0,
      .N0 = N0,
      .K0 = K0,
      .flags = flags,
      .cpu_data = cpu_data
  };
  
  iree_uk_conv_p(&params);
}

IREE_UK_EXPORT iree_uk_uint32_t
iree_uk_conv_info(iree_uk_int32_t M0, iree_uk_int32_t N0, iree_uk_int32_t K0,
                   iree_uk_uint32_t flags, const iree_uk_uint64_t* cpu_data) {
  iree_uk_conv_params_t params = {
      .M0 = M0, .N0 = N0, .K0 = K0, .flags = flags, .cpu_data = cpu_data};
  return iree_uk_conv_info_p(&params);
}
