// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/conv_2d_nchw_fchw_internal.h"

// Generic implementation of conv tile, f32*f32->f32 case.
static void iree_uk_conv_tile_f32f32f32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_conv_params_t* params,
    iree_uk_int32_t tile_M0, iree_uk_int32_t tile_N0) {
  
  float* out_tile = out_tile_untyped;
  const float* lhs_panel = lhs_panel_untyped;
  const float* rhs_panel = rhs_panel_untyped;
  
  iree_uk_int16_t M0 = tile_M0;
  iree_uk_int16_t N0 = tile_N0;
  
  // For conv, we interpret the tile differently:
  // M0: output channels in tile
  // N0: output width in tile  
  // K0: input channels * kernel_height * kernel_width
  
  // Initialize output tile
  if (!(params->flags & IREE_UK_FLAG_CONV_ACCUMULATE)) {
    for (iree_uk_index_t i = 0; i < M0 * N0; ++i) {
      out_tile[i] = 0.0f;
    }
  }
  
  // For each element in the reduction dimension
  for (iree_uk_index_t k = 0; k < params->input_channel_count; ++k) {
    for (iree_uk_index_t kh = 0; kh < params->kernel_height; ++kh) {
      for (iree_uk_index_t kw = 0; kw < params->kernel_width; ++kw) {
        // For each output channel in tile
        for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
          // For each output width in tile
          for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
            // Compute indices
            iree_uk_index_t lhs_idx = k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t rhs_idx = i0 * params->input_channel_count * 
                                     params->kernel_height * params->kernel_width +
                                     k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t out_idx = i0 * N0 + j0;
            
            // Load values
            float lhs_val = lhs_panel[lhs_idx];
            float rhs_val = rhs_panel[rhs_idx];
            
            // Accumulate
            out_tile[out_idx] += lhs_val * rhs_val;
          }
        }
      }
    }
  }
}

// Generic implementation of conv tile, s8*s8->s32 case.
static void iree_uk_conv_tile_s8s8s32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_conv_params_t* params,
    iree_uk_int32_t tile_M0, iree_uk_int32_t tile_N0) {
  
  iree_uk_int32_t* out_tile = out_tile_untyped;
  const iree_uk_int8_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_int8_t* rhs_panel = rhs_panel_untyped;
  
  iree_uk_int16_t M0 = tile_M0;
  iree_uk_int16_t N0 = tile_N0;
  
  // Initialize output tile
  if (!(params->flags & IREE_UK_FLAG_CONV_ACCUMULATE)) {
    for (iree_uk_index_t i = 0; i < M0 * N0; ++i) {
      out_tile[i] = 0;
    }
  }
  
  // For each element in the reduction dimension
  for (iree_uk_index_t k = 0; k < params->input_channel_count; ++k) {
    for (iree_uk_index_t kh = 0; kh < params->kernel_height; ++kh) {
      for (iree_uk_index_t kw = 0; kw < params->kernel_width; ++kw) {
        // For each output channel in tile
        for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
          // For each output width in tile
          for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
            // Compute indices
            iree_uk_index_t lhs_idx = k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t rhs_idx = i0 * params->input_channel_count * 
                                     params->kernel_height * params->kernel_width +
                                     k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t out_idx = i0 * N0 + j0;
            
            // Load values and accumulate
            iree_uk_int32_t lhs_val = lhs_panel[lhs_idx];
            iree_uk_int32_t rhs_val = rhs_panel[rhs_idx];
            out_tile[out_idx] += lhs_val * rhs_val;
          }
        }
      }
    }
  }
}

// Generic implementation of conv tile, f16*f16->f32 case.
static void iree_uk_conv_tile_f16f16f32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_conv_params_t* params,
    iree_uk_int32_t tile_M0, iree_uk_int32_t tile_N0) {
  
  float* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  
  iree_uk_int16_t M0 = tile_M0;
  iree_uk_int16_t N0 = tile_N0;
  
  // Initialize output tile
  if (!(params->flags & IREE_UK_FLAG_CONV_ACCUMULATE)) {
    for (iree_uk_index_t i = 0; i < M0 * N0; ++i) {
      out_tile[i] = 0.0f;
    }
  }
  
  // For each element in the reduction dimension
  for (iree_uk_index_t k = 0; k < params->input_channel_count; ++k) {
    for (iree_uk_index_t kh = 0; kh < params->kernel_height; ++kh) {
      for (iree_uk_index_t kw = 0; kw < params->kernel_width; ++kw) {
        // For each output channel in tile
        for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
          // For each output width in tile
          for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
            // Compute indices
            iree_uk_index_t lhs_idx = k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t rhs_idx = i0 * params->input_channel_count * 
                                     params->kernel_height * params->kernel_width +
                                     k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t out_idx = i0 * N0 + j0;
            
            // Load values, convert to f32, and accumulate
            float lhs_val = iree_uk_f16_to_f32(lhs_panel[lhs_idx]);
            float rhs_val = iree_uk_f16_to_f32(rhs_panel[rhs_idx]);
            out_tile[out_idx] += lhs_val * rhs_val;
          }
        }
      }
    }
  }
}

// Generic implementation of conv tile, f16*f16->f16 case.
static void iree_uk_conv_tile_f16f16f16_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_conv_params_t* params,
    iree_uk_int32_t tile_M0, iree_uk_int32_t tile_N0) {
  
  iree_uk_uint16_t* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  
  iree_uk_int16_t M0 = tile_M0;
  iree_uk_int16_t N0 = tile_N0;
  
  // For each output channel in tile
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    // For each output width in tile
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      iree_uk_index_t out_idx = i0 * N0 + j0;
      
      float acc_f32 = (params->flags & IREE_UK_FLAG_CONV_ACCUMULATE)
                        ? iree_uk_f16_to_f32(out_tile[out_idx])
                        : 0.0f;
      
      // For each element in the reduction dimension
      for (iree_uk_index_t k = 0; k < params->input_channel_count; ++k) {
        for (iree_uk_index_t kh = 0; kh < params->kernel_height; ++kh) {
          for (iree_uk_index_t kw = 0; kw < params->kernel_width; ++kw) {
            // Compute indices
            iree_uk_index_t lhs_idx = k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t rhs_idx = i0 * params->input_channel_count * 
                                     params->kernel_height * params->kernel_width +
                                     k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            
            // Load values, convert to f32, and accumulate
            float lhs_val = iree_uk_f16_to_f32(lhs_panel[lhs_idx]);
            float rhs_val = iree_uk_f16_to_f32(rhs_panel[rhs_idx]);
            
            if (params->flags & IREE_UK_FLAG_CONV_SKIP_INTERMEDIATE_ROUNDINGS) {
              acc_f32 += lhs_val * rhs_val;
            } else {
              // Round at each accumulation step
              acc_f32 = iree_uk_f16_to_f32(
                  iree_uk_f32_to_f16(acc_f32 + lhs_val * rhs_val));
            }
          }
        }
      }
      
      // Store result
      out_tile[out_idx] = iree_uk_f32_to_f16(acc_f32);
    }
  }
}

// Generic implementation of conv tile, bf16*bf16->f32 case.
static void iree_uk_conv_tile_bf16bf16f32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_conv_params_t* params,
    iree_uk_int32_t tile_M0, iree_uk_int32_t tile_N0) {
  
  float* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  
  iree_uk_int16_t M0 = tile_M0;
  iree_uk_int16_t N0 = tile_N0;
  
  // Initialize output tile
  if (!(params->flags & IREE_UK_FLAG_CONV_ACCUMULATE)) {
    for (iree_uk_index_t i = 0; i < M0 * N0; ++i) {
      out_tile[i] = 0.0f;
    }
  }
  
  // For each element in the reduction dimension
  for (iree_uk_index_t k = 0; k < params->input_channel_count; ++k) {
    for (iree_uk_index_t kh = 0; kh < params->kernel_height; ++kh) {
      for (iree_uk_index_t kw = 0; kw < params->kernel_width; ++kw) {
        // For each output channel in tile
        for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
          // For each output width in tile
          for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
            // Compute indices
            iree_uk_index_t lhs_idx = k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t rhs_idx = i0 * params->input_channel_count * 
                                     params->kernel_height * params->kernel_width +
                                     k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t out_idx = i0 * N0 + j0;
            
            // Load values, convert to f32, and accumulate
            float lhs_val = iree_uk_bf16_to_f32(lhs_panel[lhs_idx]);
            float rhs_val = iree_uk_bf16_to_f32(rhs_panel[rhs_idx]);
            out_tile[out_idx] += lhs_val * rhs_val;
          }
        }
      }
    }
  }
}

// Generic implementation of conv tile, bf16*bf16->bf16 case.
static void iree_uk_conv_tile_bf16bf16bf16_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_conv_params_t* params,
    iree_uk_int32_t tile_M0, iree_uk_int32_t tile_N0) {
  
  iree_uk_uint16_t* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  
  iree_uk_int16_t M0 = tile_M0;
  iree_uk_int16_t N0 = tile_N0;
  
  // For each output channel in tile
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    // For each output width in tile
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      iree_uk_index_t out_idx = i0 * N0 + j0;
      
      float acc_f32 = (params->flags & IREE_UK_FLAG_CONV_ACCUMULATE)
                        ? iree_uk_bf16_to_f32(out_tile[out_idx])
                        : 0.0f;
      
      // For each element in the reduction dimension
      for (iree_uk_index_t k = 0; k < params->input_channel_count; ++k) {
        for (iree_uk_index_t kh = 0; kh < params->kernel_height; ++kh) {
          for (iree_uk_index_t kw = 0; kw < params->kernel_width; ++kw) {
            // Compute indices
            iree_uk_index_t lhs_idx = k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            iree_uk_index_t rhs_idx = i0 * params->input_channel_count * 
                                     params->kernel_height * params->kernel_width +
                                     k * params->kernel_height * params->kernel_width +
                                     kh * params->kernel_width + kw;
            
            // Load values, convert to f32, and accumulate
            float lhs_val = iree_uk_bf16_to_f32(lhs_panel[lhs_idx]);
            float rhs_val = iree_uk_bf16_to_f32(rhs_panel[rhs_idx]);
            
            if (params->flags & IREE_UK_FLAG_CONV_SKIP_INTERMEDIATE_ROUNDINGS) {
              acc_f32 += lhs_val * rhs_val;
            } else {
              // Round at each accumulation step
              acc_f32 = iree_uk_bf16_to_f32(
                  iree_uk_f32_to_bf16(acc_f32 + lhs_val * rhs_val));
            }
          }
        }
      }
      
      // Store result
      out_tile[out_idx] = iree_uk_f32_to_bf16(acc_f32);
    }
  }
}

// Generic tile function selector
iree_uk_conv_tile_func_t iree_uk_conv_2d_nchw_fchw_select_tile_func_generic(
    const iree_uk_conv_params_t* params) {
  switch (iree_uk_conv_type(params->flags)) {
    case iree_uk_conv_type_f32f32f32:
      return iree_uk_conv_tile_f32f32f32_generic;
    case iree_uk_conv_type_s8s8s32:
      return iree_uk_conv_tile_s8s8s32_generic;
    case iree_uk_conv_type_f16f16f32:
      return iree_uk_conv_tile_f16f16f32_generic;
    case iree_uk_conv_type_f16f16f16:
      return iree_uk_conv_tile_f16f16f16_generic;
    case iree_uk_conv_type_bf16bf16f32:
      return iree_uk_conv_tile_bf16bf16f32_generic;
    case iree_uk_conv_type_bf16bf16bf16:
      return iree_uk_conv_tile_bf16bf16bf16_generic;
    case iree_uk_conv_type_s16s16s32:
    case iree_uk_conv_type_s16u4s32:
    case iree_uk_conv_type_s16s8s32:
    case iree_uk_conv_type_s8s4s32:
      // TODO: Implement these types
      return 0;
    default:
      // Shouldn't happen, validated earlier.
      return 0;
  }
}
