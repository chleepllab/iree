// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_CONV_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_CONV_INTERNAL_H_

#include "iree/builtins/ukernel/conv_2d_nchw_fchw.h"

// While the iree_uk_conv public entry point takes separate parameters,
// internally the implementation functions pass parameters as this struct.
typedef struct iree_uk_conv_params_t {
  const void* lhs_buffer;
  iree_uk_index_t lhs_offset;
  const void* rhs_buffer;
  iree_uk_index_t rhs_offset;
  void* out_buffer;
  iree_uk_index_t out_offset;
  iree_uk_index_t batch_count;
  iree_uk_index_t input_channel_count;
  iree_uk_index_t output_channel_count;
  iree_uk_index_t input_height;
  iree_uk_index_t input_width;
  iree_uk_index_t kernel_height;
  iree_uk_index_t kernel_width;
  iree_uk_index_t output_height;
  iree_uk_index_t output_width;
  iree_uk_index_t tile_size0;
  iree_uk_index_t tile_size1;
  iree_uk_uint32_t flags;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_conv_params_t;

// Same as the iree_uk_conv public entry point, but taking the struct.
void iree_uk_conv_p(const iree_uk_conv_params_t* params);

typedef enum iree_uk_conv_type_t {
  iree_uk_conv_type_f32f32f32 =
      IREE_UK_TIE_3_TYPES_LITERAL(FLOAT_32, FLOAT_32, FLOAT_32),
  iree_uk_conv_type_s8s8s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_8, SINT_8, SINT_32),
  iree_uk_conv_type_f16f16f32 =
      IREE_UK_TIE_3_TYPES_LITERAL(FLOAT_16, FLOAT_16, FLOAT_32),
  iree_uk_conv_type_f16f16f16 =
      IREE_UK_TIE_3_TYPES_LITERAL(FLOAT_16, FLOAT_16, FLOAT_16),
  iree_uk_conv_type_bf16bf16f32 =
      IREE_UK_TIE_3_TYPES_LITERAL(BFLOAT_16, BFLOAT_16, FLOAT_32),
  iree_uk_conv_type_bf16bf16bf16 =
      IREE_UK_TIE_3_TYPES_LITERAL(BFLOAT_16, BFLOAT_16, BFLOAT_16),
  iree_uk_conv_type_s16s16s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_16, SINT_16, SINT_32),
  iree_uk_conv_type_s16u4s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_16, UINT_4, SINT_32),
  iree_uk_conv_type_s16s8s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_16, SINT_8, SINT_32),
  iree_uk_conv_type_s8s4s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_8, SINT_4, SINT_32),
} iree_uk_conv_type_t;

static inline iree_uk_conv_type_t iree_uk_conv_type(iree_uk_uint32_t flags) {
  switch (flags & IREE_UK_FLAG_CONV_TYPE_MASK) {
    case IREE_UK_FLAG_CONV_TYPE_F32F32F32:
      return iree_uk_conv_type_f32f32f32;
    case IREE_UK_FLAG_CONV_TYPE_S8S8S32:
      return iree_uk_conv_type_s8s8s32;
    case IREE_UK_FLAG_CONV_TYPE_F16F16F32:
      return iree_uk_conv_type_f16f16f32;
    case IREE_UK_FLAG_CONV_TYPE_F16F16F16:
      return iree_uk_conv_type_f16f16f16;
    case IREE_UK_FLAG_CONV_TYPE_BF16BF16F32:
      return iree_uk_conv_type_bf16bf16f32;
    case IREE_UK_FLAG_CONV_TYPE_BF16BF16BF16:
      return iree_uk_conv_type_bf16bf16bf16;
    case IREE_UK_FLAG_CONV_TYPE_S16S16S32:
      return iree_uk_conv_type_s16s16s32;
    case IREE_UK_FLAG_CONV_TYPE_S16U4S32:
      return iree_uk_conv_type_s16u4s32;
    case IREE_UK_FLAG_CONV_TYPE_S16S8S32:
      return iree_uk_conv_type_s16s8s32;
    case IREE_UK_FLAG_CONV_TYPE_S8S4S32:
      return iree_uk_conv_type_s8s4s32;
    default:
      // Shouldn't happen, validated earlier.
      return (iree_uk_conv_type_t)0;
  }
}

static inline iree_uk_type_t iree_uk_conv_lhs_type(iree_uk_conv_type_t type) {
  return iree_uk_untie_type(0, type);
}

static inline iree_uk_type_t iree_uk_conv_rhs_type(iree_uk_conv_type_t type) {
  return iree_uk_untie_type(1, type);
}

static inline iree_uk_type_t iree_uk_conv_out_type(iree_uk_conv_type_t type) {
  return iree_uk_untie_type(2, type);
}

// Function pointer type for tile functions
typedef void (*iree_uk_conv_tile_func_t)(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, const void* IREE_UK_RESTRICT kernel_tile_ptr,
    iree_uk_index_t in_size0, iree_uk_index_t in_size1,
    iree_uk_index_t kernel_size0, iree_uk_index_t kernel_size1,
    iree_uk_index_t outer_size0, iree_uk_index_t outer_size1,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,
    iree_uk_index_t elem_size);

// Tile kernel declarations
#define IREE_UK_CONV_TILE_FUNC_DECL(NAME)                             \
  void NAME(void* IREE_UK_RESTRICT out_tile_ptr,                      \
            const void* IREE_UK_RESTRICT in_tile_ptr,                 \
            const void* IREE_UK_RESTRICT kernel_tile_ptr,             \
            iree_uk_index_t in_size0, iree_uk_index_t in_size1,       \
            iree_uk_index_t kernel_size0, iree_uk_index_t kernel_size1, \
            iree_uk_index_t outer_size0, iree_uk_index_t outer_size1,   \
            iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,     \
            iree_uk_index_t elem_size);

// Architecture-specific implementation, or generic fallback returning null.
iree_uk_conv_tile_func_t iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(
    const iree_uk_conv_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_CONV_INTERNAL_H_
