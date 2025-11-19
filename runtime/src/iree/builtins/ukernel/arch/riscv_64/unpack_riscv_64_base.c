// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/general_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/unpack_riscv_64_internal.h"

void iree_uk_unpack_tile_8x1_x8_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 1);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  // A further 2x unrolling (outer_size1-=16) yields another 1.2x speedup on
  // A710 thanks to using 16-byte loads. Is it worth the code size? This 8x1
  // tile is used on baseline aarch64 where the matmul kernel is slow anyway.
  for (; outer_size1 >= 8; outer_size1 -= 8) {
    iree_uk_copy_8x8xi8_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride0, in_stride1);
    out_ptr += 8;
    in_ptr += 8 * in_stride1;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_copy_8x1xi8_strided_to_unstrided(out_ptr, in_ptr, out_stride0);
    out_ptr += 1;
    in_ptr += in_stride1;
  }
}

void iree_uk_unpack_tile_8x4_x8_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 4);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 >= 2; outer_size1 -= 2) {
    iree_uk_copy_8x8xi8_tiled_1x4_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride0, in_stride1);
    out_ptr += 8;
    in_ptr += 2 * in_stride1;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_copy_8x4xi8_strided_to_unstrided(out_ptr, in_ptr, out_stride0);
    out_ptr += 4;
    in_ptr += in_stride1;
  }
}

void iree_uk_unpack_tile_8x1_x32_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 1);
  iree_uk_unpack_tile_8x4_x8_riscv_64_direct(out_tile_ptr, in_tile_ptr, outer_size1,
                                         out_stride0 * 4, in_stride1 * 4, 1, 8,
                                         4);
}

void iree_uk_unpack_tile_8x8_x8_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_copy_8x8xi8_strided_to_unstrided(out_ptr, in_ptr, out_stride0);
    out_ptr += 8;
    in_ptr += in_stride1;
  }
}

void iree_uk_unpack_tile_8x1_x32_riscv_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int32_t* IREE_UK_RESTRICT out_tile_i32_ptr = out_tile_ptr;
  const iree_uk_int32_t* IREE_UK_RESTRICT in_tile_ptr_i32 = in_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    //iree_uk_memcpy(out_tile_i32_ptr, in_tile_ptr_i32, 32);
    size_t vl = __riscv_vsetvl_e32m1(32);
    vint32m1_t vec = __riscv_vle32_v_i32m1(in_tile_ptr_i32, vl);
    __riscv_vse32_v_i32m1(out_tile_i32_ptr, vec, vl);
    out_tile_i32_ptr += 8;
    in_tile_ptr_i32 += in_stride1;
  }
}

void iree_uk_unpack_tile_8x1_x8_riscv_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 >= 4; outer_size1 -= 4) {
    //iree_uk_memcpy(out_ptr + 0 * out_stride1, in_ptr + 0, 8);
    //iree_uk_memcpy(out_ptr + 1 * out_stride1, in_ptr + 8, 8);
    //iree_uk_memcpy(out_ptr + 2 * out_stride1, in_ptr + 16, 8);
    //iree_uk_memcpy(out_ptr + 3 * out_stride1, in_ptr + 24, 8);
    size_t vl = __riscv_vsetvl_e8m1(8);
    vint8m1_t vec0 = __riscv_vle8_v_i8m1(in_ptr + 0, vl);
    __riscv_vse8_v_i8m1(out_ptr + 0 * out_stride0, vec0, vl);
    vint8m1_t vec1 = __riscv_vle8_v_i8m1(in_ptr + 8, vl);
    __riscv_vse8_v_i8m1(out_ptr + 1 * out_stride0, vec1, vl);
    vint8m1_t vec2 = __riscv_vle8_v_i8m1(in_ptr + 16, vl);
    __riscv_vse8_v_i8m1(out_ptr + 2 * out_stride0, vec2, vl);
    vint8m1_t vec3 = __riscv_vle8_v_i8m1(in_ptr + 24, vl);
    __riscv_vse8_v_i8m1(out_ptr + 3 * out_stride0, vec3, vl);
    out_ptr += 32;
    in_ptr += 4 * in_stride1;
  }
  for (; outer_size1 > 0; --outer_size1) {
    //iree_uk_memcpy(out_ptr, in_ptr, 8);
    size_t vl = __riscv_vsetvl_e8m1(8);
    vint8m1_t vec = __riscv_vle8_v_i8m1(in_ptr, vl);
    __riscv_vse8_v_i8m1(out_ptr, vec, vl);
    out_ptr += 8;
    in_ptr += in_stride1;
  }
}

void iree_uk_unpack_tile_8x4_x8_riscv_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 4);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    /*int8x16x2_t in;
    in.val[0] = vcombine_s8(vld1_s8(in_ptr + 0 * in_stride0),
                            vld1_s8(in_ptr + 2 * in_stride0));
    in.val[1] = vcombine_s8(vld1_s8(in_ptr + 1 * in_stride0),
                            vld1_s8(in_ptr + 3 * in_stride0));
    int16x8x2_t zip_i16 = iree_uk_neon_zip_16xi8_as_8xi16(in.val[0], in.val[1]);
    int32x4x2_t zip_i32 =
        iree_uk_neon_zip_8xi16_as_4xi32(zip_i16.val[0], zip_i16.val[1]);
    vst1q_s8(out_ptr, vreinterpretq_s8_s32(zip_i32.val[0]));
    vst1q_s8(out_ptr + 16, vreinterpretq_s8_s32(zip_i32.val[1]));*/
    size_t vl = __riscv_vsetvl_e8m1(8);
    vint8m1_t row0 = __riscv_vle8_v_i8m1(in_ptr + 0 * in_stride1, vl);
    vint8m1_t row1 = __riscv_vle8_v_i8m1(in_ptr + 1 * in_stride1, vl);
    vint8m1_t row2 = __riscv_vle8_v_i8m1(in_ptr + 2 * in_stride1, vl);
    vint8m1_t row3 = __riscv_vle8_v_i8m1(in_ptr + 3 * in_stride1, vl);
    vl = __riscv_vsetvl_e8m1(16);
    size_t half_vl = __riscv_vsetvl_e8m1(8);
    vint8m1_t in0 = __riscv_vslideup_vx_i8m1(row0, row2, half_vl, vl);
    vint8m1_t in1 = __riscv_vslideup_vx_i8m1(row1, row3, half_vl, vl);
    vint16m1x2_t zip_i16 = iree_uk_zip_16xi8_as_8xi16(in0, in1);
    vint16m1_t zip_i16_0 = __riscv_vget_v_i16m1x2_i16m1(zip_i16, 0);
    vint16m1_t zip_i16_1 = __riscv_vget_v_i16m1x2_i16m1(zip_i16, 1);
    vint32m1x2_t zip_i32 =
      iree_uk_zip_8xi16_as_4xi32(zip_i16_0, zip_i16_1);
    vint32m1_t zip_i32_0 = __riscv_vget_v_i32m1x2_i32m1(zip_i32, 0);
    vint32m1_t zip_i32_1 = __riscv_vget_v_i32m1x2_i32m1(zip_i32, 1);
    vint8m1_t out0 = __riscv_vreinterpret_v_i32m1_i8m1(zip_i32_0);
    vint8m1_t out1 = __riscv_vreinterpret_v_i32m1_i8m1(zip_i32_1);
    __riscv_vse8_v_i8m1(out_ptr + 0, out0, vl);
    __riscv_vse8_v_i8m1(out_ptr + 16, out1, vl);
    out_ptr += 8;
    in_ptr += in_stride1;
  }
}

void iree_uk_unpack_tile_8x8_x8_riscv_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_copy_8x8xi8_transpose_strided_to_unstrided(out_ptr, in_ptr,
                                                            out_stride0);

    out_ptr += 8;
    in_ptr += in_stride1;
  }
}

void iree_uk_unpack_tile_8x8_x32_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_copy_8x32xi8_strided_to_strided(out_ptr, in_ptr,
                                                 4 * out_stride0, 32);
    out_ptr += 32;
    in_ptr += 4 * in_stride1;
  }
}

void iree_uk_unpack_tile_generic_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  char* IREE_UK_RESTRICT out_ptr_l1 = out_tile_ptr;
  const char* IREE_UK_RESTRICT in_ptr_l1 = in_tile_ptr;
  for (iree_uk_index_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
    const char* IREE_UK_RESTRICT in_ptr = in_ptr_l1;
    char* IREE_UK_RESTRICT out_ptr = out_ptr_l1;
    for (iree_uk_index_t tile_i0 = 0; tile_i0 < tile_size0; ++tile_i0) {
      iree_uk_memcpy_riscv_64(out_ptr, in_ptr, tile_size1 * elem_size, elem_size);
      in_ptr += tile_size1 * elem_size;
      out_ptr += out_stride0 * elem_size;
    }
    in_ptr_l1 += in_stride1 * elem_size;
    out_ptr_l1 += tile_size1 * elem_size;
  }
}

void iree_uk_unpack_tile_generic_riscv_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  char* IREE_UK_RESTRICT out_ptr_l1 = out_tile_ptr;
  const char* IREE_UK_RESTRICT in_ptr_l1 = in_tile_ptr;
  for (iree_uk_index_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
    const char* IREE_UK_RESTRICT in_ptr_l2 = in_ptr_l1;
    char* IREE_UK_RESTRICT out_ptr_l2 = out_ptr_l1;
    for (iree_uk_index_t tile_i0 = 0; tile_i0 < tile_size0; ++tile_i0) {
      const char* IREE_UK_RESTRICT in_ptr = in_ptr_l2;
      char* IREE_UK_RESTRICT out_ptr = out_ptr_l2;
      for (iree_uk_index_t tile_i1 = 0; tile_i1 < tile_size1; ++tile_i1) {
        iree_uk_memcpy_riscv_64(out_ptr, in_ptr, elem_size, elem_size);
        in_ptr += tile_size0 * elem_size;
        out_ptr += elem_size;
      }
      in_ptr_l2 += elem_size;
      out_ptr_l2 += out_stride0 * elem_size;
    }
    in_ptr_l1 += in_stride1 * elem_size;
    out_ptr_l1 += tile_size1 * elem_size;
  }
}
