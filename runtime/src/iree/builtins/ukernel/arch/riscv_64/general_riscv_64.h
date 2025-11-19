// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_GENERAL_RISCV_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_GENERAL_RISCV_64_H_

//#include <stdlib.h>
#include <riscv_vector.h>

#include "iree/builtins/ukernel/common.h"

/*static inline int get_lmul_env() {
  char* lmul_env = getenv("IREE_RVV_LMUL");
  if (lmul_env) {
    int lmul = atoi(lmul_env);
    if (lmul == 1 || lmul == 2 || lmul == 4 || lmul == 8) {
      return lmul;
    }
  }
  return 1;
}*/

static inline void iree_uk_memcpy_riscv_64(void* IREE_UK_RESTRICT dst,
                                  const void* IREE_UK_RESTRICT src,
                                  iree_uk_index_t size,
                                  iree_uk_index_t elem_size) {
  //for (iree_uk_index_t i = 0; i < size; ++i)
  //  ((char*)dst)[i] = ((const char*)src)[i];
  iree_uk_index_t n = size;
  if (elem_size == 4) {
    iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = dst;
    const iree_uk_int32_t* IREE_UK_RESTRICT in_ptr = src;
    size_t vl = __riscv_vsetvl_e32m1(n);
    for (iree_uk_index_t i = 0; i < n; i += vl) {
      vl = __riscv_vsetvl_e32m1(n - i);
      vint32m1_t vec = __riscv_vle32_v_i32m1((const int32_t*)(in_ptr + i), vl);
      __riscv_vse32_v_i32m1((int32_t*)(out_ptr + i), vec, vl);
    }
  } else if (elem_size == 1) {
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = dst;
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = src;
    size_t vl = __riscv_vsetvl_e8m1(n);
    for (iree_uk_index_t i = 0; i < n; i += vl) {
      vl = __riscv_vsetvl_e8m1(n - i);
      vint8m1_t vec = __riscv_vle8_v_i8m1((const int8_t*)(in_ptr + i), vl);
      __riscv_vse8_v_i8m1((int8_t*)(out_ptr + i), vec, vl);
    }
  }
}

//[8x4_x8]
static inline vint8m1x2_t iree_uk_load_8x4xi8_strided(
    const iree_uk_int8_t* src, iree_uk_index_t stride) {
  size_t vl = __riscv_vsetvl_e32m1(4);
  vint32m1_t lane0 = __riscv_vle32_v_i32m1((const int32_t*)(src + 0 * stride), vl);
  vint32m1_t lane1 = __riscv_vle32_v_i32m1((const int32_t*)(src + 1 * stride), vl);
  vint32m1_t lane2 = __riscv_vle32_v_i32m1((const int32_t*)(src + 2 * stride), vl);
  vint32m1_t lane3 = __riscv_vle32_v_i32m1((const int32_t*)(src + 3 * stride), vl);
  int32_t e0 = __riscv_vmv_x_s_i32m1_i32(lane0);
  int32_t e1 = __riscv_vmv_x_s_i32m1_i32(lane1);
  int32_t e2 = __riscv_vmv_x_s_i32m1_i32(lane2);
  int32_t e3 = __riscv_vmv_x_s_i32m1_i32(lane3);
  vint32m1_t temp0 = __riscv_vmv_v_x_i32m1(e0, vl);
  temp0 = __riscv_vslide1down_vx_i32m1(temp0, e1, vl);
  temp0 = __riscv_vslide1down_vx_i32m1(temp0, e2, vl);
  temp0 = __riscv_vslide1down_vx_i32m1(temp0, e3, vl);
  vint32m1_t lane4 = __riscv_vle32_v_i32m1((const int32_t*)(src + 4 * stride), vl);
  vint32m1_t lane5 = __riscv_vle32_v_i32m1((const int32_t*)(src + 5 * stride), vl);
  vint32m1_t lane6 = __riscv_vle32_v_i32m1((const int32_t*)(src + 6 * stride), vl);
  vint32m1_t lane7 = __riscv_vle32_v_i32m1((const int32_t*)(src + 7 * stride), vl);
  int32_t e4 = __riscv_vmv_x_s_i32m1_i32(lane4);
  int32_t e5 = __riscv_vmv_x_s_i32m1_i32(lane5);
  int32_t e6 = __riscv_vmv_x_s_i32m1_i32(lane6);
  int32_t e7 = __riscv_vmv_x_s_i32m1_i32(lane7);
  vint32m1_t temp1 = __riscv_vmv_v_x_i32m1(e4, vl);
  temp1 = __riscv_vslide1down_vx_i32m1(temp1, e5, vl);
  temp1 = __riscv_vslide1down_vx_i32m1(temp1, e6, vl);
  temp1 = __riscv_vslide1down_vx_i32m1(temp1, e7, vl);
  vint8m1_t v0_i8 = __riscv_vreinterpret_v_i32m1_i8m1(temp0);
  vint8m1_t v1_i8 = __riscv_vreinterpret_v_i32m1_i8m1(temp1);
  vint8m1x2_t v = __riscv_vcreate_v_i8m1x2(v0_i8, v1_i8);
  //size_t vl = __riscv_vsetvl_e32m1(4);
  //vint32m1_t dummy_i32 = __riscv_vmv_v_x_i32m1(0, vl);
  //vint8m1_t dummy = __riscv_vreinterpret_v_i32m1_i8m1(dummy_i32);
  //vint8m1x2_t v = __riscv_vcreate_v_i8m1x2(dummy, dummy);
  return v;
}

//[8x4_x8]
static inline vint8m1x4_t iree_uk_load_8x8xi8_strided_permute(
    const iree_uk_int8_t* src, iree_uk_index_t stride, int p0, int p1, int p2,
    int p3, int p4, int p5, int p6, int p7) {
  size_t vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t row0 = __riscv_vle8_v_i8m1(src + p0 * stride, vl);
  vint8m1_t row1 = __riscv_vle8_v_i8m1(src + p1 * stride, vl);
  vint8m1_t row2 = __riscv_vle8_v_i8m1(src + p2 * stride, vl);
  vint8m1_t row3 = __riscv_vle8_v_i8m1(src + p3 * stride, vl);
  vint8m1_t row4 = __riscv_vle8_v_i8m1(src + p4 * stride, vl);
  vint8m1_t row5 = __riscv_vle8_v_i8m1(src + p5 * stride, vl);
  vint8m1_t row6 = __riscv_vle8_v_i8m1(src + p6 * stride, vl);
  vint8m1_t row7 = __riscv_vle8_v_i8m1(src + p7 * stride, vl);
  vl = __riscv_vsetvl_e8m1(16);
  size_t half_vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t v0 = __riscv_vslideup_vx_i8m1(row0, row1, half_vl, vl);
  vint8m1_t v1 = __riscv_vslideup_vx_i8m1(row2, row3, half_vl, vl);
  vint8m1_t v2 = __riscv_vslideup_vx_i8m1(row4, row5, half_vl, vl);
  vint8m1_t v3 = __riscv_vslideup_vx_i8m1(row6, row7, half_vl, vl);
  //size_t vl = __riscv_vsetvl_e8m1(8);
  //vint8m1_t v0 = __riscv_vmv_v_x_i8m1(0, vl);
  //vint8m1_t v1 = __riscv_vmv_v_x_i8m1(0, vl);
  //vint8m1_t v2 = __riscv_vmv_v_x_i8m1(0, vl);
  //vint8m1_t v3 = __riscv_vmv_v_x_i8m1(0, vl);
  vint8m1x4_t v = __riscv_vcreate_v_i8m1x4(v0, v1, v2, v3);
  return v;
}

static inline vint8m1x4_t iree_uk_load_8x8xi8_strided(
    const iree_uk_int8_t* src, iree_uk_index_t stride) {
  return iree_uk_load_8x8xi8_strided_permute(src, stride, 0, 1, 2, 3, 4, 5,
                                                  6, 7);
}

static inline vint16m1x2_t iree_uk_zip_16xi8_as_8xi16(vint8m1_t a,
                                                          vint8m1_t b) {
  size_t vl = __riscv_vsetvl_e8m1(16);
  size_t half_vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t a_lo = __riscv_vslidedown_vx_i8m1(a, 0, half_vl);
  vint8m1_t a_hi = __riscv_vslidedown_vx_i8m1(a, half_vl, half_vl);
  vint8m1_t b_lo = __riscv_vslidedown_vx_i8m1(b, 0, half_vl);
  vint8m1_t b_hi = __riscv_vslidedown_vx_i8m1(b, half_vl, half_vl);
  vint8m1_t in0 = __riscv_vslideup_vx_i8m1(a_lo, b_lo, half_vl, vl);
  vint8m1_t in1 = __riscv_vslideup_vx_i8m1(a_hi, b_hi, half_vl, vl);
  uint8_t lane[16] = {0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15};
  vuint8m1_t idx = __riscv_vle8_v_u8m1(lane, vl);
  vint8m1_t z0 = __riscv_vrgather_vv_i8m1(in0, idx, vl);
  vint8m1_t z1 = __riscv_vrgather_vv_i8m1(in1, idx, vl);
  vint16m1_t r0 = __riscv_vreinterpret_v_i8m1_i16m1(z0);
  vint16m1_t r1 = __riscv_vreinterpret_v_i8m1_i16m1(z1);
  vint16m1x2_t r = __riscv_vcreate_v_i16m1x2(r0, r1);
  return r;
}

static inline vint32m1x2_t iree_uk_zip_8xi16_as_4xi32(vint16m1_t a,
                                                          vint16m1_t b) {
  size_t vl = __riscv_vsetvl_e16m1(8);
  size_t half_vl = __riscv_vsetvl_e16m1(4);
  vint16m1_t a_lo = __riscv_vslidedown_vx_i16m1(a, 0, half_vl);
  vint16m1_t a_hi = __riscv_vslidedown_vx_i16m1(a, half_vl, half_vl);
  vint16m1_t b_lo = __riscv_vslidedown_vx_i16m1(b, 0, half_vl);
  vint16m1_t b_hi = __riscv_vslidedown_vx_i16m1(b, half_vl, half_vl);
  vint16m1_t in0 = __riscv_vslideup_vx_i16m1(a_lo, b_lo, half_vl, vl);
  vint16m1_t in1 = __riscv_vslideup_vx_i16m1(a_hi, b_hi, half_vl, vl);
  uint16_t lane[8] = {0,4,1,5,2,6,3,7};
  vuint16m1_t idx = __riscv_vle16_v_u16m1(lane, vl);
  vint16m1_t z0 = __riscv_vrgather_vv_i16m1(in0, idx, vl);
  vint16m1_t z1 = __riscv_vrgather_vv_i16m1(in1, idx, vl);
  vint32m1_t r0 = __riscv_vreinterpret_v_i16m1_i32m1(z0);
  vint32m1_t r1 = __riscv_vreinterpret_v_i16m1_i32m1(z1);
  vint32m1x2_t r = __riscv_vcreate_v_i32m1x2(r0, r1);
  return r;
}

static inline vint64m1x2_t iree_uk_zip_4xi32_as_2xi64(vint32m1_t a,
                                                          vint32m1_t b) {
  size_t vl = __riscv_vsetvl_e32m1(4);
  size_t half_vl = __riscv_vsetvl_e32m1(2);
  vint32m1_t a_lo = __riscv_vslidedown_vx_i32m1(a, 0, half_vl);
  vint32m1_t a_hi = __riscv_vslidedown_vx_i32m1(a, half_vl, half_vl);
  vint32m1_t b_lo = __riscv_vslidedown_vx_i32m1(b, 0, half_vl);
  vint32m1_t b_hi = __riscv_vslidedown_vx_i32m1(b, half_vl, half_vl);
  vint32m1_t in0 = __riscv_vslideup_vx_i32m1(a_lo, b_lo, half_vl, vl);
  vint32m1_t in1 = __riscv_vslideup_vx_i32m1(a_hi, b_hi, half_vl, vl);
  uint32_t lane[4] = {0,2,1,3};
  vuint32m1_t idx = __riscv_vle32_v_u32m1(lane, vl);
  vint32m1_t z0 = __riscv_vrgather_vv_i32m1(in0, idx, vl);
  vint32m1_t z1 = __riscv_vrgather_vv_i32m1(in1, idx, vl);
  vint64m1_t r0 = __riscv_vreinterpret_v_i32m1_i64m1(z0);
  vint64m1_t r1 = __riscv_vreinterpret_v_i32m1_i64m1(z1);
  vint64m1x2_t r = __riscv_vcreate_v_i64m1x2(r0, r1);
  return r;
}

//[8x1_x8]
static inline void iree_uk_copy_8x1xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  size_t vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t lane0 = __riscv_vle8_v_i8m1(in_ptr + 0 * in_stride, vl);
  vint8m1_t lane1 = __riscv_vle8_v_i8m1(in_ptr + 1 * in_stride, vl);
  vint8m1_t lane2 = __riscv_vle8_v_i8m1(in_ptr + 2 * in_stride, vl);
  vint8m1_t lane3 = __riscv_vle8_v_i8m1(in_ptr + 3 * in_stride, vl);
  vint8m1_t lane4 = __riscv_vle8_v_i8m1(in_ptr + 4 * in_stride, vl);
  vint8m1_t lane5 = __riscv_vle8_v_i8m1(in_ptr + 5 * in_stride, vl);
  vint8m1_t lane6 = __riscv_vle8_v_i8m1(in_ptr + 6 * in_stride, vl);
  vint8m1_t lane7 = __riscv_vle8_v_i8m1(in_ptr + 7 * in_stride, vl);
  int8_t e0 = __riscv_vmv_x_s_i8m1_i8(lane0);
  int8_t e1 = __riscv_vmv_x_s_i8m1_i8(lane1);
  int8_t e2 = __riscv_vmv_x_s_i8m1_i8(lane2);
  int8_t e3 = __riscv_vmv_x_s_i8m1_i8(lane3);
  int8_t e4 = __riscv_vmv_x_s_i8m1_i8(lane4);
  int8_t e5 = __riscv_vmv_x_s_i8m1_i8(lane5);
  int8_t e6 = __riscv_vmv_x_s_i8m1_i8(lane6);
  int8_t e7 = __riscv_vmv_x_s_i8m1_i8(lane7);
  vint8m1_t temp = __riscv_vmv_v_x_i8m1(e0, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e1, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e2, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e3, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e4, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e5, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e6, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e7, vl);
  __riscv_vse8_v_i8m1(out_ptr, temp, vl);
}

//[8x4_x8]
static inline void iree_uk_copy_8x4xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  vint8m1x2_t in = iree_uk_load_8x4xi8_strided(in_ptr, in_stride);
  size_t vl = __riscv_vsetvl_e8m1(16);
  vint8m1_t in0 = __riscv_vget_v_i8m1x2_i8m1(in, 0);
  vint8m1_t in1 = __riscv_vget_v_i8m1x2_i8m1(in, 1);
  __riscv_vse8_v_i8m1(out_ptr + 0, in0, vl);
  __riscv_vse8_v_i8m1(out_ptr + 16, in1, vl);
}

//[8x8_x8]
static inline void iree_uk_copy_8x8xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  vint8m1x4_t in = iree_uk_load_8x8xi8_strided(in_ptr, in_stride);
  vint8m1_t in0 = __riscv_vget_v_i8m1x4_i8m1(in, 0);
  vint8m1_t in1 = __riscv_vget_v_i8m1x4_i8m1(in, 1);
  vint8m1_t in2 = __riscv_vget_v_i8m1x4_i8m1(in, 2);
  vint8m1_t in3 = __riscv_vget_v_i8m1x4_i8m1(in, 3);
  size_t vl = __riscv_vsetvl_e8m1(16);
  __riscv_vse8_v_i8m1(out_ptr + 0, in0, vl);
  __riscv_vse8_v_i8m1(out_ptr + 16, in1, vl);
  __riscv_vse8_v_i8m1(out_ptr + 32, in2, vl);
  __riscv_vse8_v_i8m1(out_ptr + 48, in3, vl);
}

//[8x4_x8]
static inline void
iree_uk_copy_8x8xi8_tiled_1x4_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  vint8m1x4_t in = iree_uk_load_8x8xi8_strided_permute(
      in_ptr, in_stride, 0, 2, 1, 3, 4, 6, 5, 7);
  vint8m1_t in0 = __riscv_vget_v_i8m1x4_i8m1(in, 0);
  vint8m1_t in1 = __riscv_vget_v_i8m1x4_i8m1(in, 1);
  vint8m1_t in2 = __riscv_vget_v_i8m1x4_i8m1(in, 2);
  vint8m1_t in3 = __riscv_vget_v_i8m1x4_i8m1(in, 3);
  vint32m1_t row0 = __riscv_vreinterpret_v_i8m1_i32m1(in0);
  vint32m1_t row1 = __riscv_vreinterpret_v_i8m1_i32m1(in1);
  vint32m1_t row2 = __riscv_vreinterpret_v_i8m1_i32m1(in2);
  vint32m1_t row3 = __riscv_vreinterpret_v_i8m1_i32m1(in3);
  size_t vl = __riscv_vsetvl_e32m1(8);
  size_t half_vl = __riscv_vsetvl_e32m1(4);
  vint32m1_t temp0 = __riscv_vslideup_vx_i32m1(row0, row1, half_vl, vl);
  vint32m1_t temp1 = __riscv_vslideup_vx_i32m1(row2, row3, half_vl, vl);
  vl = __riscv_vsetvl_e32m1(4);
  uint32_t lane0[4] = {0,4,2,6};
  vuint32m1_t idx0 = __riscv_vle32_v_u32m1(lane0, vl);
  uint32_t lane1[4] = {1,5,3,7};
  vuint32m1_t idx1 = __riscv_vle32_v_u32m1(lane1, vl);
  vint32m1_t c0_0 = __riscv_vrgather_vv_i32m1(temp0, idx0, vl);
  vint32m1_t c0_1 = __riscv_vrgather_vv_i32m1(temp0, idx1, vl);
  vint32m1_t c1_0 = __riscv_vrgather_vv_i32m1(temp1, idx0, vl);
  vint32m1_t c1_1 = __riscv_vrgather_vv_i32m1(temp1, idx1, vl);
  vint8m1_t c0_0_i8 = __riscv_vreinterpret_v_i32m1_i8m1(c0_0);
  vint8m1_t c1_0_i8 = __riscv_vreinterpret_v_i32m1_i8m1(c1_0);
  vint8m1_t c0_1_i8 = __riscv_vreinterpret_v_i32m1_i8m1(c0_1);
  vint8m1_t c1_1_i8 = __riscv_vreinterpret_v_i32m1_i8m1(c1_1);
  vl = __riscv_vsetvl_e8m1(16);
  __riscv_vse8_v_i8m1(out_ptr + 0 + 0 * out_stride, c0_0_i8, vl);
  __riscv_vse8_v_i8m1(out_ptr + 16 + 0 * out_stride, c1_0_i8, vl);
  __riscv_vse8_v_i8m1(out_ptr + 0 + 1 * out_stride, c0_1_i8, vl);
  __riscv_vse8_v_i8m1(out_ptr + 16 + 1 * out_stride, c1_1_i8, vl);
}

static inline void iree_uk_copy_8x32xi8_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  for (int i = 0; i < 8; ++i) {
    //iree_uk_memcpy(out_ptr + i * out_stride, in_ptr + i * in_stride, 32);
    size_t vl = __riscv_vsetvl_e8m4(32);
    vint8m4_t vec = __riscv_vle8_v_i8m4(in_ptr + i * in_stride, vl);
    __riscv_vse8_v_i8m4(out_ptr + i * out_stride, vec, vl);
  }
}

//[8x1_x8 8x8_x8]
static inline void iree_uk_copy_8x8xi8_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  vint8m1x4_t in = iree_uk_load_8x8xi8_strided_permute(
      in_ptr, in_stride, 0, 4, 1, 5, 2, 6, 3, 7);
  vint8m1_t in0 = __riscv_vget_v_i8m1x4_i8m1(in, 0);
  vint8m1_t in1 = __riscv_vget_v_i8m1x4_i8m1(in, 1);
  vint8m1_t in2 = __riscv_vget_v_i8m1x4_i8m1(in, 2);
  vint8m1_t in3 = __riscv_vget_v_i8m1x4_i8m1(in, 3);
  vint16m1x2_t zip_i16_0 = iree_uk_zip_16xi8_as_8xi16(in0, in1);
  vint16m1x2_t zip_i16_1 = iree_uk_zip_16xi8_as_8xi16(in2, in3);
  vint16m1_t zip_i16_00 = __riscv_vget_v_i16m1x2_i16m1(zip_i16_0, 0);
  vint16m1_t zip_i16_01 = __riscv_vget_v_i16m1x2_i16m1(zip_i16_0, 1);
  vint16m1_t zip_i16_10 = __riscv_vget_v_i16m1x2_i16m1(zip_i16_1, 0);
  vint16m1_t zip_i16_11 = __riscv_vget_v_i16m1x2_i16m1(zip_i16_1, 1);
  vint32m1x2_t zip_i32_0 =
      iree_uk_zip_8xi16_as_4xi32(zip_i16_00, zip_i16_10);
  vint32m1x2_t zip_i32_1 =
      iree_uk_zip_8xi16_as_4xi32(zip_i16_01, zip_i16_11);
  vint32m1_t zip_i32_00 = __riscv_vget_v_i32m1x2_i32m1(zip_i32_0, 0);
  vint32m1_t zip_i32_01 = __riscv_vget_v_i32m1x2_i32m1(zip_i32_0, 1);
  vint32m1_t zip_i32_10 = __riscv_vget_v_i32m1x2_i32m1(zip_i32_1, 0);
  vint32m1_t zip_i32_11 = __riscv_vget_v_i32m1x2_i32m1(zip_i32_1, 1);
  vint64m1x2_t zip_i64_0 =
      iree_uk_zip_4xi32_as_2xi64(zip_i32_00, zip_i32_10);
  vint64m1x2_t zip_i64_1 =
      iree_uk_zip_4xi32_as_2xi64(zip_i32_01, zip_i32_11);
  vint64m1_t zip_i64_00 = __riscv_vget_v_i64m1x2_i64m1(zip_i64_0, 0);
  vint64m1_t zip_i64_01 = __riscv_vget_v_i64m1x2_i64m1(zip_i64_0, 1);
  vint64m1_t zip_i64_10 = __riscv_vget_v_i64m1x2_i64m1(zip_i64_1, 0);
  vint64m1_t zip_i64_11 = __riscv_vget_v_i64m1x2_i64m1(zip_i64_1, 1);
  vint8m1_t out0 = __riscv_vreinterpret_v_i64m1_i8m1(zip_i64_00);
  vint8m1_t out1 = __riscv_vreinterpret_v_i64m1_i8m1(zip_i64_01);
  vint8m1_t out2 = __riscv_vreinterpret_v_i64m1_i8m1(zip_i64_10);
  vint8m1_t out3 = __riscv_vreinterpret_v_i64m1_i8m1(zip_i64_11);
  size_t vl = __riscv_vsetvl_e8m1(16);
  size_t half_vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t out0_lo = __riscv_vslidedown_vx_i8m1(out0, 0, half_vl);
  vint8m1_t out0_hi = __riscv_vslidedown_vx_i8m1(out0, half_vl, half_vl);
  vint8m1_t out1_lo = __riscv_vslidedown_vx_i8m1(out1, 0, half_vl);
  vint8m1_t out1_hi = __riscv_vslidedown_vx_i8m1(out1, half_vl, half_vl);
  vint8m1_t out2_lo = __riscv_vslidedown_vx_i8m1(out2, 0, half_vl);
  vint8m1_t out2_hi = __riscv_vslidedown_vx_i8m1(out2, half_vl, half_vl);
  vint8m1_t out3_lo = __riscv_vslidedown_vx_i8m1(out3, 0, half_vl);
  vint8m1_t out3_hi = __riscv_vslidedown_vx_i8m1(out3, half_vl, half_vl);
  vl = __riscv_vsetvl_e8m1(8);
  __riscv_vse8_v_i8m1(out_ptr + 0 * out_stride, out0_lo, vl);
  __riscv_vse8_v_i8m1(out_ptr + 1 * out_stride, out0_hi, vl);
  __riscv_vse8_v_i8m1(out_ptr + 2 * out_stride, out1_lo, vl);
  __riscv_vse8_v_i8m1(out_ptr + 3 * out_stride, out1_hi, vl);
  __riscv_vse8_v_i8m1(out_ptr + 4 * out_stride, out2_lo, vl);
  __riscv_vse8_v_i8m1(out_ptr + 5 * out_stride, out2_hi, vl);
  __riscv_vse8_v_i8m1(out_ptr + 6 * out_stride, out3_lo, vl);
  __riscv_vse8_v_i8m1(out_ptr + 7 * out_stride, out3_hi, vl);
}

static inline void iree_uk_copy_8x8xi8_transpose_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  // Clang (Android NDK r25) actually produces worse code when this code is
  // specialized for out_stride==8 using longer contiguous stores!
  iree_uk_copy_8x8xi8_transpose_strided_to_strided(out_ptr, in_ptr, 8,
                                                        in_stride);
}

#endif  // IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_GENERAL_RISCV_64_H_
