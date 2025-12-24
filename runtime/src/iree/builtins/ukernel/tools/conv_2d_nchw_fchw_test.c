// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//#include <riscv_vector.h>

#include "iree/base/api.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/conv_2d_nchw_fchw_internal.h"
#include "iree/builtins/ukernel/tools/test.h"
#include "iree/builtins/ukernel/tools/util.h"

static void iree_conv_2d_nchw_fchw_generic_reference(
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
      //size_t vl = __riscv_vsetvl_e32m1(filter_size_w);
      for (iree_uk_index_t kw = 0; kw < filter_size_w; kw++) {
        //vl = __riscv_vsetvl_e32m1(filter_size_w - kw);
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
	  //vfloat32m1_t v0 = __riscv_vle32_v_f32m1(in_ptr, vl);
          //vfloat32m1_t v1 = __riscv_vle32_v_f32m1(filter_ptr, vl);
          //vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(v0, v1, vl);
          //sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vprod, __riscv_vfmv_s_f_f32m1(0.0, vl), vl));
        }
      }
    }
  }
  *out_ptr = sum;
}

static void iree_conv_2d_nchw_fchw_reference(
    iree_uk_test_t* test, const iree_uk_conv_params_t* params) {
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
    char msg[128];
    snprintf(msg, sizeof msg, "(%lld,%lld) %lld (%lld,%lld) %f %f", ih,iw, filter_idx, oh,ow, *in_ptr, *filter_ptr);
    iree_uk_test_log_info(test, "🦕", msg);
                }
              }
            }
          }
    char msg[128];
    snprintf(msg, sizeof msg, "%f", sum);
    iree_uk_test_log_info(test, "🦕", msg);
          *out_ptr = sum;*/
          char* out_ptr = (char*)params->out_buffer + out_idx * out_type_size;
          iree_conv_2d_nchw_fchw_generic_reference(out_ptr, params->in_buffer, params->filter_buffer,
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
        }
      }
    }
  }
}

static void iree_uk_test_conv_2d_nchw_fchw_for_shape_params(
    iree_uk_test_t* test, const iree_uk_conv_params_t* src_params) {
  iree_uk_conv_params_t params;
  memcpy(&params, src_params, sizeof params);
  
  iree_uk_conv_type_t conv_type = iree_uk_conv_type(params.flags);
  iree_uk_type_t in_type = iree_uk_conv_in_type(conv_type);
  iree_uk_type_t filter_type = iree_uk_conv_filter_type(conv_type);
  iree_uk_type_t out_type = iree_uk_conv_out_type(conv_type);
  
  //iree_uk_random_engine_t* engine = iree_uk_test_random_engine(test);
  
  params.in_stride0 = params.in_size_c * params.in_size_h * params.in_size_w;
  params.in_stride1 = params.in_size_h * params.in_size_w;
  //params.in_stride_h = params.in_size_w;
  //params.in_stride_w = 1;
  
  params.filter_stride0 = params.in_size_c * params.filter_size_h * params.filter_size_w;
  params.filter_stride1 = params.filter_size_h * params.filter_size_w;
  //params.filter_stride_h = params.filter_size_w;
  //params.filter_stride_w = 1;
  
  params.out_stride0 = params.out_size_c * params.out_size_h * params.out_size_w;
  params.out_stride1 = params.out_size_h * params.out_size_w;
  //params.out_stride_h = params.out_size_w;
  //params.out_stride_w = 1;
  
  iree_uk_index_t in_buffer_size = 
      params.out_size_n * params.in_stride0 * iree_uk_type_size(in_type);
  iree_uk_index_t filter_buffer_size = 
      params.out_size_c * params.filter_stride0 * iree_uk_type_size(filter_type);
  iree_uk_index_t out_buffer_size = 
      params.out_size_n * params.out_stride0 * iree_uk_type_size(out_type);
  
  void* in_buffer = malloc(in_buffer_size);
  memset(in_buffer, 0, in_buffer_size);
  ((float*)(in_buffer))[0] = 1.0;
  ((float*)(in_buffer))[34] = 2.0;
  void* filter_buffer = malloc(filter_buffer_size);
  for (iree_uk_index_t i = 0; i < filter_buffer_size / sizeof(float); ++i) {
    ((float*)filter_buffer)[i] = 1.0;
  }
  ((float*)(filter_buffer))[8] = 3.0;
  //iree_uk_write_random_buffer(in_buffer, in_buffer_size, in_type, engine);
  //iree_uk_write_random_buffer(filter_buffer, filter_buffer_size, filter_type, engine);
  
  params.in_buffer = in_buffer;
  params.filter_buffer = filter_buffer;
  
  //int random_val = iree_uk_random_engine_get_0_65535(engine);
  params.in_offset = 0;
  params.filter_offset = 0;
  params.out_offset = 0;
  
  iree_uk_conv_params_t reference_params;
  memcpy(&reference_params, &params, sizeof params);
  
  void* reference_out_buffer = malloc(out_buffer_size);
  //iree_uk_write_random_buffer(reference_out_buffer, out_buffer_size, out_type, engine);
  memset(reference_out_buffer, 0, out_buffer_size);
  reference_params.out_buffer = reference_out_buffer;
  
  iree_uk_conv_params_t actual_params;
  memcpy(&actual_params, &params, sizeof params);
  
  void* actual_out_buffer = malloc(out_buffer_size);
  memcpy(actual_out_buffer, reference_out_buffer, out_buffer_size);
  actual_params.out_buffer = actual_out_buffer;
  
  iree_conv_2d_nchw_fchw_reference(test, &reference_params);
  iree_uk_conv_p(&actual_params);
  for (iree_uk_index_t i = 0; i < 16; ++i) {
    char msg[128];
    snprintf(msg, sizeof msg, "out_buffer[%lld]: %f",
             i, ((float*)(reference_params.out_buffer))[i]);
    iree_uk_test_log_info(test, "🦕", msg);
  }
  for (iree_uk_index_t i = 0; i < 16; ++i) {
    char msg[128];
    snprintf(msg, sizeof msg, "out_buffer[%lld]: %f",
             i, ((float*)(actual_params.out_buffer))[i]);
    iree_uk_test_log_info(test, "🦕", msg);
  }

  bool fail = memcmp(actual_out_buffer, reference_out_buffer, out_buffer_size);
  if (fail) {
    IREE_UK_TEST_FAIL(test);
  }
  
  free(reference_out_buffer);
  free(actual_out_buffer);
  free(in_buffer);
  free(filter_buffer);
}

static void iree_uk_test_conv_2d_nchw_fchw_for_tile_params(iree_uk_test_t* test,
                                              const void* src_params) {
  typedef struct outer_shape_t {
    int size0, size1;
  } outer_shape_t;
  const outer_shape_t outer_shapes[] = {
      {4, 4},
  };
  typedef enum {
    pad_none,
    pad_enum_end
  } pad_t;
  for (int i = 0; i < IREE_ARRAYSIZE(outer_shapes); ++i) {
    for (int transpose_inner = 0; transpose_inner < 1; ++transpose_inner) {
      for (int transpose_outer = 0; transpose_outer < 1; ++transpose_outer) {
        for (pad_t pad = 0; pad < pad_enum_end; ++pad) {
          iree_uk_conv_params_t params;
          memcpy(&params, src_params, sizeof params);
          params.cpu_data = iree_uk_test_cpu_data(test);
	  params.out_size_n = 1;
	  params.in_size_c = 1;
	  params.out_size_c = 1;
          outer_shape_t outer_shape = outer_shapes[i];
          params.out_size_h = outer_shape.size0;
          params.out_size_w = outer_shape.size1;
          iree_uk_index_t tile_size0 = params.tile_size0;
          iree_uk_index_t tile_size1 = params.tile_size1;
          params.in_size_h = outer_shape.size0 + tile_size0 - 1;
          params.in_size_w = outer_shape.size1 + tile_size1 - 1;
          iree_uk_test_conv_2d_nchw_fchw_for_shape_params(test, &params);
        }
      }
    }
  }
}

static void iree_uk_test_conv_2d_nchw_fchw(iree_uk_uint32_t flags, int tile_size0,
                              int tile_size1, const char* cpu_features) {
  iree_uk_conv_params_t params = {
      .flags = flags, .filter_size_h = tile_size0, .filter_size_w = tile_size1, .tile_size0 = tile_size0, .tile_size1 = tile_size1};
  char types_str[32];
  iree_uk_conv_type_t type = iree_uk_conv_type(flags);
  iree_uk_type_pair_str(types_str, sizeof types_str, type);
  char test_label_str[256];
  snprintf(test_label_str, sizeof test_label_str, "types:%s tile:%dx%d",
           types_str, tile_size0, tile_size1);
  iree_uk_test(test_label_str, iree_uk_test_conv_2d_nchw_fchw_for_tile_params, &params,
               cpu_features);
}

int main(int argc, char** argv) {
  // Generic tests, not matching any particular CPU feature. This is the place
  // to test weird tile shapes to ensure e.g. that we haven't unwittingly baked
  // in a power-of-two assumption
  iree_uk_test_conv_2d_nchw_fchw(IREE_UK_FLAG_CONV_TYPE_F32F32F32, 3, 3, "");
  //iree_uk_test_conv_2d_nchw_fchw(IREE_UK_FLAG_CONV_TYPE_I8I8, 4, 2, "");
  //iree_uk_test_conv_2d_nchw_fchw(IREE_UK_FLAG_CONV_TYPE_I32I32, 3, 4, "");
  //iree_uk_test_conv_2d_nchw_fchw(IREE_UK_FLAG_CONV_TYPE_F16F16, 6, 7, "");
  //iree_uk_test_conv_2d_nchw_fchw(IREE_UK_FLAG_CONV_TYPE_BF16BF16, 9, 2, "");

  return iree_uk_test_exit_status();
}
