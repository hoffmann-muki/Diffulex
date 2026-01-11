#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ O, bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ V, int* __restrict__ cu_seqlens_k, int* __restrict__ cu_seqlens_q, int max_seqlen_q);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ O, bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ V, int* __restrict__ cu_seqlens_k, int* __restrict__ cu_seqlens_q, int max_seqlen_q) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_output[128];
  float acc_score[256];
  float log_sum[4];
  float scores_max[4];
  float scores_max_prev[4];
  float scores_scale[4];
  float scores_sum[4];
  bfloat16_t acc_score_cast[256];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    uint2 condval;
    if (((((0 <= ((((((int)blockIdx.x) * 256) + (i * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0])) && (((((int)blockIdx.x) * 4) + ((((i * 8) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0]) >> 6)) < 1)) && (((((int)blockIdx.x) * 4) + ((((i * 8) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0]) >> 6)) < 1)) && (0 <= ((((((int)blockIdx.x) * 256) + (i * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0])))) {
      condval = *(uint2*)(Q + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)917504) + (((int64_t)i) * (int64_t)28672)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)3584)) + (((int64_t)cu_seqlens_q[(int64_t)0]) * (int64_t)3584)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)));
    } else {
      condval = make_uint2(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 31) >> 4) * 16384) + (i * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 65536)) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 64; ++i_1) {
    *(float2*)(acc_output + (i_1 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 128; ++i_2) {
    *(float2*)(acc_score + (i_2 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    log_sum[i_3] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    scores_max[i_4] = -CUDART_INF_F;
  }
  if ((0 < ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (0 < (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 31) >> 4) * 32768) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 5) * 128)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)), K+(((((((int64_t)i_5) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)), ((((0 <= (((i_5 * 8) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && (((((((int)threadIdx.x) >> 5) + cu_seqlens_k[0]) >> 3) + i_5) < 8)) && (((((((int)threadIdx.x) >> 5) + cu_seqlens_k[0]) >> 3) + i_5) < 8)) && (0 <= (((i_5 * 8) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
  }
  if ((0 < ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (0 < (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 32; ++i_6) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((((((((((int)threadIdx.x) & 31) >> 4) * 32768) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 5) * 128)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 65536), V+(((((((int64_t)i_6) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)), ((((0 <= (((i_6 * 8) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && (((((((int)threadIdx.x) >> 5) + cu_seqlens_k[0]) >> 3) + i_6) < 8)) && (((((((int)threadIdx.x) >> 5) + cu_seqlens_k[0]) >> 3) + i_6) < 8)) && (0 <= (((i_6 * 8) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
  }
  for (int kv_block_idx = 0; kv_block_idx < (min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) - 1); ++kv_block_idx) {
    #pragma unroll
    for (int i_7 = 0; i_7 < 256; ++i_7) {
      float condval_1;
      if ((((((((((int)blockIdx.x) * 256) + (((i_7 & 7) >> 2) * 128)) + ((((int)threadIdx.x) >> 6) * 32)) + 32) <= ((((kv_block_idx * 256) + ((i_7 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))) || ((cu_seqlens_q[1] - cu_seqlens_q[0]) <= (((((((int)blockIdx.x) * 256) + (((i_7 & 7) >> 2) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) || ((cu_seqlens_k[1] - cu_seqlens_k[0]) <= ((((kv_block_idx * 256) + ((i_7 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))))) {
        condval_1 = -0x1.dcd65p+29f/*-1.000000e+09*/;
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      acc_score[i_7] = condval_1;
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    tl::gemm_ss<256, 256, 128, 8, 1, 0, 1, 0, 128, 128, 0, 0>((&(((bfloat16_t*)buf_dyn_shmem)[65536])), (&(((bfloat16_t*)buf_dyn_shmem)[0])), (&(acc_score[0])));
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 32; ++i_8) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 31) >> 4) * 32768) + (i_8 * 1024)) + ((((int)threadIdx.x) >> 5) * 128)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)), K+(((((((((int64_t)kv_block_idx) * (int64_t)131072) + (((int64_t)i_8) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)) + (int64_t)131072), ((((-256 <= ((((kv_block_idx * 256) + (i_8 * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && (((((kv_block_idx * 256) + (i_8 * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -192)) && (((((kv_block_idx * 256) + (i_8 * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -192)) && (-256 <= ((((kv_block_idx * 256) + (i_8 * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      scores_max_prev[i_9] = scores_max[i_9];
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      scores_max[i_10] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      #pragma unroll
      for (int rv = 0; rv < 64; ++rv) {
        scores_max[i_11] = max(scores_max[i_11], acc_score[((((rv & 31) * 8) + (i_11 * 2)) + (rv >> 5))]);
      }
      scores_max[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      scores_max[i_12] = max(scores_max[i_12], scores_max_prev[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      scores_scale[i_13] = exp2f(((scores_max_prev[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 256; ++i_14) {
      acc_score[i_14] = exp2f(((acc_score[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_14 & 7) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      scores_sum[i_15] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 64; ++rv_1) {
        scores_sum[i_15] = (scores_sum[i_15] + acc_score[((((rv_1 & 31) * 8) + (i_15 * 2)) + (rv_1 >> 5))]);
      }
      scores_sum[i_15] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 4; ++i_16) {
      log_sum[i_16] = ((log_sum[i_16] * scores_scale[i_16]) + scores_sum[i_16]);
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 128; ++i_17) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_score + (((((i_17 >> 3) * 16) + (((i_17 & 3) >> 1) * 8)) + (((i_17 & 7) >> 2) * 4)) + ((i_17 & 1) * 2)));
#ifdef ENABLE_BF16
      reinterpret_cast<__nv_bfloat162 &>(__1) = fastertransformer::float22bf162(reinterpret_cast<float2 const &>(v_));
#else
      ((nv_bfloat162*)(&(__1.x)))->x = (bfloat16_t)(v_.x);
      ((nv_bfloat162*)(&(__1.x)))->y = (bfloat16_t)(v_.y);
#endif
      *(uint1*)(acc_score_cast + (i_17 * 2)) = __1;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 128; ++i_18) {
      acc_output[i_18] = (acc_output[i_18] * scores_scale[((i_18 & 7) >> 1)]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    tl::gemm_rs<256, 128, 256, 8, 1, 0, 0, 0, 256, 128, 0, 0>((&(acc_score_cast[0])), (&(((bfloat16_t*)buf_dyn_shmem)[32768])), (&(acc_output[0])));
    __syncthreads();
    #pragma unroll
    for (int i_19 = 0; i_19 < 32; ++i_19) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((((((((((int)threadIdx.x) & 31) >> 4) * 32768) + (i_19 * 1024)) + ((((int)threadIdx.x) >> 5) * 128)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 65536), V+(((((((((int64_t)kv_block_idx) * (int64_t)131072) + (((int64_t)i_19) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)) + (int64_t)131072), ((((-256 <= ((((kv_block_idx * 256) + (i_19 * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && (((((kv_block_idx * 256) + (i_19 * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -192)) && (((((kv_block_idx * 256) + (i_19 * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -192)) && (-256 <= ((((kv_block_idx * 256) + (i_19 * 8)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_20 = 0; i_20 < 256; ++i_20) {
      float condval_2;
      if ((((36 <= (((((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) * 32) + (i_20 >> 3)) - ((((int)threadIdx.x) >> 6) * 4)) - (((i_20 & 7) >> 2) * 16)) - (((int)blockIdx.x) * 32))) || ((cu_seqlens_q[1] - cu_seqlens_q[0]) <= (((((((int)blockIdx.x) * 256) + (((i_20 & 7) >> 2) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_20 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) || (((cu_seqlens_k[1] + 256) - cu_seqlens_k[0]) <= ((((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) * 256) + ((i_20 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_20 & 1))))) {
        condval_2 = -0x1.dcd65p+29f/*-1.000000e+09*/;
      } else {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      }
      acc_score[i_20] = condval_2;
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    tl::cp_async_wait<1>();
    __syncthreads();
    tl::gemm_ss<256, 256, 128, 8, 1, 0, 1, 0, 128, 128, 0, 0>((&(((bfloat16_t*)buf_dyn_shmem)[65536])), (&(((bfloat16_t*)buf_dyn_shmem)[0])), (&(acc_score[0])));
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_21 = 0; i_21 < 4; ++i_21) {
      scores_max_prev[i_21] = scores_max[i_21];
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_22 = 0; i_22 < 4; ++i_22) {
      scores_max[i_22] = -CUDART_INF_F;
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_23 = 0; i_23 < 4; ++i_23) {
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 64; ++rv_2) {
        scores_max[i_23] = max(scores_max[i_23], acc_score[((((rv_2 & 31) * 8) + (i_23 * 2)) + (rv_2 >> 5))]);
      }
      scores_max[i_23] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_23]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_24 = 0; i_24 < 4; ++i_24) {
      scores_max[i_24] = max(scores_max[i_24], scores_max_prev[i_24]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_25 = 0; i_25 < 4; ++i_25) {
      scores_scale[i_25] = exp2f(((scores_max_prev[i_25] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_25] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_26 = 0; i_26 < 256; ++i_26) {
      acc_score[i_26] = exp2f(((acc_score[i_26] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_26 & 7) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_27 = 0; i_27 < 4; ++i_27) {
      scores_sum[i_27] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 64; ++rv_3) {
        scores_sum[i_27] = (scores_sum[i_27] + acc_score[((((rv_3 & 31) * 8) + (i_27 * 2)) + (rv_3 >> 5))]);
      }
      scores_sum[i_27] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_27]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_28 = 0; i_28 < 4; ++i_28) {
      log_sum[i_28] = ((log_sum[i_28] * scores_scale[i_28]) + scores_sum[i_28]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_29 = 0; i_29 < 128; ++i_29) {
      uint1 __2;
      float2 v__1 = *(float2*)(acc_score + (((((i_29 >> 3) * 16) + (((i_29 & 3) >> 1) * 8)) + (((i_29 & 7) >> 2) * 4)) + ((i_29 & 1) * 2)));
#ifdef ENABLE_BF16
      reinterpret_cast<__nv_bfloat162 &>(__2) = fastertransformer::float22bf162(reinterpret_cast<float2 const &>(v__1));
#else
      ((nv_bfloat162*)(&(__2.x)))->x = (bfloat16_t)(v__1.x);
      ((nv_bfloat162*)(&(__2.x)))->y = (bfloat16_t)(v__1.y);
#endif
      *(uint1*)(acc_score_cast + (i_29 * 2)) = __2;
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 128; ++i_30) {
      acc_output[i_30] = (acc_output[i_30] * scores_scale[((i_30 & 7) >> 1)]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    tl::cp_async_wait<0>();
    __syncthreads();
    tl::gemm_rs<256, 128, 256, 8, 1, 0, 0, 0, 256, 128, 0, 0>((&(acc_score_cast[0])), (&(((bfloat16_t*)buf_dyn_shmem)[32768])), (&(acc_output[0])));
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 128; ++i_31) {
    acc_output[i_31] = (acc_output[i_31] / log_sum[((i_31 & 7) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_32 = 0; i_32 < 64; ++i_32) {
    uint1 __3;
    float2 v__2 = *(float2*)(acc_output + (i_32 * 2));
#ifdef ENABLE_BF16
    reinterpret_cast<__nv_bfloat162 &>(__3) = fastertransformer::float22bf162(reinterpret_cast<float2 const &>(v__2));
#else
    ((nv_bfloat162*)(&(__3.x)))->x = (bfloat16_t)(v__2.x);
    ((nv_bfloat162*)(&(__3.x)))->y = (bfloat16_t)(v__2.y);
#endif
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((i_32 >> 5) * 16384) + (((i_32 & 3) >> 1) * 8192)) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_32 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((i_32 & 31) >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_32 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_32 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 65536)) = __3;
  }
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 128; ++i_33) {
    if ((((((int)blockIdx.x) * 256) + (i_33 * 2)) + (((int)threadIdx.x) >> 7)) < (cu_seqlens_q[1] - cu_seqlens_q[0])) {
      if (0 <= ((((((int)blockIdx.x) * 256) + (i_33 * 2)) + (((int)threadIdx.x) >> 7)) + cu_seqlens_q[0])) {
        if (((((int)blockIdx.x) * 4) + ((((i_33 * 2) + (((int)threadIdx.x) >> 7)) + cu_seqlens_q[0]) >> 6)) < 1) {
          O[((((((((int64_t)((int)blockIdx.x)) * (int64_t)917504) + (((int64_t)i_33) * (int64_t)7168)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)3584)) + (((int64_t)cu_seqlens_q[(int64_t)0]) * (int64_t)3584)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) & (int64_t)127))] = ((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 127) >> 6) * 16384) + (i_33 * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_33 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_33 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 65536)];
        }
      }
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_kernel_kernel = cudaFuncSetAttribute(kernel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 196608);
    if (result_kernel_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 196608, cudaGetErrorString(result_kernel_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ V, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int max_seqlen_q, bfloat16_t* __restrict__ O, cudaStream_t stream=cudaStreamDefault) {
	kernel_kernel<<<dim3((max_seqlen_q + 255) / 256, 28, 1), dim3(256, 1, 1), 196608, stream>>>(K, O, Q, V, cu_seqlens_k, cu_seqlens_q, max_seqlen_q);
	TILELANG_CHECK_LAST_ERROR("kernel_kernel");

	return 0;
}
