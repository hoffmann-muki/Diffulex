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
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ O, bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ V, int* __restrict__ cu_seqlens_k, int* __restrict__ cu_seqlens_q, int max_seqlen_q) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_output[256];
  float acc_score[512];
  float log_sum[8];
  float scores_max[8];
  float scores_max_prev[8];
  float scores_scale[8];
  float scores_sum[8];
  bfloat16_t acc_score_cast[512];
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint2 condval;
    if (((((0 <= ((((((int)blockIdx.x) * 256) + (i * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0])) && (((((int)blockIdx.x) * 4) + ((((i * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0]) >> 6)) < 1)) && (((((int)blockIdx.x) * 4) + ((((i * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0]) >> 6)) < 1)) && (0 <= ((((((int)blockIdx.x) * 256) + (i * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0])))) {
      condval = *(uint2*)(Q + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)917504) + (((int64_t)i) * (int64_t)14336)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)3584)) + (((int64_t)cu_seqlens_q[(int64_t)0]) * (int64_t)3584)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)));
    } else {
      condval = make_uint2(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 31) >> 4) * 16384) + (i * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 15) >> 3) + (i & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 131072)) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 128; ++i_1) {
    *(float2*)(acc_output + (i_1 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 256; ++i_2) {
    *(float2*)(acc_score + (i_2 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    log_sum[i_3] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    scores_max[i_4] = -CUDART_INF_F;
  }
  if ((0 < ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (0 < (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_5 = 0; i_5 < 64; ++i_5) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 31) >> 4) * 32768) + (i_5 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_5 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)), K+(((((((int64_t)i_5) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)), ((((0 <= (((i_5 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && ((((i_5 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < 64)) && ((((i_5 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < 64)) && (0 <= (((i_5 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
  }
  if ((0 < ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (0 < (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 64; ++i_6) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((((((((((int)threadIdx.x) & 31) >> 4) * 32768) + (i_6 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_6 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 131072), V+(((((((int64_t)i_6) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)), ((((0 <= (((i_6 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && ((((i_6 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < 64)) && ((((i_6 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < 64)) && (0 <= (((i_6 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
  }
  if ((1 < ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (256 < (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_7 = 0; i_7 < 64; ++i_7) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((((((((((int)threadIdx.x) & 31) >> 4) * 32768) + (i_7 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_7 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 65536), K+((((((((int64_t)i_7) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)) + (int64_t)131072), ((((-256 <= (((i_7 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && ((((i_7 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -192)) && ((((i_7 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -192)) && (-256 <= (((i_7 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
  }
  if ((1 < ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (256 < (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_8 = 0; i_8 < 64; ++i_8) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((((((((((int)threadIdx.x) & 31) >> 4) * 32768) + (i_8 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_8 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 196608), V+((((((((int64_t)i_8) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)) + (int64_t)131072), ((((-256 <= (((i_8 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && ((((i_8 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -192)) && ((((i_8 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -192)) && (-256 <= (((i_8 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
  }
  for (int kv_block_idx = 0; kv_block_idx < (min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) - 2); ++kv_block_idx) {
    #pragma unroll
    for (int i_9 = 0; i_9 < 512; ++i_9) {
      float condval_1;
      if ((((((((((int)blockIdx.x) * 256) + (((i_9 & 15) >> 2) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 32) <= ((((kv_block_idx * 256) + ((i_9 >> 4) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_9 & 1))) || ((cu_seqlens_q[1] - cu_seqlens_q[0]) <= (((((((int)blockIdx.x) * 256) + (((i_9 & 15) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_9 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) || ((cu_seqlens_k[1] - cu_seqlens_k[0]) <= ((((kv_block_idx * 256) + ((i_9 >> 4) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_9 & 1))))) {
        condval_1 = -0x1.dcd65p+29f/*-1.000000e+09*/;
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      acc_score[i_9] = condval_1;
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    tl::gemm_ss<256, 256, 128, 4, 1, 0, 1, 0, 128, 128, 0, 0>((&(((bfloat16_t*)buf_dyn_shmem)[131072])), (&(((bfloat16_t*)buf_dyn_shmem)[((kv_block_idx & 1) * 32768)])), (&(acc_score[0])));
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+(((((((((kv_block_idx & 1) * 65536) + (((((int)threadIdx.x) & 31) >> 4) * 32768)) + (i_10 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_10 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)), K+(((((((((int64_t)kv_block_idx) * (int64_t)131072) + (((int64_t)i_10) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)) + (int64_t)262144), ((((-512 <= ((((kv_block_idx * 256) + (i_10 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && (((((kv_block_idx * 256) + (i_10 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -448)) && (((((kv_block_idx * 256) + (i_10 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -448)) && (-512 <= ((((kv_block_idx * 256) + (i_10 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      scores_max_prev[i_11] = scores_max[i_11];
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 8; ++i_12) {
      scores_max[i_12] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      #pragma unroll
      for (int rv = 0; rv < 64; ++rv) {
        scores_max[i_13] = max(scores_max[i_13], acc_score[((((rv & 31) * 16) + (i_13 * 2)) + (rv >> 5))]);
      }
      scores_max[i_13] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 8; ++i_14) {
      scores_max[i_14] = max(scores_max[i_14], scores_max_prev[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 8; ++i_15) {
      scores_scale[i_15] = exp2f(((scores_max_prev[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 512; ++i_16) {
      acc_score[i_16] = exp2f(((acc_score[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_16 & 15) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 8; ++i_17) {
      scores_sum[i_17] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 64; ++rv_1) {
        scores_sum[i_17] = (scores_sum[i_17] + acc_score[((((rv_1 & 31) * 16) + (i_17 * 2)) + (rv_1 >> 5))]);
      }
      scores_sum[i_17] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      log_sum[i_18] = ((log_sum[i_18] * scores_scale[i_18]) + scores_sum[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 256; ++i_19) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_score + (((((i_19 >> 4) * 32) + (((i_19 & 3) >> 1) * 16)) + (((i_19 & 15) >> 2) * 4)) + ((i_19 & 1) * 2)));
#ifdef ENABLE_BF16
      reinterpret_cast<__nv_bfloat162 &>(__1) = fastertransformer::float22bf162(reinterpret_cast<float2 const &>(v_));
#else
      ((nv_bfloat162*)(&(__1.x)))->x = (bfloat16_t)(v_.x);
      ((nv_bfloat162*)(&(__1.x)))->y = (bfloat16_t)(v_.y);
#endif
      *(uint1*)(acc_score_cast + (i_19 * 2)) = __1;
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 256; ++i_20) {
      acc_output[i_20] = (acc_output[i_20] * scores_scale[((i_20 & 15) >> 1)]);
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    tl::gemm_rs<256, 128, 256, 4, 1, 0, 0, 0, 256, 128, 0, 0>((&(acc_score_cast[0])), (&(((bfloat16_t*)buf_dyn_shmem)[(((kv_block_idx & 1) * 32768) + 65536)])), (&(acc_output[0])));
    __syncthreads();
    #pragma unroll
    for (int i_21 = 0; i_21 < 64; ++i_21) {
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((((((((kv_block_idx & 1) * 65536) + (((((int)threadIdx.x) & 31) >> 4) * 32768)) + (i_21 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_21 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 131072), V+(((((((((int64_t)kv_block_idx) * (int64_t)131072) + (((int64_t)i_21) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)) + (int64_t)262144), ((((-512 <= ((((kv_block_idx * 256) + (i_21 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && (((((kv_block_idx * 256) + (i_21 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -448)) && (((((kv_block_idx * 256) + (i_21 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) < -448)) && (-512 <= ((((kv_block_idx * 256) + (i_21 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]))));
    }
    tl::cp_async_commit();
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_22 = 0; i_22 < 512; ++i_22) {
      float condval_2;
      if ((((68 <= (((((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) * 32) + (i_22 >> 4)) - ((((int)threadIdx.x) >> 6) * 4)) - (((i_22 & 15) >> 2) * 8)) - (((int)blockIdx.x) * 32))) || ((cu_seqlens_q[1] - cu_seqlens_q[0]) <= (((((((int)blockIdx.x) * 256) + (((i_22 & 15) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_22 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) || (((cu_seqlens_k[1] + 512) - cu_seqlens_k[0]) <= ((((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) * 256) + ((i_22 >> 4) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_22 & 1))))) {
        condval_2 = -0x1.dcd65p+29f/*-1.000000e+09*/;
      } else {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      }
      acc_score[i_22] = condval_2;
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    tl::cp_async_wait<3>();
    __syncthreads();
    tl::gemm_ss<256, 256, 128, 4, 1, 0, 1, 0, 128, 128, 0, 0>((&(((bfloat16_t*)buf_dyn_shmem)[131072])), (&(((bfloat16_t*)buf_dyn_shmem)[((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) & 1) * 32768)])), (&(acc_score[0])));
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_23 = 0; i_23 < 8; ++i_23) {
      scores_max_prev[i_23] = scores_max[i_23];
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_24 = 0; i_24 < 8; ++i_24) {
      scores_max[i_24] = -CUDART_INF_F;
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_25 = 0; i_25 < 8; ++i_25) {
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 64; ++rv_2) {
        scores_max[i_25] = max(scores_max[i_25], acc_score[((((rv_2 & 31) * 16) + (i_25 * 2)) + (rv_2 >> 5))]);
      }
      scores_max[i_25] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_25]);
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_26 = 0; i_26 < 8; ++i_26) {
      scores_max[i_26] = max(scores_max[i_26], scores_max_prev[i_26]);
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_27 = 0; i_27 < 8; ++i_27) {
      scores_scale[i_27] = exp2f(((scores_max_prev[i_27] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_27] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_28 = 0; i_28 < 512; ++i_28) {
      acc_score[i_28] = exp2f(((acc_score[i_28] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_28 & 15) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_29 = 0; i_29 < 8; ++i_29) {
      scores_sum[i_29] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 64; ++rv_3) {
        scores_sum[i_29] = (scores_sum[i_29] + acc_score[((((rv_3 & 31) * 16) + (i_29 * 2)) + (rv_3 >> 5))]);
      }
      scores_sum[i_29] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_29]);
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 8; ++i_30) {
      log_sum[i_30] = ((log_sum[i_30] * scores_scale[i_30]) + scores_sum[i_30]);
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_31 = 0; i_31 < 256; ++i_31) {
      uint1 __2;
      float2 v__1 = *(float2*)(acc_score + (((((i_31 >> 4) * 32) + (((i_31 & 3) >> 1) * 16)) + (((i_31 & 15) >> 2) * 4)) + ((i_31 & 1) * 2)));
#ifdef ENABLE_BF16
      reinterpret_cast<__nv_bfloat162 &>(__2) = fastertransformer::float22bf162(reinterpret_cast<float2 const &>(v__1));
#else
      ((nv_bfloat162*)(&(__2.x)))->x = (bfloat16_t)(v__1.x);
      ((nv_bfloat162*)(&(__2.x)))->y = (bfloat16_t)(v__1.y);
#endif
      *(uint1*)(acc_score_cast + (i_31 * 2)) = __2;
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_32 = 0; i_32 < 256; ++i_32) {
      acc_output[i_32] = (acc_output[i_32] * scores_scale[((i_32 & 15) >> 1)]);
    }
  }
  if ((2 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (257 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    tl::cp_async_wait<3>();
    __syncthreads();
    tl::gemm_rs<256, 128, 256, 4, 1, 0, 0, 0, 256, 128, 0, 0>((&(acc_score_cast[0])), (&(((bfloat16_t*)buf_dyn_shmem)[(((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) & 1) * 32768) + 65536)])), (&(acc_output[0])));
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_33 = 0; i_33 < 512; ++i_33) {
      float condval_3;
      if ((((36 <= (((((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) * 32) + (i_33 >> 4)) - ((((int)threadIdx.x) >> 6) * 4)) - (((i_33 & 15) >> 2) * 8)) - (((int)blockIdx.x) * 32))) || ((cu_seqlens_q[1] - cu_seqlens_q[0]) <= (((((((int)blockIdx.x) * 256) + (((i_33 & 15) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_33 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) || (((cu_seqlens_k[1] + 256) - cu_seqlens_k[0]) <= ((((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) * 256) + ((i_33 >> 4) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_33 & 1))))) {
        condval_3 = -0x1.dcd65p+29f/*-1.000000e+09*/;
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      acc_score[i_33] = condval_3;
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    tl::cp_async_wait<3>();
    __syncthreads();
    tl::gemm_ss<256, 256, 128, 4, 1, 0, 1, 0, 128, 128, 0, 0>((&(((bfloat16_t*)buf_dyn_shmem)[131072])), (&(((bfloat16_t*)buf_dyn_shmem)[(((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) + 1) & 1) * 32768)])), (&(acc_score[0])));
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_34 = 0; i_34 < 8; ++i_34) {
      scores_max_prev[i_34] = scores_max[i_34];
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_35 = 0; i_35 < 8; ++i_35) {
      scores_max[i_35] = -CUDART_INF_F;
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_36 = 0; i_36 < 8; ++i_36) {
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 64; ++rv_4) {
        scores_max[i_36] = max(scores_max[i_36], acc_score[((((rv_4 & 31) * 16) + (i_36 * 2)) + (rv_4 >> 5))]);
      }
      scores_max[i_36] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_36]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_37 = 0; i_37 < 8; ++i_37) {
      scores_max[i_37] = max(scores_max[i_37], scores_max_prev[i_37]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_38 = 0; i_38 < 8; ++i_38) {
      scores_scale[i_38] = exp2f(((scores_max_prev[i_38] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_38] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_39 = 0; i_39 < 512; ++i_39) {
      acc_score[i_39] = exp2f(((acc_score[i_39] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_39 & 15) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_40 = 0; i_40 < 8; ++i_40) {
      scores_sum[i_40] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 64; ++rv_5) {
        scores_sum[i_40] = (scores_sum[i_40] + acc_score[((((rv_5 & 31) * 16) + (i_40 * 2)) + (rv_5 >> 5))]);
      }
      scores_sum[i_40] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_40]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_41 = 0; i_41 < 8; ++i_41) {
      log_sum[i_41] = ((log_sum[i_41] * scores_scale[i_41]) + scores_sum[i_41]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_42 = 0; i_42 < 256; ++i_42) {
      uint1 __3;
      float2 v__2 = *(float2*)(acc_score + (((((i_42 >> 4) * 32) + (((i_42 & 3) >> 1) * 16)) + (((i_42 & 15) >> 2) * 4)) + ((i_42 & 1) * 2)));
#ifdef ENABLE_BF16
      reinterpret_cast<__nv_bfloat162 &>(__3) = fastertransformer::float22bf162(reinterpret_cast<float2 const &>(v__2));
#else
      ((nv_bfloat162*)(&(__3.x)))->x = (bfloat16_t)(v__2.x);
      ((nv_bfloat162*)(&(__3.x)))->y = (bfloat16_t)(v__2.y);
#endif
      *(uint1*)(acc_score_cast + (i_42 * 2)) = __3;
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    #pragma unroll
    for (int i_43 = 0; i_43 < 256; ++i_43) {
      acc_output[i_43] = (acc_output[i_43] * scores_scale[((i_43 & 15) >> 1)]);
    }
  }
  if ((1 <= ((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x))) && (1 <= (cu_seqlens_k[1] - cu_seqlens_k[0]))) {
    tl::cp_async_wait<3>();
    __syncthreads();
    tl::gemm_rs<256, 128, 256, 4, 1, 0, 0, 0, 256, 128, 0, 0>((&(acc_score_cast[0])), (&(((bfloat16_t*)buf_dyn_shmem)[((((min(((((cu_seqlens_q[1] + 511) - cu_seqlens_q[0]) >> 8) + ((int)blockIdx.x)), (((cu_seqlens_k[1] + 255) - cu_seqlens_k[0]) >> 8)) + 1) & 1) * 32768) + 65536)])), (&(acc_output[0])));
  }
  #pragma unroll
  for (int i_44 = 0; i_44 < 256; ++i_44) {
    acc_output[i_44] = (acc_output[i_44] / log_sum[((i_44 & 15) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_45 = 0; i_45 < 128; ++i_45) {
    uint1 __4;
    float2 v__3 = *(float2*)(acc_output + (i_45 * 2));
#ifdef ENABLE_BF16
    reinterpret_cast<__nv_bfloat162 &>(__4) = fastertransformer::float22bf162(reinterpret_cast<float2 const &>(v__3));
#else
    ((nv_bfloat162*)(&(__4.x)))->x = (bfloat16_t)(v__3.x);
    ((nv_bfloat162*)(&(__4.x)))->y = (bfloat16_t)(v__3.y);
#endif
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((i_45 >> 6) * 16384) + (((i_45 & 7) >> 1) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_45 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((i_45 & 63) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_45 & 31) >> 4) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_45 & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 131072)) = __4;
  }
  __syncthreads();
  #pragma unroll
  for (int i_46 = 0; i_46 < 256; ++i_46) {
    if (((((int)blockIdx.x) * 256) + i_46) < (cu_seqlens_q[1] - cu_seqlens_q[0])) {
      if (0 <= (((((int)blockIdx.x) * 256) + i_46) + cu_seqlens_q[0])) {
        if (((((int)blockIdx.x) * 4) + ((i_46 + cu_seqlens_q[0]) >> 6)) < 1) {
          O[(((((((int64_t)((int)blockIdx.x)) * (int64_t)917504) + (((int64_t)i_46) * (int64_t)3584)) + (((int64_t)cu_seqlens_q[(int64_t)0]) * (int64_t)3584)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = ((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) >> 6) * 16384) + (i_46 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_46 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_46 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_46 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 131072)];
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
    
    cudaError_t result_kernel_kernel = cudaFuncSetAttribute(kernel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 327680);
    if (result_kernel_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 327680, cudaGetErrorString(result_kernel_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ V, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int max_seqlen_q, bfloat16_t* __restrict__ O, cudaStream_t stream=cudaStreamDefault) {
	kernel_kernel<<<dim3((max_seqlen_q + 255) / 256, 28, 1), dim3(128, 1, 1), 327680, stream>>>(K, O, Q, V, cu_seqlens_k, cu_seqlens_q, max_seqlen_q);
	TILELANG_CHECK_LAST_ERROR("kernel_kernel");

	return 0;
}
