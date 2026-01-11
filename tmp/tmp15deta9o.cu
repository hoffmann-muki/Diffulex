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
  float acc_output[128];
  float acc_score[256];
  float log_sum[4];
  float scores_max[4];
  float scores_max_prev[4];
  float scores_sum[4];
  bfloat16_t acc_score_cast[256];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    uint2 condval;
    if (((((0 <= ((((((int)blockIdx.x) * 128) + (i * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0])) && (((((int)blockIdx.x) * 2) + ((((i * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0]) >> 6)) < 1)) && (((((int)blockIdx.x) * 2) + ((((i * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0]) >> 6)) < 1)) && (0 <= ((((((int)blockIdx.x) * 128) + (i * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_q[0])))) {
      condval = *(uint2*)(Q + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)458752) + (((int64_t)i) * (int64_t)14336)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)3584)) + (((int64_t)cu_seqlens_q[(int64_t)0]) * (int64_t)3584)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)));
    } else {
      condval = make_uint2(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 31) >> 4) * 8192) + (i * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 15) >> 3) + (i & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 65536)) = condval;
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
  for (int kv_block_idx = 0; kv_block_idx < (min(((((((int)blockIdx.x) * 128) + cu_seqlens_q[1]) + 383) - cu_seqlens_q[0]), ((cu_seqlens_k[1] + 255) - cu_seqlens_k[0])) >> 8); ++kv_block_idx) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 64; ++i_5) {
      uint2 condval_1;
      if (((((0 <= ((((kv_block_idx * 256) + (i_5 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && (((kv_block_idx * 4) + ((((i_5 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) >> 6)) < 1)) && (((kv_block_idx * 4) + ((((i_5 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) >> 6)) < 1)) && (0 <= ((((kv_block_idx * 256) + (i_5 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])))) {
        condval_1 = *(uint2*)(K + ((((((((int64_t)kv_block_idx) * (int64_t)131072) + (((int64_t)i_5) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)));
      } else {
        condval_1 = make_uint2(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
      }
      *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 31) >> 4) * 16384) + (i_5 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_5 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))) = condval_1;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 256; ++i_6) {
      float condval_2;
      if ((((((((((int)blockIdx.x) * 128) + (((i_6 & 7) >> 2) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 32) <= ((((kv_block_idx * 256) + ((i_6 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1))) || ((cu_seqlens_q[1] - cu_seqlens_q[0]) <= (((((((int)blockIdx.x) * 128) + (((i_6 & 7) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) || ((cu_seqlens_k[1] - cu_seqlens_k[0]) <= ((((kv_block_idx * 256) + ((i_6 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1))))) {
        condval_2 = -0x1.dcd65p+29f/*-1.000000e+09*/;
      } else {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      }
      acc_score[i_6] = condval_2;
    }
    __syncthreads();
    tl::gemm_ss<128, 256, 128, 4, 1, 0, 1, 0, 128, 128, 0, 0>((&(((bfloat16_t*)buf_dyn_shmem)[65536])), (&(((bfloat16_t*)buf_dyn_shmem)[0])), (&(acc_score[0])));
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      scores_max_prev[i_7] = scores_max[i_7];
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      scores_max[i_8] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      #pragma unroll
      for (int rv = 0; rv < 64; ++rv) {
        scores_max[i_9] = max(scores_max[i_9], acc_score[((((rv & 31) * 8) + (i_9 * 2)) + (rv >> 5))]);
      }
      scores_max[i_9] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      scores_max[i_10] = max(scores_max[i_10], scores_max_prev[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      scores_max_prev[i_11] = exp2f(((scores_max_prev[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 256; ++i_12) {
      acc_score[i_12] = exp2f(((acc_score[i_12] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_12 & 7) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      scores_sum[i_13] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 64; ++rv_1) {
        scores_sum[i_13] = (scores_sum[i_13] + acc_score[((((rv_1 & 31) * 8) + (i_13 * 2)) + (rv_1 >> 5))]);
      }
      scores_sum[i_13] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      log_sum[i_14] = ((log_sum[i_14] * scores_max_prev[i_14]) + scores_sum[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 128; ++i_15) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_score + (((((i_15 >> 3) * 16) + (((i_15 & 3) >> 1) * 8)) + (((i_15 & 7) >> 2) * 4)) + ((i_15 & 1) * 2)));
#ifdef ENABLE_BF16
      reinterpret_cast<__nv_bfloat162 &>(__1) = fastertransformer::float22bf162(reinterpret_cast<float2 const &>(v_));
#else
      ((nv_bfloat162*)(&(__1.x)))->x = (bfloat16_t)(v_.x);
      ((nv_bfloat162*)(&(__1.x)))->y = (bfloat16_t)(v_.y);
#endif
      *(uint1*)(acc_score_cast + (i_15 * 2)) = __1;
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 128; ++i_16) {
      acc_output[i_16] = (acc_output[i_16] * scores_max_prev[((i_16 & 7) >> 1)]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_17 = 0; i_17 < 64; ++i_17) {
      uint2 condval_3;
      if (((((0 <= ((((kv_block_idx * 256) + (i_17 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])) && (((kv_block_idx * 4) + ((((i_17 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) >> 6)) < 1)) && (((kv_block_idx * 4) + ((((i_17 * 4) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0]) >> 6)) < 1)) && (0 <= ((((kv_block_idx * 256) + (i_17 * 4)) + (((int)threadIdx.x) >> 5)) + cu_seqlens_k[0])))) {
        condval_3 = *(uint2*)(V + ((((((((int64_t)kv_block_idx) * (int64_t)131072) + (((int64_t)i_17) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)cu_seqlens_k[(int64_t)0]) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)7) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)));
      } else {
        condval_3 = make_uint2(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
      }
      *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 31) >> 4) * 16384) + (i_17 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_17 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 32768)) = condval_3;
    }
    __syncthreads();
    tl::gemm_rs<128, 128, 256, 4, 1, 0, 0, 0, 256, 128, 0, 0>((&(acc_score_cast[0])), (&(((bfloat16_t*)buf_dyn_shmem)[32768])), (&(acc_output[0])));
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 128; ++i_18) {
    acc_output[i_18] = (acc_output[i_18] / log_sum[((i_18 & 7) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_19 = 0; i_19 < 64; ++i_19) {
    uint1 __2;
    float2 v__1 = *(float2*)(acc_output + (i_19 * 2));
#ifdef ENABLE_BF16
    reinterpret_cast<__nv_bfloat162 &>(__2) = fastertransformer::float22bf162(reinterpret_cast<float2 const &>(v__1));
#else
    ((nv_bfloat162*)(&(__2.x)))->x = (bfloat16_t)(v__1.x);
    ((nv_bfloat162*)(&(__2.x)))->y = (bfloat16_t)(v__1.y);
#endif
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((i_19 >> 5) * 8192) + (((i_19 & 3) >> 1) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_19 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((i_19 & 31) >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_19 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_19 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 65536)) = __2;
  }
  __syncthreads();
  #pragma unroll
  for (int i_20 = 0; i_20 < 128; ++i_20) {
    if (((((int)blockIdx.x) * 128) + i_20) < (cu_seqlens_q[1] - cu_seqlens_q[0])) {
      if (0 <= (((((int)blockIdx.x) * 128) + i_20) + cu_seqlens_q[0])) {
        if (((((int)blockIdx.x) * 2) + ((i_20 + cu_seqlens_q[0]) >> 6)) < 1) {
          O[(((((((int64_t)((int)blockIdx.x)) * (int64_t)458752) + (((int64_t)i_20) * (int64_t)3584)) + (((int64_t)cu_seqlens_q[(int64_t)0]) * (int64_t)3584)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = ((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) >> 6) * 8192) + (i_20 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_20 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_20 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_20 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 65536)];
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
    
    cudaError_t result_kernel_kernel = cudaFuncSetAttribute(kernel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
    if (result_kernel_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 163840, cudaGetErrorString(result_kernel_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ V, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int max_seqlen_q, bfloat16_t* __restrict__ O, cudaStream_t stream=cudaStreamDefault) {
	kernel_kernel<<<dim3((max_seqlen_q + 127) / 128, 28, 1), dim3(128, 1, 1), 163840, stream>>>(K, O, Q, V, cu_seqlens_k, cu_seqlens_q, max_seqlen_q);
	TILELANG_CHECK_LAST_ERROR("kernel_kernel");

	return 0;
}
