/*
 * Copyright (C) SPQR Kernel.2024 Elvir Crncevic (elvir.crncevic@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bit_array.cuh"
#include <ATen/cuda/Exceptions.h>

#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cusparse.h>

#include <vector>
#include <cuda_pipeline.h>
#include <iostream>

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *);

// TODO: Why isn't this already available?
__device__ __forceinline__ __half operator+(const __half &lh,
                                            const __half &rh) {
  return __hadd(lh, rh);
}

template<class Acc_t> constexpr __device__ __host__ bool is_fp32() {
  if constexpr (std::is_same_v<Acc_t, float> || std::is_same_v<Acc_t, float2>) {
    return true;
  }
  return false;
}

__forceinline__ __device__ float shfl_reduce_float(float val) {
  unsigned int FULL_MASK = 0xffffffff;
  val += __shfl_down_sync(FULL_MASK, val, 16);
  val += __shfl_down_sync(FULL_MASK, val, 8);
  val += __shfl_down_sync(FULL_MASK, val, 4);
  val += __shfl_down_sync(FULL_MASK, val, 2);
  val += __shfl_down_sync(FULL_MASK, val, 1);
  return val;
}

__forceinline__ __device__ half shfl_reduce_half(half val) {
  unsigned int FULL_MASK = 0xffffffff;
  val = __hadd(val, __shfl_down_sync(FULL_MASK, val, 16));
  val = __hadd(val, __shfl_down_sync(FULL_MASK, val, 8));
  val = __hadd(val, __shfl_down_sync(FULL_MASK, val, 4));
  val = __hadd(val, __shfl_down_sync(FULL_MASK, val, 2));
  val = __hadd(val, __shfl_down_sync(FULL_MASK, val, 1));
  return val;
}

using u64 = unsigned long long;
using s32 = int;
using u32 = unsigned int;
using u16 = unsigned short;

union RowBits {
  uint64_t mask;

  struct {
    uint64_t s: 3;
    uint64_t z: 3;
    uint64_t w: 48;
  };

  __device__ __forceinline__ u16 get_w(int i) const {
    return (w >> (i * 3u)) & ((1u << 3u) - 1u);
  }

  __device__ __forceinline__ u32 get_w2(int i) const {
    return (mask >> (i * 6u)) & ((1u << 6u) - 1u);
  }
};

static constexpr uint64_t magic = 0b10001111000110100010110001001000010100000001111000000000ull;


__device__ __forceinline__ unsigned short fast_decode(unsigned short q) {
  // The magic constant is now in constant memory for faster access.

  // Shift the magic value right by 7 bits * q, and directly return the result.
  // Since im is used for a lookup table, we shift and extract the correct value.
  return static_cast<unsigned short>(0x4000 + q * 0x100);
}

template<class Bit_t, uint64_t BITS> __forceinline__ __host__ __device__ Bit_t get_bit(Bit_t w, Bit_t w_id) {
  return (w >> (w_id * BITS)) & ((1ull << BITS) - 1ull);
}


half2 __forceinline__ __device__ dequantize2(const half2 &q,
                                             const half2 &s,
                                             const half2 &z) {
  const half2 &res = __hmul2(s, __hsub2(q, z));
#if 0
  printf("dequantize2 called :: %f = %f x (%f - %f)\n%f = %f x (%f - %f)\n",
         __half2float(res.x), __half2float(s.x), __half2float(q.x),
         __half2float(z.x), __half2float(res.y), __half2float(s.y),
         __half2float(q.y), __half2float(z.y));
#endif
  return res;
}

float2 __forceinline__ __device__ dequantize2_fp32(const float2 &q,
                                                   const float2 &s,
                                                   const float2 &z) {
  return make_float2(s.x * (q.x - z.x), s.y * (q.y - z.y));
}

template<class Bit_t, class Scalar_t> __forceinline__ __device__ Scalar_t dequantize(Bit_t q,
                                                                                     Scalar_t s,
                                                                                     Scalar_t z) {
  if constexpr (std::is_same<Bit_t, half>::value) {
    return __hmul(s, __hsub(q, z));
  } else {
    return __hmul(s, __hsub(__uint2half_rd(q, z)));
  }
}

#define CUINLINE __forceinline__

#define UPDIV(X, Y) (((X) + (Y)-1) / (Y))

[[nodiscard]] __device__ __host__ CUINLINE int updiv(int x, int y) {
  return (x + y - 1) / y;
}

enum class ThreadDim { X, Y, Z, XY, YX, YZ, XYZ, YZX };

template<ThreadDim t> CUINLINE __device__ unsigned int get_thread_count() {
  if constexpr (t == ThreadDim::X) {
    return blockDim.x;
  } else if constexpr (t == ThreadDim::Y) {
    return blockDim.y;
  } else if constexpr (t == ThreadDim::Z) {
    return blockDim.z;
  } else if constexpr (t == ThreadDim::XY || t == ThreadDim::YX) {
    return blockDim.x * blockDim.y;
  } else if constexpr (t == ThreadDim::YZ) {
    return blockDim.y * blockDim.z;
  } else {
    static_assert("Invalid ID requested.");
  }
}

CUINLINE __device__ __host__ unsigned int get_global_id_x() {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

template<ThreadDim t> CUINLINE __device__ __host__ unsigned int get_thread_id() {
  if constexpr (t == ThreadDim::X) {
    return threadIdx.x;
  } else if constexpr (t == ThreadDim::Y) {
    return threadIdx.y;
  } else if constexpr (t == ThreadDim::Z) {
    return threadIdx.z;
  } else if constexpr (t == ThreadDim::XY) {
    return threadIdx.x * blockDim.y + threadIdx.y;
  } else if constexpr (t == ThreadDim::YX) {
    return threadIdx.y * blockDim.x + threadIdx.x;
  } else if constexpr (t == ThreadDim::YZ) {
    return threadIdx.y * blockDim.z + threadIdx.z;
  } else if constexpr (t == ThreadDim::XYZ) {
    return threadIdx.x * blockDim.x * blockDim.y +
           (threadIdx.y * blockDim.z + threadIdx.z);
  } else if constexpr (t == ThreadDim::YZX) {
    return (threadIdx.y * blockDim.z + threadIdx.z) * blockDim.x + threadIdx.x;
  } else {
    // not possible
  }
}

/**
 * Async and branchless clear. Zeros out data block of size n using THREAD_COUNT
 * threads.
 * @tparam T
 * @tparam D
 * @param ptr
 * @param n
 */
template<class T, ThreadDim D> __device__ CUINLINE void clr_bless_async(T *__restrict__ ptr,
                                                                        int n,
                                                                        unsigned int thread_count,
                                                                        T val = T{}) {
  if (std::is_same<half2, T>::value) {
    unsigned int thread_id = get_thread_id<D>();
    unsigned int work_to_do = updiv(n, thread_count);
    for (int i = 0; i < work_to_do; i++) {
      ptr[work_to_do * thread_id + i] = val;
    }
  } else {
    unsigned int thread_id = get_thread_id<D>();
    unsigned int thread_count = get_thread_count<D>();
    unsigned int work_to_do = updiv(n, thread_count);
    for (int i = 0; i < work_to_do; i++) {
      ptr[work_to_do * thread_id + i] = val;
    }
  }
}

/**
 * Async and branchless clear. Zeros out data block of size n using THREAD_COUNT
 * threads.
 * @tparam T
 * @tparam D
 * @param ptr
 * @param n
 */
template<class T, ThreadDim D> __device__ CUINLINE void memcpy_flat(const T *__restrict__ in,
                                                                    T *__restrict__ out,
                                                                    int n) {
  unsigned int thread_id = get_thread_id<D>();
  unsigned int thread_count = get_thread_count<D>();
  unsigned int work_to_do = updiv(n, thread_count);
  for (int i = 0; i < work_to_do; i++) {
    if (work_to_do * thread_id + i < n) {
      out[work_to_do * thread_id + i] = in[work_to_do * thread_id + i];
    }
  }
}

struct Timer {
  cudaEvent_t ce_start{}, ce_stop{};
  cudaStream_t stream;

  void start() { AT_CUDA_CHECK(cudaEventRecord(ce_start, stream)); }

  float end() {
    float time;
    AT_CUDA_CHECK(cudaEventRecord(ce_stop, 0));
    AT_CUDA_CHECK(cudaEventSynchronize(ce_stop));
    AT_CUDA_CHECK(cudaEventElapsedTime(&time, ce_start, ce_stop));
    // Returns ms
    return time;
  }

  Timer(cudaStream_t stream) : stream(stream) {
    AT_CUDA_CHECK(cudaEventCreate(&ce_start));
    AT_CUDA_CHECK(cudaEventCreate(&ce_stop));
  }

  Timer(Timer &&timer) = delete;

  Timer(const Timer &timer) = delete;

  ~Timer() {
    AT_CUDA_CHECK(cudaEventDestroy(ce_start));
    AT_CUDA_CHECK(cudaEventDestroy(ce_stop));
  }
};

#define DEVICE_INLINE __forceinline__ __device__

__device__ void _debug_halfs() {
}

template<typename T> __device__ T _debug_halfs(T v) {
  if constexpr (std::is_same<T, half>::value) {
    printf(" %f\n", __half2float(v));
  } else if constexpr (std::is_same<T, half2>::value) {
    printf(" %f %f\n", __half2float(v.x), __half2float(v.y));
  }
  return v;
}

template<typename T, typename... Arguments> __device__ void _debug_halfs(T v, Arguments... vals) {
  if constexpr (std::is_same<T, half>::value) {
    printf(" %f", __half2float(v));
  } else if constexpr (std::is_same<T, half2>::value) {
    printf(" %f %f", __half2float(v.x), __half2float(v.y));
  }
  _debug_halfs(vals...);
}

template<typename T, typename... Arguments> __device__ void debug_halfs(const char *prefix, Arguments... vals) {
  printf("%s", prefix);
  _debug_halfs(vals...);
}

__device__ float half2int2float(const __half &v) {
  return static_cast<const float>(
      *reinterpret_cast<const unsigned short *>(&v));
}

template<typename T, typename... Arguments> __device__ void printf_fp16(const char *fmt, Arguments... vals) {
#define CONV __half2float
  // #define CONV half2int2float
  printf(fmt, CONV(vals)...);
}

// Debug utils
template<class T> __device__ void debug_value(const char *str, T v) {
  if constexpr (std::is_same<T, half>::value) {
    printf("threadIdx.x = %d %s = %f\n", threadIdx.x, str, __half2float(v));
  } else if constexpr (std::is_same<T, half2>::value) {
    printf("threadIdx.x = %d %s = %f %f\n",
           threadIdx.x,
           str,
           __half2float(v.x),
           __half2float(v.y));
  } else if constexpr (std::is_same<T, int>::value) {
    printf("threadIdx.x = %d %s = %d\n", threadIdx.x, str, v);
  }
}

template<int BETA1> struct IterSecondOrder {
  const SecondOrder *base_ptr;
  SecondOrder *s_base;
  int advance;
  unsigned int n;
  int id;

  DEVICE_INLINE void next() {
    base_ptr += advance;
    id += advance;
  }

  DEVICE_INLINE half2 get_sws2() const { return s_base[0].members.ss; }

  DEVICE_INLINE half2 get_swz2() const { return s_base[0].members.zz; }

  DEVICE_INLINE void load_async_fast() {
    if (threadIdx.x < advance) {
      s_base[threadIdx.x].members = base_ptr[threadIdx.x].members;
    }
  }

  DEVICE_INLINE void load_async() {
    if (id < n && (threadIdx.x % BETA1) == 0) {
      s_base[0].members = base_ptr[0].members;
    }
  }

  DEVICE_INLINE void load_sync() {
    load_async();
    __syncthreads();
  }
};

template<unsigned int BETA1, unsigned int BETA2> struct IterX {
  const half *x;
  half2 *s_x;
  unsigned int num_x_halfs;
  unsigned int num_x_half_shared;
  unsigned int num_participating_threads;

  DEVICE_INLINE void next() { x += num_x_half_shared; }

  DEVICE_INLINE void load_async() {
    int num_half2_to_load = num_x_half_shared / 2;
    const half2 *x2 = reinterpret_cast<const half2 *>(x);
    unsigned int thread_id = threadIdx.x;

    if (thread_id >= num_participating_threads) {
      return;
    }

    int work_to_do = UPDIV(num_half2_to_load, num_participating_threads);

    for (int i = 0;
         i < work_to_do && work_to_do * thread_id + i < num_half2_to_load;
         i++) {
      s_x[work_to_do * thread_id + i] = x2[work_to_do * thread_id + i];
    }
  }

  DEVICE_INLINE half2 operator[](unsigned int local_x2_id) const {
    auto tix = threadIdx.x;
    unsigned int sx2_offset = ((tix / BETA1) * BETA2) / 2;
    return s_x[sx2_offset + local_x2_id];
  }

  DEVICE_INLINE void load_sync() {
    load_async();
    __syncthreads();
#if 0
    int num_half_to_load = min(num_x_half_shared, num_x_halfs - page * num_x_half_shared);
    if (num_half_to_load % 2 == 0) {
      load_sync_fast_sync(num_half_to_load);
    } else {

    }
    __syncthreads();
#endif
  }
};

template<unsigned int BETA1, unsigned int BETA2> struct IterXPaged {
  const half *x;
  half2 *s_x;
  unsigned int num_x_halfs;
  unsigned int num_x_half_shared;
  unsigned int num_participating_threads;

  DEVICE_INLINE void next() { x += num_x_half_shared; }

  DEVICE_INLINE void load_fast_async(int num_halfs_to_load) {
    int num_half2_to_load = num_halfs_to_load / 2;
    const half2 *x2 = reinterpret_cast<const half2 *>(x);
    unsigned int thread_id = threadIdx.x;

    if (thread_id >= num_participating_threads) {
      return;
    }

    int work_to_do = UPDIV(num_half2_to_load, num_participating_threads);

#if 0
    printf("thread_id=%d\nnum_half2_to_load = %d\nnum_participating_threads = "
           "%d\nnum_halfs_to_load = %d\nwork_to_do = %d\n",
           thread_id, num_half2_to_load, num_participating_threads,
           num_halfs_to_load, work_to_do);
#endif
    for (int i = 0;
         i < work_to_do && work_to_do * thread_id + i < num_half2_to_load;
         i++) {
#if 0
      //      printf("i = %d\n", work_to_do * thread_id + i);
      printf("loading x at location %d threadIdx.x: %d %f %f\n",
             work_to_do * thread_id + i, threadIdx.x,
             __half2float(x2[work_to_do * thread_id + i].x),
             __half2float(x2[work_to_do * thread_id + i].y));
#endif
      s_x[work_to_do * thread_id + i] = x2[work_to_do * thread_id + i];
    }
  }

  DEVICE_INLINE void load_async() { load_fast_async(num_x_half_shared); }

  DEVICE_INLINE half2 operator[](unsigned int local_x2_id) const {
    auto tix = threadIdx.x;
    unsigned int sx2_offset = ((tix / BETA1) * BETA2) / 2;
    return s_x[sx2_offset + local_x2_id];
  }

  DEVICE_INLINE void load_sync() {
    load_async();
    __syncthreads();
#if 0
    int num_half_to_load = min(num_x_half_shared, num_x_halfs - page * num_x_half_shared);
    if (num_half_to_load % 2 == 0) {
      load_sync_fast_sync(num_half_to_load);
    } else {

    }
    __syncthreads();
#endif
  }
};

template<class Scalar_t> __host__ __device__ auto vectorize(Scalar_t *ptr) {
  if constexpr (std::is_same<Scalar_t, float>::value) {
    return reinterpret_cast<float2 *>(ptr);
  } else if constexpr (std::is_same<Scalar_t, half>::value) {
    return reinterpret_cast<half2 *>(ptr);
  } else {
    return ptr;
  }
}

template<class Acc_t, int BETA1> __device__ constexpr int calc_output_size() {
  if constexpr (std::is_same<Acc_t, half2>::value ||
                std::is_same<Acc_t, float2>::value) {
    return BETA1 / 2;
  } else {
    return BETA1;
  }
}

template<class Vec_t> __host__ __device__ auto scalarize(void *ptr) {
  if constexpr (std::is_same<Vec_t, float>::value ||
                std::is_same<Vec_t, float2>::value) {
    return reinterpret_cast<float *>(ptr);
  } else if constexpr (std::is_same<Vec_t, half2>::value) {
    return reinterpret_cast<half *>(ptr);
  } else {
    return ptr;
  }
}

__device__ __forceinline__ float add_and_accum(float a, float b) {
  return a + b;
}

__device__ __forceinline__ half add_and_accum(const half2 &a, const half2 &b) {
  half2 r = __hadd2(a, b);
  return __hadd(r.x, r.y);
}

template<class T> DEVICE_INLINE u16 get_col(T m) {
  return static_cast<u16>(m & T((1u << 16u) - 1u));
}

DEVICE_INLINE half get_val(u32 m) {
  u16 _v = m >> 16u;
  half v = *reinterpret_cast<half *>(&_v);
  return v;
}

DEVICE_INLINE half get_val(u64 m) {
  u64 _v = (m >> 16u) & u64((1u << 16u) - 1u);
  half v = *reinterpret_cast<half *>(&_v);
  return v;
}

DEVICE_INLINE u16 get_row(u64 m) { return m >> 32u; }

#define CALL_FUSED(_BLOCK_HEIGHT, _BLOCK_WIDTH, PIPELINE_DEPTH) \
    constexpr int BLOCK_HEIGHT = _BLOCK_HEIGHT; \
    constexpr int BLOCK_WIDTH = _BLOCK_WIDTH; \
    size_t smem_size = sizeof(half) * prob_n; \
    spqr_quantized_matvec_fused<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, float, uint64_t, PIPELINE_DEPTH> \
            <<<dim3(updiv(prob_m, 16 * BLOCK_HEIGHT), 1, 1), \
            dim3(__min(updiv(prob_n, 16), BLOCK_WIDTH) * 16, 1, 1), smem_size, \
            stream>>>(prob_m, \
            prob_n, \
            raw_data, \
            second_order_data_ptr, \
            X_ptr, \
            row_offsets_ptr, \
            col_vals_ptr, \
            order_ptr, \
            y_ptr);

//
template<int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH, class Acc_t, class W_t /* = uint64_t */, int PIPELINE_DEPTH>
__global__ void spqr_quantized_matvec_fused(
    // W and meta
    unsigned int prob_m,
    unsigned int prob_n,
    // W 1st order stats
    const W_t *__restrict__ raw_data,
    const SecondOrder *__restrict__ second_order_data,
    const half *__restrict__ x,
    // Outliers
    const int *row_offsets,
    const u32 *col_vals,
    const short *__restrict__ order,
    // Output
    half *__restrict__ y_fp16) {
  /*
           ┌─────────────┐ ┌─┐   ┌─┐
   beta1   │   block 0   │ │ │   │ │
           ├─────────────┤ │ │   │ │
   beta1   │   block 1   │ │ │   │ │
           └─────────────┘ │x│ = │Y│
           │    ...      │ │ │   │ │
           ┌─────────────┐ │ │   │ │
   beta1   │  block m-1  │ │ │   │ │
           └─────────────┘ └─┘   └─┘
  */
  extern __shared__ half s_x[];

  __shared__ half2 s_half2_lut[64];

  if (!threadIdx.x) {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        s_half2_lut[j * 8 + i] = make_half2(__int2half_rd(i), __int2half_rd(j));
      }
    }
  }


  __shared__ Acc_t s_y[BETA1];
  __shared__ u32 s_row_offsets[BETA1 + 1];

  half2 *s_x2 = reinterpret_cast<half2 *>(s_x);
  const half2 *x2 = reinterpret_cast<const half2 *>(x);

  u32 t_id = blockDim.x * threadIdx.y + threadIdx.x;
  const u32 TOTAL_THREADS = blockDim.x * blockDim.y;
  u32 pipeline_depth{};

  const auto total_threads = blockDim.x;
  const auto count = prob_n / 2;
  const auto tid = threadIdx.x;
  u32 pipeline_id{};


  if constexpr (std::is_same<Acc_t, float>::value) {
    if (threadIdx.x < BETA1) {
      // TOD: Check if this really sets s_y to zero.
      asm volatile ("cp.async.ca.shared.global [%0], [%0], 4, 0 ;\n" :
          : "r"(__nvvm_get_smem_pointer(s_y + threadIdx.x))
          );
    }
  } else {
    if (threadIdx.x < BETA1 / 2) {
      asm volatile ("cp.async.ca.shared.global [%0], [%0], 4, 0 ;\n" :
          : "r"(__nvvm_get_smem_pointer(s_y + threadIdx.x))
          );
    }
  }

  // Here we load the row offsets into smem.
  if (threadIdx.x < BETA1) {
    __pipeline_memcpy_async(s_row_offsets + threadIdx.x, row_offsets + blockIdx.x * BETA1 + threadIdx.x, sizeof(u32));

    // The first thread will read the last sparse row offset since we cannot be sure that
    // we have enough threads to load all the sparse row offsets that we need.
    if (!threadIdx.x) {
      __pipeline_memcpy_async(s_row_offsets + BETA1, row_offsets + blockIdx.x * BETA1 + BETA1, sizeof(u32));
    }
  }


  for (int i = 0; i < PIPELINE_DEPTH && (i * total_threads + tid) < prob_n; i++) {
    unsigned idx = i * total_threads + tid;
    if (idx < count) {
      __pipeline_memcpy_async(s_x2 + idx, x2 + idx, sizeof(half2));
      pipeline_depth++;
      pipeline_id++;
      __pipeline_commit();
    }
  }

  int pipeline_stack_ptr = pipeline_depth;

  constexpr u32 WARP_SIZE = 32;
  const u32 THREAD_COUNT = blockDim.x; // = 128 (example)

  const u32 blockId = blockIdx.x;

  // Number of SPQR tiles that this CUDA block will process.
  u32 num_spqr_tiles_per_cuda_block = UPDIV(prob_n, BETA2);

  u32 total_tiles = UPDIV(prob_m, BETA1) * UPDIV(prob_n, BETA2);

  // Here is how we organize things here. We have THREAD_COUNT threads in a
  // block in x-dimension. We distribute 1 thread per tile row. Therefore, we
  // have BETA1 threads per tile. For now, a block only spans across 1 dimension
  // of SPQR tiles.
  const u32 num_spqr_tiles_per_iteration = THREAD_COUNT / BETA1;

  const u32 subtile_id = threadIdx.x / BETA1;

  if (subtile_id >= UPDIV(prob_n, BETA2)) {
    return;
  }

  // Now we set up the X loads. We have BLOCK_WIDTH * BETA2 x halfs.
  __shared__ SecondOrder s_second_order[BLOCK_HEIGHT * BLOCK_WIDTH];

  int tile_id = blockIdx.x * num_spqr_tiles_per_cuda_block + subtile_id;

  IterSecondOrder<BETA1> iter_second_order{
      .base_ptr = second_order_data + tile_id,
      .s_base = s_second_order + subtile_id,
      .advance = BLOCK_WIDTH,
      .n = total_tiles,
      .id = tile_id
  };

  constexpr int bits = get_bits<W_t>();

  constexpr int MAX_ADDR_PER_ROW = UPDIV(
  // Weight storage
      (BETA2 * BITS) +
      // Weight + Scale
      2 * BITS,
  // u32/u64 storage
      bits);

  const int MAX_ADDR_PER_TILE = BETA1;

  RowBits row_bits;
  u32 row_pos = threadIdx.x & 0xF; // threadIdx.x % BETA1;

  Acc_t acc{};

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;
  constexpr u32 HALF_WARP = WARP_SIZE / 2u;

  const int other_lane_idx = (threadIdx.x + HALF_WARP) % WARP_SIZE;

  if ((row_pos + blockId * BETA1) >= prob_m) {
    // TODO: Maybe don't do this, since we need these threads to load x
    // together? [1]
    return;
  } // || (threadIdx.x % BETA1)

  const int addr_per_row = MAX_ADDR_PER_ROW;

  for (int i = subtile_id, group_id = 0;; i += num_spqr_tiles_per_iteration, group_id++) {
    // TODO: It seems that it's important that this remans a syncthread instead
    // of a syncwarp for some reason...
    __syncthreads();

    bool finished = (i >= num_spqr_tiles_per_cuda_block) |
                    (i * BETA2 >= prob_n) |
                    ((row_pos + blockId * BETA1) >= prob_m);

    bool other_finished =
        __shfl_sync(FULL_MASK, finished, other_lane_idx, WARP_SIZE) |
        // We also have the case where the matrix dimension is smaller than
        // the warp size. Maybe use __activemask() here?
        !(__activemask() & (1u << other_lane_idx)) |
        ((other_lane_idx + blockId * BETA1) >= prob_m);

    if (finished & other_finished) {
      break;
    }

    if (!finished) {
      iter_second_order.load_async();
      row_bits.mask =
          raw_data[MAX_ADDR_PER_TILE * tile_id + row_pos * addr_per_row];
    }

    __syncthreads();

    if (pipeline_stack_ptr > 0) {
      __pipeline_wait_prior(pipeline_stack_ptr - 1);
      pipeline_stack_ptr--;
    }

    if (!finished) {
      half2 first_order_quantized = s_half2_lut[row_bits.get_w2(0)];
      half2 first_order_dequantized = dequantize2(first_order_quantized,
                                                  iter_second_order.get_sws2(),
                                                  iter_second_order.get_swz2());

      half2 ws2 = __half2half2(first_order_dequantized.x);
      half2 wz2 = __half2half2(first_order_dequantized.y);

#pragma unroll
      for (int j = 0; j < BETA2 / 2; j++) {
        if constexpr (std::is_same<Acc_t, float>::value) {
          half2 q = s_half2_lut[row_bits.get_w2(j + 1)];
          half2 w = dequantize2(q, ws2, wz2);
          float2 x_fp32 = __half22float2(s_x2[i * (BETA2 / 2) + j]);
          float2 w_fp32 = __half22float2(w);
          acc = fmaf(x_fp32.x, w_fp32.x, acc);
          acc = fmaf(x_fp32.y, w_fp32.y, acc);
        } else {
          int q_x = row_bits.get_w(2 * j);
          int q_y = row_bits.get_w(2 * j + 1);
          half2 q = make_half2(__int2half_rd(q_x), __int2half_rd(q_y));
          half2 w = dequantize2(q, ws2, wz2);
          acc = __hfma2(s_x2[i * BETA2 / 2 + j], w, acc);
        }
      }

      iter_second_order.next();
      tile_id += num_spqr_tiles_per_iteration;
    }

    unsigned idx = pipeline_id * total_threads + tid;
    if (idx < prob_n / 2) {
      __pipeline_memcpy_async(s_x2 + idx, x2 + idx, sizeof(half2));
      pipeline_id++;
      pipeline_stack_ptr++;
      __pipeline_commit();
    }
  }

  auto s_y_scalar = scalarize<Acc_t>(s_y);
  auto s_y_vectorized = vectorize(s_y_scalar);
  using Vector_ptr_t = decltype(s_y_vectorized);

  int t = threadIdx.x % BETA1;
  int s = s_row_offsets[t];
  int e = s_row_offsets[t + 1];
  int wid = threadIdx.x / BETA1;

  // We need to help out the compiler here - step size needs to be constexpr.
  if (blockDim.x == 512) {
    constexpr int step = 32;
    for (int i = s + wid; i < e; i += step) {
      auto colval = col_vals[i];
      auto c = get_col(colval);
      auto v = get_val(colval);
      acc += __half2float(v) * __half2float(s_x[c]);
    }
  } else if (blockDim.x == 256) {
    constexpr int step = 16;
    for (int i = s + wid; i < e; i += step) {
      auto colval = col_vals[i];
      auto c = get_col(colval);
      auto v = get_val(colval);
      acc += __half2float(v) * __half2float(s_x[c]);
    }
  } else if (blockDim.x == 128) {
    constexpr int step = 8;
    for (int i = s + wid; i < e; i += step) {
      auto colval = col_vals[i];
      auto c = get_col(colval);
      auto v = get_val(colval);
      acc += __half2float(v) * __half2float(s_x[c]);
    }
  } else {
    int step = blockDim.x / BETA1;
    for (int i = s + wid; i < e; i += step) {
      auto colval = col_vals[i];
      auto c = get_col(colval);
      auto v = get_val(colval);
      acc += __half2float(v) * __half2float(s_x[c]);
    }
  }

  auto result_scalar = acc;
  auto other = __shfl_down_sync(HALF_MASK, result_scalar, BETA1);
  auto result = add_and_accum(other, result_scalar);
  const unsigned int lane_id = threadIdx.x & 0x1F;
  if constexpr (std::is_same_v<Acc_t, float>) {
    __syncwarp();
    if (lane_id < BETA1) {
      atomicAdd(s_y_scalar + lane_id, result);
    }
  } else {
    auto result0 = __shfl_down_sync(0, result, threadIdx.x);
    auto result1 = __shfl_down_sync(0, result, threadIdx.x + 1);
    __syncwarp();
    if (lane_id < BETA1 / 2) {
      atomicAdd(s_y_vectorized + lane_id, make_half2(result0, result1));
    }
  }

  __syncthreads();


  if (order == nullptr) {
    if (threadIdx.x < BETA1 / 2) {
      reinterpret_cast<half2 *>(y_fp16)[blockIdx.x * (BETA1 / 2) + threadIdx.x] = __float22half2_rn(s_y_vectorized[threadIdx.x]);
    }
  } else {
    if (threadIdx.x < BETA1) {
      short row = order[blockIdx.x * BETA1 + threadIdx.x];
      y_fp16[row] = __float2half(s_y_scalar[threadIdx.x]);
    }
  }
}


//
template<int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH,
    int A, class Acc_t, class W_t /* = uint64_t */> __global__ void spqr_quantized_matvec(
    // W and meta
    unsigned int prob_m,
    unsigned int prob_n,
    // W 1st order stats
    const W_t *__restrict__ raw_data,
    const SecondOrder *__restrict__ second_order_data,
    const half *__restrict__ X,
    // Output
    float *__restrict__ y,
    half *__restrict__ y_fp16,
    bool dense_only) {
  /*
           ┌─────────────┐ ┌─┐   ┌─┐
   beta1   │   block 0   │ │ │   │ │
           ├─────────────┤ │ │   │ │
   beta1   │   block 1   │ │ │   │ │
           └─────────────┘ │x│ = │Y│
           │    ...      │ │ │   │ │
           ┌─────────────┐ │ │   │ │
   beta1   │  block m-1  │ │ │   │ │
           └─────────────┘ └─┘   └─┘
  */

  constexpr u32 WARP_SIZE = 32;
  const u32 THREAD_COUNT = blockDim.x; // = 128 (example)

  const u32 blockId = blockIdx.x;

  // Number of SPQR tiles that this CUDA block will process.
  u32 num_spqr_tiles_per_cuda_block = UPDIV(prob_n, BETA2);

  u32 total_tiles = UPDIV(prob_m, BETA1) * UPDIV(prob_n, BETA2);

  // Here is how we organize things here. We have THREAD_COUNT threads in a
  // block in x-dimension. We distribute 1 thread per tile row. Therefore, we
  // have BETA1 threads per tile. For now, a block only spans across 1 dimension
  // of SPQR tiles.
  const u32 num_spqr_tiles_per_iteration = THREAD_COUNT / BETA1;

  const u32 subtile_id = threadIdx.x / BETA1;

  if (subtile_id >= UPDIV(prob_n, BETA2)
    // || (threadIdx.x % BETA1)
      ) {
    return;
  }

  // Now we set up the X loads. We have BLOCK_WIDTH * BETA2 x halfs.
  __shared__ half _s_X[BLOCK_WIDTH * BETA2];
  __shared__ SecondOrder s_second_order[BLOCK_HEIGHT * BLOCK_WIDTH];

  int tile_id = blockIdx.x * num_spqr_tiles_per_cuda_block + subtile_id;

  IterSecondOrder<BETA1> iter_second_order{
      .base_ptr = second_order_data + tile_id,
      .s_base = s_second_order + subtile_id,
      .advance = BLOCK_WIDTH,
      .n = total_tiles,
      .id = tile_id
  };

  constexpr int bits = get_bits<W_t>();

  constexpr int MAX_ADDR_PER_ROW = UPDIV(
  // Weight storage
      (BETA2 * BITS) +
      // Weight + Scale
      2 * BITS,
  // u32/u64 storage
      bits);

  const int MAX_ADDR_PER_TILE = BETA1;

  RowBits row_bits;
  u32 row_pos = threadIdx.x & 0xF; // threadIdx.x % BETA1;

  Acc_t acc{};

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;
  constexpr u32 HALF_WARP = WARP_SIZE / 2u;

  const int other_lane_idx = (threadIdx.x + HALF_WARP) % WARP_SIZE;

  if ((row_pos + blockId * BETA1) >= prob_m) {
    // TODO: Maybe don't do this, since we need these threads to load x
    // together? [1]
    return;
  }

  u32 _num_participating_threads = {};

  if (prob_n <= 128 || prob_m <= 128) {
    u32 activemask = __activemask();
    while (activemask & 1u) {
      _num_participating_threads++;
      activemask >>= 1u;
    }
  } else {
    _num_participating_threads = blockDim.x;
  }

  IterXPaged<BETA1, BETA2> iter_x{
      .x = X,
      .s_x = reinterpret_cast<half2 *>(_s_X),
      .num_x_halfs = prob_n,
      .num_x_half_shared = min(BLOCK_WIDTH * BETA2, prob_n),
      // NOTE: See [1]
      .num_participating_threads = _num_participating_threads
  };

  const int addr_per_row = MAX_ADDR_PER_ROW;

  for (int i = subtile_id;; i += num_spqr_tiles_per_iteration) {
    // TODO: It seems that it's important that this remans a syncthread instead
    // of a syncwarp for some reason...
    __syncthreads();

    bool finished = (i >= num_spqr_tiles_per_cuda_block) |
                    (i * BETA2 >= prob_n) |
                    ((row_pos + blockId * BETA1) >= prob_m);

    bool other_finished =
        __shfl_sync(FULL_MASK, finished, other_lane_idx, WARP_SIZE) |
        // We also have the case where the matrix dimension is smaller than
        // the warp size. Maybe use __activemask() here?
        !(__activemask() & (1u << other_lane_idx)) |
        ((other_lane_idx + blockId * BETA1) >= prob_m);

    if (finished & other_finished) {
      break;
    }

    if (!finished) {
      iter_x.load_async();
      iter_second_order.load_async();
      row_bits.mask =
          raw_data[MAX_ADDR_PER_TILE * tile_id + row_pos * addr_per_row];
    }

    __syncthreads();

    if (!finished) {
      bool row_valid = (blockId + row_pos < prob_m);

      int s = row_bits.s;
      int z = row_bits.z;
      half2 first_order_quantized =
          make_half2(__int2half_rd(s), __int2half_rd(z));

      half2 first_order_dequantized =
          dequantize2(first_order_quantized,
                      iter_second_order.get_sws2(),
                      iter_second_order.get_swz2());

      half2 ws2 = __half2half2(first_order_dequantized.x);
      half2 wz2 = __half2half2(first_order_dequantized.y);

#pragma unroll
      for (int j = 0; j < BETA2 / 2; j++) {
        int q_x = row_bits.get_w(2 * j);
        int q_y = row_bits.get_w(2 * j + 1);
        if constexpr (std::is_same<Acc_t, float>::value) {
          half2 q = make_half2(__int2half_rd(q_x), __int2half_rd(q_y));
          half2 w = dequantize2(q, ws2, wz2);

          float2 x_fp32 = __half22float2(iter_x[j]);
          float2 w_fp32 = __half22float2(w);
          acc = fmaf(x_fp32.x, w_fp32.x, acc);
          acc = fmaf(x_fp32.y, w_fp32.y, acc);
        } else {
          half2 q = make_half2(__int2half_rd(q_x), __int2half_rd(q_y));
          half2 w = dequantize2(q, ws2, wz2);
          acc = __hfma2(iter_x[j], w, acc);
        }
      }

      iter_x.next();
      iter_second_order.next();
      tile_id += num_spqr_tiles_per_iteration;
    }
  }

  auto s_y_scalar = scalarize<Acc_t>(reinterpret_cast<void *>(_s_X));
  auto s_y_vectorized = vectorize(s_y_scalar);
  using Vector_ptr_t = decltype(s_y_vectorized);
  using Vector_t = std::remove_pointer_t<Vector_ptr_t>;

  if constexpr (!std::is_same<Acc_t, float>::value) {
    if (threadIdx.x < BETA1 / 2) {
      clr_bless_async<Vector_t, ThreadDim::X>(s_y_vectorized,
                                              BETA1 / 2,
                                              BETA1 / 2,
                                              Vector_t());
    }
  } else {
    if (threadIdx.x < BETA1) {
      s_y_scalar[threadIdx.x] = 0.f;
    }
  }

  auto result_scalar = acc;

  auto other = __shfl_down_sync(HALF_MASK, result_scalar, BETA1);

  auto result = add_and_accum(other, result_scalar);

  const unsigned int lane_id = threadIdx.x & 0x1F;

  if constexpr (std::is_same<Acc_t, float>::value) {
    __syncthreads();
    if (lane_id < BETA1) {
      atomicAdd(s_y_scalar + lane_id, result);
    }
  } else {
    auto result0 = __shfl_down_sync(0, result, threadIdx.x);
    auto result1 = __shfl_down_sync(0, result, threadIdx.x + 1);
    __syncthreads();
    if (lane_id < BETA1 / 2) {
      atomicAdd(s_y_vectorized + lane_id, make_half2(result0, result1));
    }
  }
  __syncthreads();

  // At this point, the result is in s_y.
  if (threadIdx.x < BETA1) {
    if (dense_only) {
      y_fp16[blockIdx.x * BETA1 + threadIdx.x] =
          __float2half(s_y_scalar[threadIdx.x]);
    } else {
      y[blockIdx.x * BETA1 + threadIdx.x] = s_y_scalar[threadIdx.x];
    }
  }
}

template<typename T> __forceinline__ __device__ T myLoad(const T *d) {
  return *d;
}

__global__ void spmv_naive_single_thread(u32 m,
                                         u32 n,
                                         const int *__restrict__ row_offsets,
                                         const short *__restrict__ col_ids,
                                         const half *__restrict__ values,
                                         const half *__restrict__ x,
                                         half *__restrict__ y) {
  for (int i = 0; i < m; i++) {
    half sum{};
    for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
      short c = col_ids[j];
      sum = __hadd(sum, __hmul(values[j], x[c]));
    }
    y[i] = __hadd(sum, y[i]);
  }
}

__global__ void spmv_naive_mixed(uint32_t m,
                                 uint32_t n,
                                 const int *__restrict__ row_offsets,
                                 const short *__restrict__ col_ids,
                                 const half *__restrict__ values,
                                 const half *__restrict__ x,
                                 half *__restrict__ y) {
  extern __shared__ half s_x[];
  const half2 *x2 = reinterpret_cast<const half2 *>(x);

  const auto TILE_SIZE = blockDim.x;

  half2 *s_x2 = reinterpret_cast<half2 *>(s_x);

  // NOTE: n should be divisible by 2!
  assert(n % 2 == 0);

  for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
    s_x2[i] = x2[i];
  }

  __syncthreads();

  uint32_t row_tile = blockDim.x;
  uint32_t row_id = TILE_SIZE * row_tile + threadIdx.x;

  if (row_id >= m) {
    return;
  }

  int row_end = row_offsets[row_id + 1];
  int row_start = row_offsets[row_id];

  int nnz = row_end - row_start;

  u32 mask = __activemask();
  u32 nnz_sum{};

  nnz_sum = __reduce_add_sync(mask, nnz_sum);

  u32 active_row_count = __popc(__activemask());

  bool _is_balanced = nnz < ((nnz_sum - 1) / 2);
  bool is_balanced = __reduce_or_sync(mask, _is_balanced);

  half sum{};
  if (is_balanced) {
    for (int j = row_start; j < row_start; j++) {
      short c = col_ids[j];
      auto v_fp16 = values[j];
      auto x_fp16 = x[c];
      sum = __hadd(sum, __hmul(x_fp16, v_fp16));
    }
  } else {
    /* TODO: Ignored for now. */
  }

  y[row_id] = __hadd(sum, y[row_id]);
}

template<class Acc_t> __global__ void
spmv_naive_shared_sorted(uint32_t m,
                         uint32_t n,
                         const short *__restrict__ row_ids,
                         const int *__restrict__ row_offsets,
                         const u32 *col_vals,
                         const half *__restrict__ x,
                         half *__restrict__ y,
                         float *__restrict__ y_fp32) {
  extern __shared__ half s_x[];

  __shared__ u32 s_row_offsets[9];
  const half2 *x2 = reinterpret_cast<const half2 *>(x);
  const u32 TILE_SIZE = blockDim.y;

  u32 t_id = blockDim.x * threadIdx.y + threadIdx.x;
  u32 TOTAL_THREADS = blockDim.x * blockDim.y;

  half2 *s_x2 = reinterpret_cast<half2 *>(s_x);
  for (int i = t_id; i < n / 2; i += TOTAL_THREADS) {
    s_x2[i] = x2[i];
  }

  for (int i = t_id; i < blockDim.y + 1; i++) {
    s_row_offsets[t_id] = row_offsets[blockDim.y * blockIdx.x + t_id];
  }

  __syncthreads();

  u32 _row_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (_row_id >= m) {
    return;
  }

  auto row_id = row_ids[_row_id];

  u32 current_start = row_offsets[row_id];
  u32 current_end = row_offsets[row_id + 1];

#if 0
  if (row_id == 1) {
    printf("shared start start end = %d %d %d %d\n", current_start,
           current_end, row_start, row_end);
  }
#endif

  Acc_t sum{};
  for (u32 j = current_start + threadIdx.x; j < current_end; j += blockDim.x) {
    auto colval = col_vals[j];
    u16 c = get_col(colval);
    half v_fp16 = get_val(colval);

    auto x_fp16 = s_x[c];
#define h2f __half2float
    if constexpr (is_fp32<Acc_t>()) {
      // TODO: Intrinsics?
      sum += h2f(x_fp16) * h2f(v_fp16);
    } else {
      sum = __hfma(x_fp16, v_fp16, sum);
    }
  }

  if constexpr (is_fp32<Acc_t>()) {
    sum = shfl_reduce_float(sum);
  } else {
    sum = shfl_reduce_half(sum);
  }

  if (threadIdx.x == 0) {
    if constexpr (is_fp32<Acc_t>()) {
      y[row_id] = __float2half(sum + y_fp32[row_id]);
    }
  }
}

template<class Acc_t> __global__ void spmv_naive_sparse_sorted(int offset,
                                                               uint32_t m,
                                                               uint32_t n,
                                                               const short *__restrict__ row_ids,
                                                               const int *__restrict__ row_offsets,
                                                               const u32 *__restrict__ col_vals,
                                                               const half *__restrict__ x,
                                                               half *__restrict__ y,
                                                               float *__restrict__ y_fp32) {
  u32 TILE_SIZE = blockDim.x;
  uint32_t row_tile = blockIdx.x;
  uint32_t _row_id = TILE_SIZE * row_tile + threadIdx.x + offset;
  if (_row_id >= m) {
    return;
  }
  short row_id = row_ids[_row_id];
  int row_start = row_offsets[row_id];
  int row_end = row_offsets[row_id + 1];

  Acc_t sum{};

  const ColVal *col_vals_u = reinterpret_cast<const ColVal *>(col_vals);

  for (int j = row_start; j < row_end; j++) {
    auto colval = col_vals[j];
    u16 c = get_col(colval);
    half v_fp16 = get_val(colval);

    auto x_fp16 = x[c];

#define h2f __half2float
    if constexpr (is_fp32<Acc_t>()) {
      // TODO: Intrinsics?
      sum += h2f(x_fp16) * h2f(v_fp16);
    } else {
      sum = __hfma(x_fp16, v_fp16, sum);
    }
  }

  y[row_id] = __float2half(sum + y_fp32[row_id]);
}

__global__ void spmv_naive_shared_baseline(uint32_t m,
                                           uint32_t n,
                                           const int *__restrict__ row_offsets,
                                           const short *__restrict__ col_ids,
                                           const u32 *__restrict__ col_vals,
                                           const half *__restrict__ x,
                                           half *__restrict__ y) {
  extern __shared__ half s_x[];
  const half2 *x2 = reinterpret_cast<const half2 *>(x);

  half2 *s_x2 = reinterpret_cast<half2 *>(s_x);

  unsigned int num_participating_threads = blockDim.x;
  for (int i = threadIdx.x; i < n / 2; i += num_participating_threads) {
    s_x2[i] = x2[i];
  }

  u32 TILE_SIZE = blockDim.x;
  uint32_t row_tile = blockIdx.x;
  uint32_t row_id = TILE_SIZE * row_tile + threadIdx.x;
  if (row_id >= m) {
    return;
  }
  int row_end = row_offsets[row_id + 1];

  // Everyone except the 0-th.
  constexpr u32 FULL_MASK = 0xffffffffu - 1u;

  int row_start = __shfl_up_sync(FULL_MASK, row_end, 1);

  if (!threadIdx.x) {
    row_start = row_offsets[row_id];
  }

  __syncthreads();

  half sum{};

  const ColVal *col_vals_u = reinterpret_cast<const ColVal *>(col_vals);

  for (int j = row_start; j < row_end; j++) {
    u16 c = col_vals_u[j].members.c;
    half v_fp16 = col_vals_u[j].members.v;
    auto x_fp16 = s_x[c];
    sum = __hadd(sum, __hmul(x_fp16, v_fp16));
  }

  y[row_id] = __hadd(sum, y[row_id]);
}

__global__ void spmv_naive_shared(uint32_t m,
                                  uint32_t n,
                                  const int *__restrict__ row_offsets,
                                  const u32 *col_vals,
                                  const half *__restrict__ x,
                                  half *__restrict__ y) {
  const ColVal *col_val_u = reinterpret_cast<const ColVal *>(col_vals);
  extern __shared__ half s_x[];
  __shared__ u32 s_row_offsets[5];
  const half2 *x2 = reinterpret_cast<const half2 *>(x);
  const u32 TILE_SIZE = blockDim.y;

  u32 t_id = blockDim.x * threadIdx.y + threadIdx.x;
  u32 TOTAL_THREADS = blockDim.x * blockDim.y;

  half2 *s_x2 = reinterpret_cast<half2 *>(s_x);

  for (int i = t_id; i < n / 2; i += TOTAL_THREADS) {
    s_x2[i] = x2[i];
  }

  for (int i = t_id; i < blockDim.y + 1; i++) {
    s_row_offsets[t_id] = row_offsets[blockDim.y * blockIdx.x + t_id];
  }

  __syncthreads();

  u32 row_id = blockIdx.x * blockDim.y + threadIdx.y;

  if (row_id >= m) {
    return;
  }

  u32 current_start = row_offsets[row_id];
  u32 current_end = row_offsets[row_id + 1];

#if 0
  if (row_id == 1) {
    printf("shared start start end = %d %d %d %d\n", current_start,
           current_end, row_start, row_end);
  }
#endif

  half sum{};
  for (u32 j = current_start + threadIdx.x; j < current_end; j += blockDim.x) {
    ColVal col_val = col_val_u[j];
    u16 c = col_val.members.c;
    half v_fp16 = col_val.members.v;
    auto x_fp16 = s_x[c];
    sum = __hadd(sum, __hmul(x_fp16, v_fp16));
  }

  sum = shfl_reduce_half(sum);

  if (threadIdx.x == 0) {
    y[row_id] = __hadd(sum, y[row_id]);
#if 0
    printf("shared y[%d] = %f\n", row_id, __half2float(y[row_id]));
#endif
  }
}

__global__ void spmv_naive(uint32_t m,
                           const int *__restrict__ row_offsets,
                           const short *__restrict__ col_ids,
                           const half *__restrict__ values,
                           const half *__restrict__ x,
                           half *__restrict__ y) {
  uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= m) {
    return;
  }

  float sum{};
  for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
    short c = col_ids[j];
    auto v_fp32 = __half2float(values[j]);
    auto x_fp32 = __half2float(x[c]);
    sum += x_fp32 * v_fp32;
  }
  y[i] = __float2half(sum + __half2float(y[i]));
}

__global__ void spmv_naive_fp16(uint32_t m,
                                const int *__restrict__ row_offsets,
                                const short *__restrict__ col_ids,
                                const half *__restrict__ values,
                                const half *__restrict__ x,
                                half *__restrict__ y) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) {
    return;
  }

  half sum{};
  for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
    short c = col_ids[j];
    sum = __hadd(sum, __hmul(values[j], x[c]));
  }

  y[i] = __hadd(sum, y[i]);
}

// double atomic add hack for devices that do not support it in hardware
template<typename T> __device__ inline T tempAtomicAdd(T *address, T val) {
  return atomicAdd(address, val);
}

#if __CUDA_ARCH__ < 600

// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
template<> __device__ inline double tempAtomicAdd<double>(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *) address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif

template<typename ValueType, typename IndexType, typename OffsetType> __global__ void
spmvt(uint32_t num_non_zeroes,
      uint32_t out_size,
      uint32_t num_other,
      const ValueType *__restrict matrix,
      const IndexType *__restrict inIndex,
      const OffsetType *__restrict offsets,
      const ValueType *__restrict inVec,
      ValueType *__restrict outVec) {
  uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_other)
    return;

  ValueType inV = myLoad(inVec + i);
  for (OffsetType j = myLoad(offsets + i); j < myLoad(offsets + i + 1); ++j) {
    IndexType ind = myLoad(inIndex + j);
    atomicAdd(outVec + ind, __hmul(inV, myLoad(matrix + j)));
  }
}

template<class T> const T &__min(const T &a, const T &b) {
  return (b < a) ? b : a;
}

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

static constexpr int DENSE_ONLY = 0;
static constexpr int NAIVE_SPARSE = 1;
static constexpr int DENSE_NAIVE_SPARSE = 2;
static constexpr int CUSPARSE = 3;
static constexpr int TILED_MATVEC = 4;

union Features {
  uint32_t _;

  struct {
    uint32_t is_fp32: 1;
    uint32_t dense_only: 1;
    uint32_t naive_sparse: 1;
    uint32_t torch: 1;
    uint32_t is_async: 1;
    uint32_t shared_sparse: 1;
    uint32_t single_sparse: 1;
    uint32_t cusparse: 1;
    uint32_t fused_sparse: 1;
    uint32_t shared_sparse_baseline: 1;
    uint32_t shared_mixture: 1;
    uint32_t rest: 21;
  } flags;
};

int spqr_matvec(
    // W and meta
    int bits,
    int prob_m,
    int prob_n,
    // Quantization
    int beta1,
    int beta2,
    const void *_raw_data,
    const void *second_order_data,
    void *row_ids,
    // 32-bit
    void *row_offsets,
    // 16-bit
    void *col_vals,
    int nnz,
    int dense_row_count,
    // 16-bit
    // Input
    void *X,
    void *order,
    // Output
    void *y,
    cudaStream_t stream,
    void *measurements,
    uint32_t feature_flag) {
  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties, 0);
  std::string gpu_name = device_properties.name;

  bool is_a100 = gpu_name.find("A100") != std::string::npos;


  if (prob_m == 0 || prob_n == 0) {
    return 0;
  }

  cudaStream_t stream_sparse;
  cudaStream_t stream_sparse0;
  Features features{._ = feature_flag};

  bool dense_only = (nnz == 0) | features.flags.dense_only;

  if (features.flags.shared_mixture && nnz) {
    cudaStreamCreate(&stream_sparse);
    cudaStreamCreate(&stream_sparse0);
  }

  const uint64_t *raw_data = (const uint64_t *) _raw_data;
  const half *X_ptr = (const half *) X;
  const int *row_offsets_ptr = (const int *) row_offsets;
  const short *row_ids_ptr = (short *) row_ids;
  half *y_ptr = (half *) y;
  const auto *second_order_data_ptr = static_cast<const SecondOrder *>(second_order_data);
  const auto *col_vals_ptr = (const u32 *) col_vals;
  const short *order_ptr = (const short *) order;

  float *d_yfp32;
  if (features.flags.shared_mixture) {
    cudaMalloc((void **) &d_yfp32, sizeof(float) * prob_m);
    cudaMemset(d_yfp32, 0, sizeof(float) * prob_m);
    cudaDeviceSynchronize();
  }

  int ret = 0;

  Timer *timer{};
  if (measurements) {
    timer = new Timer(stream);
    timer->start();
  }


  if (features.flags.fused_sparse) {
    if (is_a100) {
      CALL_FUSED(1, 64, 2);
    } else {
      CALL_FUSED(1, 16, 4);
    }
  } else {
    constexpr int BLOCK_HEIGHT = 1;
    constexpr int BLOCK_WIDTH = 16;
    spqr_quantized_matvec<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, 32, float, uint64_t>
    <<<dim3(updiv(prob_m, 16 * BLOCK_HEIGHT), 1, 1),
    dim3(__min(updiv(prob_n, 16), BLOCK_WIDTH) * 16, 1, 1), 0, stream>>>(
        prob_m,
        prob_n,
        raw_data,
        second_order_data_ptr,
        X_ptr,
        d_yfp32,
        y_ptr,
        dense_only);

    if (!dense_only) {
      CHECK_CUDA(cudaDeviceSynchronize());

      if (features.flags.shared_mixture && nnz) {
        constexpr int WARP_SIZE = 32;
        constexpr int BLOCK_HEIGHT = 4;
        int sparse_row_count = prob_m - dense_row_count;
        if (sparse_row_count) {
          spmv_naive_sparse_sorted<float>
          <<<UPDIV(prob_m - dense_row_count, WARP_SIZE), WARP_SIZE, 0,
          stream_sparse>>>(dense_row_count,
                           prob_m,
                           prob_n,
                           row_ids_ptr,
                           row_offsets_ptr,
                           col_vals_ptr,
                           X_ptr,
                           y_ptr,
                           d_yfp32);
        }

        if (dense_row_count) {
          size_t smem_size = sizeof(half) * prob_n;
          spmv_naive_shared_sorted<float>
          <<<UPDIV(dense_row_count, BLOCK_HEIGHT),
          dim3(WARP_SIZE, BLOCK_HEIGHT, 1), smem_size, stream_sparse0>>>(
              dense_row_count,
              prob_n,
              row_ids_ptr,
              row_offsets_ptr,
              col_vals_ptr,
              X_ptr,
              y_ptr,
              d_yfp32);
        }
      }
    }
  }

  if (!features.flags.is_async) {
    CHECK_CUDA(cudaDeviceSynchronize());
    if (features.flags.shared_mixture && nnz) {
      cudaStreamDestroy(stream_sparse);
      cudaStreamDestroy(stream_sparse0);
    }
  }


  if (measurements) {
    static_cast<float *>(measurements)[0] = timer->end();
    delete timer;
  }

  if (!features.flags.fused_sparse) {
    cudaFree(d_yfp32);
  }

  return ret;
}
