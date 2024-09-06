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

// TODO: Why isn't this already available?
__device__ __forceinline__ __half operator+(const __half &lh,
                                            const __half &rh) {
  return __hadd(lh, rh);
}

template <class Acc_t> constexpr __device__ __host__ bool is_fp32() {
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

union RowBits {
  uint64_t mask;
  struct {
    uint64_t s : 3;
    uint64_t z : 3;
    uint64_t w : 48;
  };

  __device__ uint64_t get_w(int i) {
    return (w >> (i * 3u)) & ((1u << 3u) - 1u);
  }
};

template <class Bit_t, uint64_t BITS>
__forceinline__ __host__ __device__ Bit_t get_bit(Bit_t w, Bit_t w_id) {
  return (w >> (w_id * BITS)) & ((1ull << BITS) - 1ull);
}

using u32 = unsigned int;
using u16 = unsigned short;

half2 __forceinline__ __device__ dequantize2(const half2 &q, const half2 &s,
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

template <class Bit_t, class Scalar_t>
__forceinline__ __device__ Scalar_t dequantize(Bit_t q, Scalar_t s,
                                               Scalar_t z) {
  if constexpr (std::is_same<Bit_t, half>::value) {
    return __hmul(s, __hsub(q, z));
  } else {
    return __hmul(s, __hsub(__uint2half_rd(q, z)));
  }
}

#define CUINLINE __forceinline__

#define UPDIV(X, Y) (((X) + (Y) - 1) / (Y))

[[nodiscard]] __device__ __host__ CUINLINE int updiv(int x, int y) { return (x + y - 1) / y; }

enum class ThreadDim { X, Y, Z, XY, YX, YZ, XYZ, YZX };

template <ThreadDim t> CUINLINE __device__ unsigned int get_thread_count() {
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

template <ThreadDim t>
CUINLINE __device__ __host__ unsigned int get_thread_id() {
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
    return threadIdx.x * blockDim.x * blockDim.y + (threadIdx.y * blockDim.z + threadIdx.z);
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
template <class T, ThreadDim D>
__device__ CUINLINE void clr_bless_async(T *__restrict__ ptr, int n,
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
template <class T, ThreadDim D>
__device__ CUINLINE void memcpy_flat(const T *__restrict__ in,
                                     T *__restrict__ out, int n) {
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

__device__ void _debug_halfs() {}

template <typename T> __device__ T _debug_halfs(T v) {
  if constexpr (std::is_same<T, half>::value) {
    printf(" %f\n", __half2float(v));
  } else if constexpr (std::is_same<T, half2>::value) {
    printf(" %f %f\n", __half2float(v.x), __half2float(v.y));
  }
  return v;
}

template <typename T, typename... Arguments>
__device__ void _debug_halfs(T v, Arguments... vals) {
  if constexpr (std::is_same<T, half>::value) {
    printf(" %f", __half2float(v));
  } else if constexpr (std::is_same<T, half2>::value) {
    printf(" %f %f", __half2float(v.x), __half2float(v.y));
  }
  _debug_halfs(vals...);
}

template <typename T, typename... Arguments>
__device__ void debug_halfs(const char *prefix, Arguments... vals) {
  printf("%s", prefix);
  _debug_halfs(vals...);
}

__device__ float half2int2float(const __half &v) {
  return static_cast<const float>(
      *reinterpret_cast<const unsigned short *>(&v));
}

template <typename T, typename... Arguments>
__device__ void printf_fp16(const char *fmt, Arguments... vals) {
#define CONV __half2float
  // #define CONV half2int2float
  printf(fmt, CONV(vals)...);
}

// Debug utils
template <class T> __device__ void debug_value(const char *str, T v) {
  if constexpr (std::is_same<T, half>::value) {
    printf("threadIdx.x = %d %s = %f\n", threadIdx.x, str, __half2float(v));
  } else if constexpr (std::is_same<T, half2>::value) {
    printf("threadIdx.x = %d %s = %f %f\n", threadIdx.x, str, __half2float(v.x),
           __half2float(v.y));
  } else if constexpr (std::is_same<T, int>::value) {
    printf("threadIdx.x = %d %s = %d\n", threadIdx.x, str, v);
  }
}

template <int BETA1> struct IterSecondOrder {
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

template <unsigned int BETA1, unsigned int BETA2> struct IterX {
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

template <class Scalar_t> __host__ __device__ auto vectorize(Scalar_t *ptr) {
  if constexpr (std::is_same<Scalar_t, float>::value) {
    return reinterpret_cast<float2 *>(ptr);
  } else if constexpr (std::is_same<Scalar_t, half>::value) {
    return reinterpret_cast<half2 *>(ptr);
  } else {
    return ptr;
  }
}

template <class Acc_t, int BETA1> __device__ constexpr int calc_output_size() {
  if constexpr (std::is_same<Acc_t, half2>::value ||
                std::is_same<Acc_t, float2>::value) {
    return BETA1 / 2;
  } else {
    return BETA1;
  }
}

template <class Vec_t> __host__ __device__ auto scalarize(void *ptr) {

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

#define DBG 0

#if DBG
#define DEBUG_PARAMS_FP32 , nullptr, nullptr, nullptr
#define DEBUG_PARAMS_FP16 , dbg_deq_w.get(), dbg_first.get(), dbg_second.get()
#define DEBUG_PARAMS_DECL , half *dbg_deq_w, half *dbg_first, half *dbg_second
#else
#define DEBUG_PARAMS_FP32
#define DEBUG_PARAMS_FP16
#define DEBUG_PARAMS_DECL
#endif

//
template <int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH,
          int A, class Acc_t, class W_t /* = uint64_t */>
__global__ void spqr_quantized_matvec_fused(
    // W and meta
    unsigned int prob_m, unsigned int prob_n,
    // W 1st order stats
    const W_t *__restrict__ raw_data,
    const half *__restrict__ second_order_data, const half *__restrict__ X,
    // Outliers
    const int *row_offsets, const short *col_ids, const half *values,
    // 32-bit
    const u32 *col_vals,
    // extra global storage for barrier synchronization
    // Output
    half *__restrict__ y
        // Debug
        DEBUG_PARAMS_DECL) {
#if 0
  extern __shared__ half _s_x[];

  const half2 *x2 = reinterpret_cast<const half2 *>(X);
  half2 *s_x2 = reinterpret_cast<half2 *>(_s_x);
  unsigned int num_participating_threads = blockDim.x;
  for (int i = threadIdx.x; i < prob_n / 2; i += num_participating_threads) {
    s_x2[i] = x2[i];
  }

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

  __shared__ half2 s_W[BLOCK_HEIGHT * BLOCK_WIDTH];
  __shared__ Acc_t s_y[calc_output_size<Acc_t, BETA1>()];

  auto s_y_scalar = scalarize(s_y);

  int tile_id = blockIdx.x * num_spqr_tiles_per_cuda_block + subtile_id;

  IterSecondOrder<BETA1> iter_second_order{
      .base_ptr =
          reinterpret_cast<const half2 *>(second_order_data + 4 * tile_id),
      .s_ws2 = s_W + 2 * subtile_id,
      .s_wz2 = s_W + 2 * subtile_id + 1,
      .advance = 2 * BLOCK_WIDTH,
      .n = 2 * total_tiles,
      .id = 2 * tile_id};

  constexpr int bits = get_bits<W_t>();

  constexpr int MAX_ADDR_PER_ROW = UPDIV(
      // Weight storage
      (BETA2 * BITS) +
          // Weight + Scale
          2 * BITS,
      // u32/u64 storage
      bits);

  const int MAX_ADDR_PER_TILE = BETA1;

  W_t _weight_bits[MAX_ADDR_PER_ROW];

  u32 row_pos = threadIdx.x & 0xF; // threadIdx.x % BETA1;

  Acc_t acc{};

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;
  constexpr u32 HALF_WARP = WARP_SIZE / 2u;

  const int other_lane_idx = (threadIdx.x + HALF_WARP) % WARP_SIZE;

  if ((row_pos + blockId * BETA1) >= prob_m) {
    return;
  }

  const int addr_per_row = MAX_ADDR_PER_ROW;
#if 0
     UPDIV(
     // Weight storage
     // min(BETA2, prob_n - subtile_id * BETA2) * BITS +
     BETA2, * BITS +
         // Weight + Scale
         2 * BITS,
     // u32 storage
     32);
#endif

  __syncthreads();
  for (int i = subtile_id;; i += num_spqr_tiles_per_iteration) {
    half2 *iter_x = s_x2 + i * (BETA2 / 2);

    u32 global_tile_id = blockId * num_spqr_tiles_per_cuda_block + i;

    __syncwarp();

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

      _weight_bits[0] =
          raw_data[MAX_ADDR_PER_TILE * tile_id + row_pos * addr_per_row];

#if 0
     printf("threadIdx.x = %d reading addr = %d got = %lu\n", threadIdx.x,
            MAX_ADDR_PER_TILE * tile_id + row_pos * addr_per_row,
            _weight_bits[0]);
#endif

      __syncthreads();

      if (!finished) {
        const W_t row_bits = _weight_bits[0];

        int row_valid = (blockId + row_pos < prob_m);

        int s = static_cast<int>(get_bit<W_t, BITS>(row_bits, 0));
        int z = static_cast<int>(get_bit<W_t, BITS>(row_bits, 1));
        half2 first_order_quantized =
            make_half2(__int2half_rd(s), __int2half_rd(z));

        half2 first_order_dequantized =
            dequantize2(first_order_quantized, iter_second_order.get_sws2(),
                        iter_second_order.get_swz2());

        half2 ws2 = __half2half2(first_order_dequantized.x);
        half2 wz2 = __half2half2(first_order_dequantized.y);

#if 0
       if (threadIdx.x == 16)
                 printf("GPU threadidx.x = %d s = %d z = %d dequantized s z = %f %f "
                        "given row_bit = %lu\n",
                        threadIdx.x, s, z, __half2float(first_order_dequantized.x),
                        __half2float(first_order_dequantized.y), row_bits);
       printf("threadId.x = %d GPU second order = %f %f %f %f\n", threadIdx.x,
              __half2float(iter_second_order.get_sws2().x),
              __half2float(iter_second_order.get_swz2().x),
              __half2float(iter_second_order.get_sws2().y),
              __half2float(iter_second_order.get_swz2().y));
#endif

#if DBG
        if constexpr (!std::is_same<Acc_t, float>::value) {
          u32 idx = global_tile_id * BETA1 * 2 + 2 * row_pos;
          dbg_first[idx] = first_order_quantized.x;
          dbg_first[idx + 1] = first_order_quantized.y;

          dbg_second[4 * global_tile_id + 0] = iter_second_order.get_sws2().x;
          dbg_second[4 * global_tile_id + 1] = iter_second_order.get_sws2().y;
          dbg_second[4 * global_tile_id + 2] = iter_second_order.get_swz2().x;
          dbg_second[4 * global_tile_id + 3] = iter_second_order.get_swz2().y;
        }
#endif

        // Assumes that the number of columns is disible by 2.
        unsigned int iterations_to_make =
            min(BETA2, prob_n - subtile_id * BETA2);
        for (int j = 0; j < iterations_to_make / 2; j++) {
          int q_x = get_bit<W_t, BITS>(row_bits, 2 + 2 * j);
          int q_y = get_bit<W_t, BITS>(row_bits, 2 + 2 * j + 1);
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
            u32 pad_n = updiv(prob_n, BETA2) * BETA2;
            acc = __hfma2(iter_x[j], w, acc);
          }
        }
      }

      // TODO: Do we need this here?
      __syncthreads();

      iter_second_order.next();
      tile_id += num_spqr_tiles_per_iteration;
    }

    auto s_y_vectorized = vectorize(s_y);
    using Vector_ptr_t = decltype(s_y_vectorized);
    using Vector_t = std::remove_pointer_t<Vector_ptr_t>;

    if (threadIdx.x < BETA1 / 2) {
      clr_bless_async<Vector_t, ThreadDim::X>(s_y_vectorized, BETA1 / 2,
                                              BETA1 / 2, Vector_t());
    }

    __syncthreads();

    auto result_scalar = acc;

    __syncwarp();
    auto other = __shfl_down_sync(HALF_MASK, result_scalar, BETA1);
    __syncwarp();

    auto result = add_and_accum(other, result_scalar);

    if ((threadIdx.x % WARP_SIZE) < BETA1) {
      atomicAdd(s_y_scalar + threadIdx.x % WARP_SIZE, result);
    }

    __syncthreads();

    // At this point, the result is in s_y.
    // Now we do the sparse part, so we restrict to threads = BETA1
#if 1

#if 0

 // CORRECTNESS: What if m < BETA1?
 if (threadIdx.x < BETA1) {
     u32 TILE_SIZE = BETA1;
     uint32_t row_tile = blockIdx.x;
     uint32_t row_id = TILE_SIZE * row_tile + threadIdx.x;
     int row_start = row_offsets[row_id];
     int row_end = row_offsets[row_id + 1];
     // TODO: Float version missing.
     half sum{};
     for (int j = row_start; j < row_end; j++) {
       short c = col_ids[j];
       auto v_fp16 = values[j];
       auto x_fp16 = _s_x[c];
       sum = __hadd(sum, __hmul(x_fp16, v_fp16));
     }
 }

#else
    __shared__ u32 s_row_offsets[BETA1 + 1];
    u32 TILE_SIZE = BETA1;
    u32 row_tile = blockIdx.x;

    if (threadIdx.x <= TILE_SIZE) {
      s_row_offsets[threadIdx.x] =
          row_offsets[TILE_SIZE * row_tile + threadIdx.x];
    }

    __syncthreads();

    const ColVal *col_val_u = reinterpret_cast<const ColVal *>(col_vals);

    for (u32 _row_id = 0; _row_id < TILE_SIZE; _row_id++) {
      u32 row_id = TILE_SIZE * row_tile + _row_id;
      int row_start = s_row_offsets[_row_id];
      int row_end = s_row_offsets[_row_id + 1];
      // TODO: Float version missing.
      half _sum{};
      for (int j = row_start + threadIdx.x; j < row_end; j += blockDim.x) {
#if 0
         short c_gt = col_ids[j];
         auto v_fp16_gt = values[j];
#endif
        u16 c = col_val_u[j].members.c;
        half v_fp16 = col_val_u[j].members.v;

#if 0
         assert(__heq(v_fp16_gt, v_fp16));
         assert(c_gt == c);
         c = c_gt;
         v_fp16 = v_fp16_gt;
#endif

        auto x_fp16 = _s_x[c];
        _sum = __hadd(_sum, __hmul(x_fp16, v_fp16));
      }

      half sum = shfl_reduce_half(_sum);
      if (threadIdx.x % WARP_SIZE == 0) {
        if constexpr (!std::is_same<Acc_t, float>::value) {
          atomicAdd(s_y_scalar + _row_id, sum);
        }
      }
    }
#endif

#if 0
     if constexpr (std::is_same<Vector_t, float2>::value) {
       // TODO: Float version missing.
     } else {
       half first = __shfl_down_sync((1 << BETA1) - 1, sum, threadIdx.x);
       half second = __shfl_down_sync((1 << BETA1) - 1, sum, threadIdx.x + 1);

       if (threadIdx.x < BETA1 / 2) {
         half2 result2 = make_half2(first, second);
         s_y[threadIdx.x] = __hadd2(result2, s_y[threadIdx.x]);
       }
     }
#endif

    __syncthreads();
#endif
    if (threadIdx.x < BETA1 / 2) {
      Vector_t res = s_y_vectorized[threadIdx.x];

      // Now res is either float2 or half2.
      half2 res_fp16;

      if constexpr (std::is_same<Vector_t, float2>::value) {
        res_fp16 = __float22half2_rn(res);
      } else {
        res_fp16 = res;
      }
      auto y_vectorized_global = reinterpret_cast<half2 *>(y);
      auto y_vectorized_local = y_vectorized_global + blockIdx.x * (BETA1 / 2);
      y_vectorized_local[threadIdx.x] = res_fp16;
    }
  }
#endif
}

//
template <int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH,
          int A, class Acc_t, class W_t /* = uint64_t */>
__global__ void spqr_quantized_matvec(
    // W and meta
    unsigned int prob_m, unsigned int prob_n,
    // W 1st order stats
    const W_t *__restrict__ raw_data,
    const SecondOrder *__restrict__ second_order_data,
    const half *__restrict__ X,
    // Output
    float *__restrict__ y,
    half *__restrict__ y_fp16,
    bool dense_only
        // Debug
        DEBUG_PARAMS_DECL) {
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
      .id = tile_id};

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

  IterX<BETA1, BETA2> iter_x{
      .x = X,
      .s_x = reinterpret_cast<half2 *>(_s_X),
      .num_x_halfs = prob_n,
      .num_x_half_shared = min(BLOCK_WIDTH * BETA2, prob_n),
      // NOTE: See [1]
      .num_participating_threads = _num_participating_threads};

  const int addr_per_row = MAX_ADDR_PER_ROW;
#if 0
     UPDIV(
     // Weight storage
     // min(BETA2, prob_n - subtile_id * BETA2) * BITS +
     BETA2, * BITS +
         // Weight + Scale
         2 * BITS,
     // u32 storage
     32);
#endif

  for (int i = subtile_id;; i += num_spqr_tiles_per_iteration) {
    u32 global_tile_id = blockId * num_spqr_tiles_per_cuda_block + i;

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
          dequantize2(first_order_quantized, iter_second_order.get_sws2(),
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
      clr_bless_async<Vector_t, ThreadDim::X>(s_y_vectorized, BETA1 / 2,
                                              BETA1 / 2, Vector_t());
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

  //  int _lock;
  //  do {
  //    asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(_lock) : "l"(lock));
  //  } while (_lock != prob_m);

  // At this point, the result is in s_y.
  if (threadIdx.x < BETA1) {
    if (dense_only) {
      y_fp16[blockIdx.x * BETA1 + threadIdx.x] = __float2half(s_y_scalar[threadIdx.x]);
    } else {
      y[blockIdx.x * BETA1 + threadIdx.x] = s_y_scalar[threadIdx.x];
    }
  }
}

template <typename T> __forceinline__ __device__ T myLoad(const T *d) {
  return *d;
}

__global__ void spmv_naive_single_thread(u32 m, u32 n,
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

__global__ void spmv_naive_mixed(uint32_t m, uint32_t n,
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

DEVICE_INLINE u16 get_col(u32 m) {
  return m & ((1u << 16u) - 1u);
}

DEVICE_INLINE half get_val(u32 m) {
  u16 _v = m >> 16u;
  half v = *reinterpret_cast<half*>(&_v);
  return v;
}

template <class Acc_t>
__global__ void spmv_naive_shared_sorted(uint32_t m, uint32_t n,
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
      y[row_id] =__float2half(sum + y_fp32[row_id]);
    }
  }
}

template <class Acc_t>
__global__ void spmv_naive_sparse_sorted(int offset, uint32_t m, uint32_t n,
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

__global__ void spmv_naive_shared_baseline(uint32_t m, uint32_t n,
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

__global__ void spmv_naive_shared(uint32_t m, uint32_t n,
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

__global__ void spmv_naive(uint32_t m, const int *__restrict__ row_offsets,
                           const short *__restrict__ col_ids,
                           const half *__restrict__ values,
                           const half *__restrict__ x, half *__restrict__ y) {
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

__global__ void spmv_naive_fp16(uint32_t m, const int *__restrict__ row_offsets,
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
template <typename T> __device__ inline T tempAtomicAdd(T *address, T val) {
  return atomicAdd(address, val);
}
#if __CUDA_ARCH__ < 600
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
template <>
__device__ inline double tempAtomicAdd<double>(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif

template <typename ValueType, typename IndexType, typename OffsetType>
__global__ void
spmvt(uint32_t num_non_zeroes, uint32_t out_size, uint32_t num_other,
      const ValueType *__restrict matrix, const IndexType *__restrict inIndex,
      const OffsetType *__restrict offsets, const ValueType *__restrict inVec,
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

template <class T> const T &__min(const T &a, const T &b) {
  return (b < a) ? b : a;
}

/*
 * Fused.
 */
#define SPQR_CALL_IF_FUSED(BITS, BETA1, BETA2, is_fp32)                        \
  else if (bits == BITS && beta1 == BETA1 && beta2 == BETA2) {                 \
    constexpr int BYTE = 8;                                                    \
    constexpr int VALS_PER_ADDR = (sizeof(int) * BYTE) / BITS;                 \
    constexpr int WEIGHT_COUNT = UPDIV((BETA1 * BETA2), VALS_PER_ADDR);        \
    constexpr int SHARED_MEM_SIZE =                                            \
        (BETA1 + BETA1 + BETA2 + WEIGHT_COUNT) * sizeof(int);                  \
    constexpr int BLOCK_HEIGHT = 1;                                            \
    constexpr int BLOCK_WIDTH = 8;                                             \
    const size_t SMEM_SIZE = sizeof(half) * prob_n;                            \
                                                                               \
    if (is_fp32) {                                                             \
      spqr_quantized_matvec_fused<BITS, BETA1, BETA2, BLOCK_HEIGHT,            \
                                  BLOCK_WIDTH, 32, float, uint64_t>            \
          <<<dim3(updiv(prob_m, BETA1 *BLOCK_HEIGHT), 1, 1),                   \
             dim3(__min(updiv(prob_n, BETA2), BLOCK_WIDTH) * 16, 1, 1),        \
             SMEM_SIZE, stream>>>(prob_m, prob_n, raw_data,                    \
                                  second_order_data_ptr, X_ptr,                \
                                  row_offsets_ptr, col_ptr_ptr, values_ptr,    \
                                  col_vals_ptr DEBUG_PARAMS_FP32);             \
    } else {                                                                   \
      spqr_quantized_matvec_fused<BITS, BETA1, BETA2, BLOCK_HEIGHT,            \
                                  BLOCK_WIDTH, 32, half2, uint64_t>            \
          <<<dim3(updiv(prob_m, BETA1 *BLOCK_HEIGHT), 1, 1),                   \
             dim3(__min(updiv(prob_n, BETA2), BLOCK_WIDTH) * 16, 1, 1),        \
             SMEM_SIZE, stream>>>(prob_m, prob_n, raw_data,                    \
                                  second_order_data_ptr, X_ptr,                \
                                  row_offsets_ptr, col_ptr_ptr, values_ptr,    \
                                  col_vals_ptr DEBUG_PARAMS_FP16);             \
    }                                                                          \
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

int preprocess_sparse(int m, int n,
                      // Outliers
                      void *values,
                      // 16-bit
                      void *row_offsets, void *col_ptr,
                      // 16-bit
                      void *X, void *Y, int nnz, void *cusparse_buffer) {
  half alpha = __float2half(1.f);
  half beta = __float2half(1.f);

  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void *dBuffer = nullptr;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle));
  // Create dense vector X
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, X, CUDA_R_32F))
  // Create dense vector y
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, Y, CUDA_R_32F))

  CHECK_CUSPARSE(cusparseCreateCsr(
      &matA, m, n, nnz, row_offsets, col_ptr, values, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))

  // Create sparse matrix A in CSR format
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  return 0;
}

static constexpr int DENSE_ONLY = 0;
static constexpr int NAIVE_SPARSE = 1;
static constexpr int DENSE_NAIVE_SPARSE = 2;
static constexpr int CUSPARSE = 3;
static constexpr int TILED_MATVEC = 4;

union Features {
  uint32_t _;
  struct {
    uint32_t is_fp32 : 1;
    uint32_t dense_only : 1;
    uint32_t naive_sparse : 1;
    uint32_t torch : 1;
    uint32_t is_async : 1;
    uint32_t shared_sparse : 1;
    uint32_t single_sparse : 1;
    uint32_t cusparse : 1;
    uint32_t fused_sparse : 1;
    uint32_t shared_sparse_baseline : 1;
    uint32_t shared_mixture : 1;
    uint32_t rest : 21;
  } flags;
};

#include <iostream>
namespace gpu {
// assert macro
#ifndef NDEBUG
#define gpu__assert(condition, message)                                        \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__         \
                << " line " << __LINE__ << ": " << message << std::endl;       \
      /*std::exit(EXIT_FAILURE); */ throw "Assertion error!";                  \
    }                                                                          \
  } while (false)
#else
#define gpu__assert(condition, message)                                        \
  do {                                                                         \
  } while (false)
#endif

// general types
typedef unsigned int ref_index;
typedef unsigned int memcount_t;
typedef unsigned int index_t;
typedef long count_t;

namespace internal {
static std::vector<count_t> refCounts;
static std::vector<index_t> freeIndexes;

index_t newReference() {
  if (!freeIndexes.empty()) {
    index_t r = freeIndexes[freeIndexes.size() - 1];
    freeIndexes.pop_back();
    refCounts[r] = 1; // init the new reference
    return r;
  }
  index_t ref_id = refCounts.size();
  refCounts.push_back(1); // init the new reference
  return ref_id;
}

void freeReference(index_t ref_id, void *ptr, void *h_ptr) {
  count_t newCount = --refCounts[ref_id];
  gpu__assert(newCount >= 0, "Count is negative!");
  if (newCount == 0) {
    // we should free it now
    cudaFree(ptr);
    // and make the reference index available
    freeIndexes.push_back(ref_id);
    // std::cout << "Free indexes: " << freeIndexes.size() << std::endl;
    delete[] h_ptr;
  }
}

// return the number of references which are still alive (entries, not their
// count)
count_t countReferences() {
  return count_t(refCounts.size()) - freeIndexes.size();
}
} // namespace internal

template <class S> class CudaMemory {
public:
  // template types
  typedef CudaMemory<S> this_type;
  typedef S scalar_type;
  S *h_ptr;

  // constructors

  CudaMemory() : count(0), d_ptr(0), ref_id(0), h_ptr(nullptr) {
    // std::cout << "CudaMemory()" << std::endl;
  }

  CudaMemory(memcount_t c) : count(0), d_ptr(0), ref_id(0) {
    //     std::cout << "CudaMemory(" << c << ") - ";
    // trying to allocate the data
    if (cudaMalloc((void **)&d_ptr, sizeof(scalar_type) * c) == cudaSuccess) {
      // now we can do something
      count = c;
      ref_id = internal::newReference();
      // std::cout << "worked!" << std::endl;
    } else {
      printf("cuMalloc error!");
      exit(1);
    }
  }
  // copy

  CudaMemory(const CudaMemory::this_type &orig)
      : count(orig.count), d_ptr(orig.d_ptr), ref_id(orig.ref_id) {
    // std::cout << "Copying (size=" << count << ", ref=" << orig.ref_count() <<
    // ")" << std::endl;
    //     update reference count
    if (count > 0) {
      // let's increment the count
      ++internal::refCounts[ref_id];
    }
    // std::cout << "... now ref=" << ref_count() << std::endl;
  }

  this_type &operator=(const this_type &other) {
    // std::cout << "Assigning: size=" << other.count << std::endl;
    // free current content
    if (count > 0) {
      internal::freeReference(ref_id, (void *)d_ptr, (void *)h_ptr);
      count = 0;
    }
    // copy new content
    count = other.count;
    d_ptr = other.d_ptr;
    ref_id = other.ref_id;
    // update reference count
    if (count > 0) {
      ++internal::refCounts[ref_id];
    }
    // std::cout << "... now ref=" << ref_count() << std::endl;
  }

  // free

  ~CudaMemory() {
    return;
    // std::cout << "Deleting for count=" << count << std::endl;
    // free content
    if (count > 0) {
      internal::freeReference(ref_id, (void *)d_ptr, (void *)h_ptr);
    }
  }

  // transfers

  void copyFrom(S *h_ptr) {
    gpu__assert(count > 0, "Nothing to copy!");
    cudaMemcpy(d_ptr, h_ptr, sizeof(scalar_type) * count,
               cudaMemcpyHostToDevice);
  }

  void copyTo(S *h_ptr) {
    gpu__assert(count > 0, "Nothing to copy!");
    cudaMemcpy(h_ptr, d_ptr, sizeof(scalar_type) * count,
               cudaMemcpyDeviceToHost);
  }

  void to_host() {
    h_ptr = new S[count];
    copyTo(h_ptr);
  }

  // overloading i/o operators
  template <typename T> friend this_type &operator<<(this_type &mem, const T &);
  template <typename T> friend this_type &operator>>(this_type &mem, T &);

  // getters

  bool empty() const { return count == 0; }

  scalar_type *get() { return d_ptr; }
  scalar_type *get_host() { return h_ptr; }

  count_t ref_count() const {
    if (count > 0)
      return internal::refCounts[ref_id];
    else
      return 0;
  }

  memcount_t size() const { return count; }

  // implicit conversion

  /* operator scalar_type*() {
      return d_ptr;
  } */
private:
  memcount_t count;
  scalar_type *d_ptr;
  ref_index ref_id;
};

// names
typedef CudaMemory<half> CudaHalfMemory;
typedef CudaMemory<char> CudaCharMemory;
typedef CudaMemory<int> CudaIntMemory;
typedef CudaMemory<long> CudaLongMemory;
typedef CudaMemory<float> CudaFloatMemory;
typedef CudaMemory<double> CudaDoubleMemory;

} // namespace gpu

int spqr_matvec(
    // W and meta
    int bits, int prob_m, int prob_n,
    // Quantization
    int beta1, int beta2, const void *_raw_data, const void *second_order_data,
    void *row_ids,
    // 32-bit
    void *row_offsets,
    // 16-bit
    void *col_vals, int nnz, int dense_row_count,
    // 16-bit
    // Input
    void *X,
    // Output
    void *y, cudaStream_t stream, void *measurements,
    uint32_t feature_flag = 0) {
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

  int tot_m = prob_m;
  int tot_m_blocks = UPDIV(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  const uint64_t *raw_data = (const uint64_t *)_raw_data;
  const half *X_ptr = (const half *)X;
  const int *row_offsets_ptr = (const int *)row_offsets;
  const short *row_ids_ptr = (short *)row_ids;
  half *y_ptr = (half *)y;
  const SecondOrder *second_order_data_ptr =
      (const SecondOrder *)second_order_data;
  const u32 *col_vals_ptr = (const u32 *)col_vals;

  float *d_yfp32;
  cudaMalloc((void **)&d_yfp32, sizeof(float) * prob_m);
  cudaMemset(d_yfp32, 0, sizeof(float) * prob_m);
  cudaDeviceSynchronize();

#if 0
 auto tile_row_offsets_ptr = (const int *)tile_row_offsets;
 auto tile_col_ids_ptr = (const short *)tile_col_ids;
 auto tile_nnzs_ptr = (const int *)tile_nnzs;
 auto tile_data_ptr = (unsigned int *)tile_data;
#endif

  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void *dBuffer = nullptr;
  size_t bufferSize = 0;
  half alpha = __float2half(1.f);
  half beta = __float2half(1.f);

  int ret = 0;

#if DBG
  gpu::CudaHalfMemory dbg_deq_w(buffer_size);
  gpu::CudaHalfMemory dbg_first(buffer_size);
  gpu::CudaHalfMemory dbg_second(buffer_size);
#endif

  Timer *timer;
  if (measurements) {
    timer = new Timer(stream);
    timer->start();
  }

  int pad_m = updiv(prob_m, beta1) * beta1;
  int pad_n = updiv(prob_n, beta2) * beta2;

  int tile_m = updiv(prob_m, beta1);
  int tile_n = updiv(prob_n, beta2);

  int buffer_size = pad_m * pad_n;

  constexpr int BYTE = 8;
  constexpr int VALS_PER_ADDR = (sizeof(int) * BYTE) / 3;
  constexpr int WEIGHT_COUNT = UPDIV((16 * 16), VALS_PER_ADDR);
  constexpr int SHARED_MEM_SIZE = (16 + 16 + 16 + WEIGHT_COUNT) * sizeof(int);
  constexpr int BLOCK_HEIGHT = 1;
  constexpr int BLOCK_WIDTH = 8;
  spqr_quantized_matvec<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, 32, float, uint64_t> <<<dim3(updiv(prob_m, 16 * BLOCK_HEIGHT), 1, 1), dim3(__min(updiv(prob_n, 16), BLOCK_WIDTH) * 16, 1, 1), 0, stream>>>( prob_m, prob_n, raw_data, second_order_data_ptr, X_ptr, d_yfp32, y_ptr, dense_only DEBUG_PARAMS_FP32);


  if (ret) {
    cudaFree(d_yfp32);
    cudaStreamDestroy(stream_sparse0);
    cudaStreamDestroy(stream_sparse);
    return ret;
  }

  if (!dense_only) {
    CHECK_CUDA(cudaDeviceSynchronize());

    if (features.flags.shared_mixture && nnz) {
      constexpr int WARP_SIZE = 32;
      constexpr int BLOCK_HEIGHT = 4;

      int sparse_row_count = prob_m - dense_row_count;
      if (sparse_row_count) {
        spmv_naive_sparse_sorted<float>
            <<<UPDIV(prob_m - dense_row_count, WARP_SIZE), WARP_SIZE, 0,
               stream_sparse>>>(dense_row_count, prob_m, prob_n, row_ids_ptr,
                                row_offsets_ptr, col_vals_ptr, X_ptr, y_ptr, d_yfp32);
      }

      if (dense_row_count) {
        size_t smem_size = sizeof(half) * prob_n;
        spmv_naive_shared_sorted<float>
            <<<UPDIV(dense_row_count, BLOCK_HEIGHT),
               dim3(WARP_SIZE, BLOCK_HEIGHT, 1), smem_size, stream_sparse0>>>(
                dense_row_count, prob_n, row_ids_ptr, row_offsets_ptr,
                col_vals_ptr, X_ptr, y_ptr, d_yfp32);
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
    ((float *)measurements)[0] = timer->end();
    delete timer;
  }

  cudaFree(d_yfp32);

  return ret;
}
