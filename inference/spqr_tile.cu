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

template <class Bit_t, uint64_t BITS>
__forceinline__ __host__ __device__ Bit_t get_bit(Bit_t w, Bit_t w_id) {
 return (w >> (w_id * BITS)) & ((1ull << BITS) - 1ull);
}

using u32 = unsigned int;

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

__device__ __host__ CUINLINE int updiv(int x, int y) { return (x + y - 1) / y; }

/*
* TODO:
* 1) Write out an example in docstring here.
* 2) Possibly make this much faster with warp-level bit-reduction if necessary.
* 3) Test.
*/
template <class Input_t, class Output_t, int BITS>
__global__ void pack_bits(const Input_t *in, int n, const Output_t *out) {
 int id = blockIdx.x * blockDim.x + threadIdx.x;

 if (id >= n) {
   return;
 }

 auto val = in[id];

 // Figure out the number of bits per output address.
 constexpr int BYTE = 8;
 constexpr int OUT_BITS = sizeof(Output_t) * BYTE;

 // Then we find the number of inputs that can fit in one output address.
 auto INPUTS_PER_OUTPUT = OUT_BITS / BITS;

 // Find the output address.
 auto out_addr = updiv(id, INPUTS_PER_OUTPUT);

 auto pos_in_output_cell = id % INPUTS_PER_OUTPUT;

 atomicOr(out + out_addr, (val << (pos_in_output_cell * BITS)));
}

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
   return threadIdx.x * blockDim.x * blockDim.y +
          (threadIdx.y * blockDim.z + threadIdx.z);
 } else if constexpr (t == ThreadDim::YZX) {
   return ((threadIdx.y * blockDim.z + threadIdx.z)) * blockDim.x +
          threadIdx.x;
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

 void start() { AT_CUDA_CHECK(cudaEventRecord(ce_start, 0)); }

 float end() {
   float time;
   AT_CUDA_CHECK(cudaEventRecord(ce_stop, 0));
   AT_CUDA_CHECK(cudaEventSynchronize(ce_stop));
   AT_CUDA_CHECK(cudaEventElapsedTime(&time, ce_start, ce_stop));
   // Returns ms
   return time;
 }

 Timer() {
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

// Keep the second order data in shared memory.

#if 0
template<
 int BITS,
 int BETA1,
 int BETA2,
 int BLOCK_HEIGHT,
 int BLOCK_WIDTH,
 int A,
 class W_t = uint32_t> __global__ void SPQR(
 // W and meta
 int prob_m,
 int prob_n,
 // W 1st order stats
 const uint32_t *__restrict__ raw_data,
 const half *__restrict__ X,
 // Outliers
 const half *values,
 // 16-bit
 const int *row_offsets,
 // 32-bit
 const short *col_ptr,
 // extra global storage for barrier synchronization
 // Output
 half *__restrict__ y
) {
 constexpr int TILE_SIZE = 32; // UPDIV((BETA2 * BITS) + 2 * BITS, 32) * BETA1;
 __shared__ uint32_t _smem[TILE_SIZE * BLOCK_HEIGHT * BLOCK_WIDTH];
 __shared__ half _s_X[BLOCK_WIDTH * BETA2];
 __shared__ half _s_y[BLOCK_HEIGHT * BETA1];

 __shared__ half _s_W_s[BLOCK_HEIGHT * BLOCK_WIDTH * BETA1];
 __shared__ half _s_W_z[BLOCK_HEIGHT * BLOCK_WIDTH * BETA1];

 // Load first order meta here.
 constexpr int BYTE = 8;
 // For example, if BITS == 3, then VALS_PER_ADDR = 10
 constexpr int VALS_PER_ADDR = (sizeof(int) * BYTE) / BITS;

 // Load 2nd order meta first

 int total_tile_rows = updiv(prob_m, BETA1);
 int total_tile_cols = updiv(prob_n, BETA2);

 int tile_row = BLOCK_HEIGHT * blockIdx.x + threadIdx.y;
 int tile_col = BLOCK_WIDTH * blockIdx.y + threadIdx.z;

 int tile_id = tile_row * total_tile_cols + tile_col;

 auto smem = _smem + (threadIdx.y * BLOCK_WIDTH + threadIdx.z) * TILE_SIZE;

 auto s_y = _s_y + threadIdx.y * BETA1;
 auto s_X = _s_X + threadIdx.z * BETA2;

 static constexpr int WARP_COUNT = BETA1;

 constexpr int THREAD_COUNT = WARP_COUNT;
 constexpr int X_VALUES_TO_READ_WRITE = UPDIV(BETA2, THREAD_COUNT);

#pragma unroll
 for (int i = 0; i < X_VALUES_TO_READ_WRITE; i++) {
 unsigned int global_id = blockIdx.y * BETA2 + threadIdx.x * X_VALUES_TO_READ_WRITE + i;
 if (global_id < prob_n) {
   s_X[threadIdx.x * X_VALUES_TO_READ_WRITE + i] = X[global_id];
 } else {
   s_X[threadIdx.x * X_VALUES_TO_READ_WRITE + i] = half{};
 }
 }

 int subtile_id = threadIdx.y * BLOCK_WIDTH + threadIdx.z;

 half *s_W_s = _s_W_s + subtile_id * BETA1;
 half *s_W_z = _s_W_z + subtile_id * BETA1;

 auto run_tile = [&]() {
 unsigned int access =
   ((blockIdx.x * BLOCK_HEIGHT + threadIdx.y) * total_tile_cols + (blockIdx.y * BLOCK_WIDTH + threadIdx.z))
     * TILE_SIZE + threadIdx.x;
 smem[threadIdx.x] = raw_data[access];

 BitArray<W_t, BITS> W{.w = smem};

 // Second order.

 __syncwarp();

 uint32_t *sorder = smem + TILE_SIZE - 2;
 half2 *second_order_raw = reinterpret_cast<half2 *> (sorder);
 half2 Wscales = make_half2(second_order_raw[0].x, second_order_raw[1].x);
 half2 Wzeros = make_half2(second_order_raw[0].y, second_order_raw[1].y);

 unsigned int tix = get_thread_id<ThreadDim::X>();
 if (tix < BETA1) {
   int row_id = tix;
   half2 w = make_half2(__int2half_rd(static_cast<int>(W[BETA1 * BETA2 + 2 * row_id])),
              __int2half_rd(static_cast<int>(W[BETA1 * BETA2 + 2 * row_id + 1])));

   half2 res = dequantize2(w, Wscales, Wzeros);

   s_W_s[tix] = res.x;
   s_W_z[tix] = res.y;
 }

 if (threadIdx.x < BETA1 / 2) {
   clr_bless_async<half2, ThreadDim::X>(reinterpret_cast<half2 *>(s_y), BETA1 / 2);
 }

 constexpr int weights_per_thread = UPDIV(BETA1 * BETA2, WARP_COUNT);

 __syncwarp();

 int base_row = get_thread_id<ThreadDim::X>() * weights_per_thread / BETA2;
 half2 sy2{};
 // NOTE: Lots of assumptions made here
 int row_id = (tix * weights_per_thread) >> 4;

 half2 sws2 = __half2half2(s_W_s[row_id]);
 half2 swz2 = __half2half2(s_W_z[row_id]);

 auto col_id = ((tix * weights_per_thread) & ((1 << 4) - 1));
 auto sx2 = reinterpret_cast<half2 *>(s_X + col_id);
 unsigned int start_weight = row_id * BETA2 + col_id;

#pragma unroll
 for (int i = start_weight; i < (start_weight + weights_per_thread); i += 2, sx2++) {
   half2 w2 = make_half2(
     __int2half_rd(static_cast<W_t>(W[i])),
     __int2half_rd(static_cast<W_t>(W[i + 1]))
   );
   half2 w = dequantize2(w2, sws2, swz2);
   sy2 = __hfma2(w, *sx2, sy2);
 }

 atomicAdd(s_y + row_id, __hadd(sy2.x, sy2.y));

 __syncwarp();

 };

 run_tile();

 constexpr int WARP_SIZE = BETA1;
 int thread_id = threadIdx.y * BLOCK_WIDTH * WARP_SIZE + threadIdx.z * WARP_SIZE + threadIdx.x;

 constexpr int BLOCKWIDE_THREAD_COUNT = WARP_SIZE * BLOCK_WIDTH * BLOCK_HEIGHT;
 constexpr int Y_VALUES_TO_READ_WRITE = UPDIV(BETA1 * BLOCK_HEIGHT, BLOCKWIDE_THREAD_COUNT);

#pragma unroll
 for (int i = 0; i < Y_VALUES_TO_READ_WRITE; i++) {
 unsigned int local_id = thread_id * Y_VALUES_TO_READ_WRITE + i;
 unsigned int global_id = (BLOCK_HEIGHT * blockIdx.x + threadIdx.y) * BETA1 + local_id;
 if (global_id < prob_m && local_id < BETA1 * BLOCK_HEIGHT) {
   atomicAdd(y + global_id, s_y[local_id]);
 }
 }
}

template<
 int BITS,
 int BETA1,
 int BETA2,
 int BLOCK_HEIGHT,
 int BLOCK_WIDTH,
 int A,
 class W_t = uint32_t> __global__ void SPQR_v2(
 // W and meta
 int prob_m,
 int prob_n,
 // W 1st order stats
 const uint32_t *__restrict__ raw_data,
 const uint32_t *__restrict__ second_order_data,
 const half *__restrict__ X,
 // Outliers
 const half *values,
 // 16-bit
 const int *row_offsets,
 // 32-bit
 const short *col_ptr,
 // extra global storage for barrier synchronization
 // Output
 half *__restrict__ y
) {
 constexpr int ADDR_PER_ROW = UPDIV((BETA2 * BITS) + 2 * BITS, 32);
 constexpr int TILE_SIZE = UPDIV((BETA2 * BITS) + 2 * BITS, 32) * BETA1;

 __shared__ half _s_X[BLOCK_WIDTH * BETA2];
 __shared__ half _s_y[BLOCK_HEIGHT * BETA1];

 __shared__ half2 s_W[BLOCK_HEIGHT * BLOCK_WIDTH];

 // Read X
 constexpr int THREADS_PER_TILE = BETA1;

 constexpr int TOTAL_THREADS = THREADS_PER_TILE * BLOCK_HEIGHT * BLOCK_WIDTH;

 int total_tile_rows = updiv(prob_m, BETA1);
 int total_tile_cols = updiv(prob_n, BETA2);

 int base_tile_id = blockIdx.x * total_tile_cols + blockIdx.y;

 // Load first order meta here.
 constexpr int BYTE = 8;

 // For example, if BITS == 3, then VALS_PER_ADDR = 10
 constexpr int VALS_PER_ADDR = (sizeof(int) * BYTE) / BITS;

 // Global shared memory ids to tile.
 half *s_y = _s_y + threadIdx.y * BETA1;
 half *s_X = _s_X + threadIdx.z * BETA2;

 int tile_row = BLOCK_HEIGHT * blockIdx.x + threadIdx.y;
 int tile_col = BLOCK_WIDTH * blockIdx.y + threadIdx.z;

 int tile_id = tile_row * total_tile_cols + tile_col;

 int thread_id_yzx = (threadIdx.y * BLOCK_WIDTH + threadIdx.z) * blockDim.x + threadIdx.x;

 // Load X/y into shared memory
 {
 half2 *s_x2 = reinterpret_cast<half2 *>(_s_X);
 const half2 *X2 = reinterpret_cast<const half2 *>(X + blockIdx.z * BETA2);
 constexpr int S_X_SIZE = BLOCK_WIDTH * BETA2;
 constexpr int S_X2_SIZE = (S_X_SIZE) / 2;

 int x_local_size = min(prob_n - blockIdx.y * BETA2, BLOCK_WIDTH * BETA2);
 int x2_values_to_write = UPDIV(S_X2_SIZE, TOTAL_THREADS);


 // TODO: Does the case where X is odd crash this?
#pragma unroll
 for (int i = 0; i < x2_values_to_write; i++) {
   unsigned int local_id = thread_id_yzx * x2_values_to_write + i;
   if (local_id < x_local_size) {
   s_x2[local_id] = X2[local_id];
   }
 }

 // Load second order data
 if (threadIdx.x < 2 * BLOCK_WIDTH && !threadIdx.y && !threadIdx.z) {
   for (int i = 0; i < BLOCK_HEIGHT; i++) {
   *reinterpret_cast<uint32_t *>(&s_W[i * 2 * BLOCK_WIDTH + threadIdx.x]) =
     second_order_data[((BLOCK_HEIGHT * blockIdx.x + i) * total_tile_cols + blockIdx.y * BLOCK_WIDTH) * 2
       + threadIdx.x];
   }
 }
 }

 // Zero out y
 {
 if (!threadIdx.y && !threadIdx.z && threadIdx.x < BLOCK_HEIGHT * BETA1 / 2) {
   clr_bless_async<half2, ThreadDim::X>(reinterpret_cast<half2 *>(s_y), BLOCK_HEIGHT * BETA1 / 2);
 }
 }

 // This thread is now processing row_id in tile_id
 int row_id = thread_id_yzx / BETA1;

 uint32_t registers[ADDR_PER_ROW];

#pragma unroll
 for (int i = 0; i < ADDR_PER_ROW; i++) {
 registers[i] = raw_data[TILE_SIZE * tile_id + i];
 }

 BitArray<W_t, BITS> W{.w = registers};

 half2 first_order_quantized = make_half2(__int2half_rd(static_cast<int>(W[0])),
                      __int2half_rd(static_cast<int>(W[1])));

 __syncwarp();
 half2 res = dequantize2(first_order_quantized,
             s_W[2 * (threadIdx.y * blockDim.z + threadIdx.z)],
             s_W[2 * (threadIdx.y * blockDim.z + threadIdx.z) + 1]);

 half2 ws2 = __half2half2(res.x);
 half2 wz2 = __half2half2(res.y);
 half2 sy2{};

 half2 *s_x2 = reinterpret_cast<half2 *>(_s_X + threadIdx.z * BETA2);
 const half2 *X2 = reinterpret_cast<const half2 *>(X + blockIdx.z * BETA2);

 for (int i = 0; i < BETA2; i += 2, s_x2++) {
 half2 w2 = make_half2(
   __int2half_rd(static_cast<W_t>(W[2 + i])),
   __int2half_rd(static_cast<W_t>(W[2 + i + 1]))
 );
 half2 w = dequantize2(w2, ws2, wz2);
 sy2 = __hfma2(w, *s_x2, sy2);
 }

 atomicAdd(s_y + row_id, __hadd(sy2.x, sy2.y));

 __syncwarp();

 constexpr int WARP_SIZE = BETA1;
 int thread_id = threadIdx.y * BLOCK_WIDTH * WARP_SIZE + threadIdx.z * WARP_SIZE + threadIdx.x;

 constexpr int BLOCKWIDE_THREAD_COUNT = WARP_SIZE * BLOCK_WIDTH * BLOCK_HEIGHT;
 constexpr int Y_VALUES_TO_READ_WRITE = UPDIV(BETA1 * BLOCK_HEIGHT, BLOCKWIDE_THREAD_COUNT);

#pragma unroll
 for (int i = 0; i < Y_VALUES_TO_READ_WRITE; i++) {
 unsigned int local_id = thread_id * Y_VALUES_TO_READ_WRITE + i;
 unsigned int global_id = (BLOCK_HEIGHT * blockIdx.x + threadIdx.y) * BETA1 + local_id;
 if (global_id < prob_m && local_id < BETA1 * BLOCK_HEIGHT) {
   atomicAdd(y + global_id, s_y[local_id]);
 }
 }
}
#endif
