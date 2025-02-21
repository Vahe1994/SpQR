#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __forceinline__
#define __forceinline__
#endif

#include <cstdio>

using u64 = unsigned long long;
using s32 = int;
using s64 = long long int;
using u32 = unsigned int;
using u16 = unsigned short;

static constexpr u64 SECOND_ORDER_FRAGMENT_SIZE_BITS = 8ull;

template<class T> __host__ __device__ constexpr int get_bits() {
  if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>) {
    return 32;
  } else {
    return 64;
  }
}

template<class Bit_t, unsigned BITS> struct TileArray {
private:
  Bit_t *buffer{};

public:
  Bit_t *ptr{};

  explicit TileArray(Bit_t *_ptr) : buffer(_ptr), ptr(_ptr) {}

  void push(Bit_t s, Bit_t z, Bit_t *x, int n, Bit_t buff = Bit_t{}) {
    Bit_t b = (z << BITS) | s;
    int i = 0, j = 2;

    constexpr int VALUES_PER_ADDR = get_bits<Bit_t>() / BITS;
    const int VALUES_TO_ADD = n;

    for (; i < VALUES_TO_ADD;) {
      for (; j < VALUES_PER_ADDR && i < VALUES_TO_ADD; j++, i++) {
        b |= (x[i] << (j * BITS));
      }
      *buffer = b | buff;
      b = 0;
      j = 0;
      buffer++;
    }
  }
};

template<class Bit_t> struct _BitArray {
  Bit_t *w{};
  const int bits;
  Bit_t *out;

  _BitArray(Bit_t *w, int bits) : w(w), bits(bits), out(nullptr) {}

  __host__ __device__ Bit_t operator[](int w_id) {
    int addr = (w_id / 18);
    return (w[addr] >> ((w_id % 18ull) * bits)) & ((1ull << bits) - 1ull);
  }
};

union ColVal {
  u32 _;

  struct {
    unsigned short c;
    half v;
  } members;
};

union SecondOrder {
  u64 v;

  struct SO {
    half2 ss;
    half2 zz;
  } members;

  __device__ __forceinline__ half2 get_sws2() const { return members.ss; }

  __device__ __forceinline__ half2 get_swz2() const { return members.zz; }
};

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

struct Timer {
  cudaEvent_t ce_start{}, ce_stop{};
  cudaStream_t stream;

  inline void start() { cudaEventRecord(ce_start, stream); }

  inline float end_and_measure() {
    float time_ms{};
    cudaEventRecord(ce_stop, nullptr);
    cudaEventSynchronize(ce_stop);
    cudaEventElapsedTime(&time_ms, ce_start, ce_stop);
    // Returns ms
    return time_ms;
  }

  inline Timer(cudaStream_t stream) : stream(stream) {
    cudaEventCreate(&ce_start);
    cudaEventCreate(&ce_stop);
  }

  inline Timer(Timer &&timer) = delete;

  inline Timer(const Timer &timer) = delete;

  inline ~Timer() {
    cudaEventDestroy(ce_start);
    cudaEventDestroy(ce_stop);
  }
};
