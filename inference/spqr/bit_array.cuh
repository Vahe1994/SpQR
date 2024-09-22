#pragma once

#include <cuda_fp16.h>
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __forceinline__
#define __forceinline__
#endif

#include <cstdint>
#include <cstdio>
#include <type_traits>

template <class T> __host__ __device__ constexpr int get_bits() {
  if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>) {
    return 32;
  } else {
    return 64;
  }
}

template <class Bit_t, unsigned BITS> struct TileArray {
private:
  Bit_t *buffer;

public:
  Bit_t *ptr;
  explicit TileArray(Bit_t *ptr) : ptr(ptr), buffer(ptr) {}

  void push(Bit_t s, Bit_t z, Bit_t *x, int n) {
    Bit_t b = (z << BITS) | s;
    int i = 0, j = 2;

    constexpr int VALUES_PER_ADDR = get_bits<Bit_t>() / BITS;
    const int VALUES_TO_ADD = n;

    Bit_t _b;
    for (; i < VALUES_TO_ADD;) {
      for (; j < VALUES_PER_ADDR && i < VALUES_TO_ADD; j++, i++) {
        b |= (x[i] << (j * BITS));
      }
      *buffer = b;
      _b = b;
      b = 0;
      j = 0;
      buffer++;
    }
  }
};

template <class Bit_t, uint64_t BITS> struct BitArray {
  const Bit_t *w;

  __host__ __device__ Bit_t operator[](int w_id) {
    // TODO: NOTE: PERF: Make this decode faster!
    int addr = (w_id / 18);
    return (w[addr] >> ((w_id % 18ull) * BITS)) & ((1ull << BITS) - 1ull);
  }
};

template <class Bit_t> struct _BitArray {
  Bit_t *w{};
  const int bits;
  Bit_t *out;
  int local_id{};
  int addr_per_block;

  _BitArray(Bit_t *w, int bits) : w(w), bits(bits), out(nullptr), local_id{} {
    constexpr unsigned BYTE = 8u;
    constexpr unsigned BITS_PER_ADDR = sizeof(Bit_t) * BYTE;
    addr_per_block = BITS_PER_ADDR / bits;
  }

  __host__ __device__ Bit_t operator[](int w_id) {
    // TODO: NOTE: PERF: Make this decode faster!
    int addr = (w_id / 18);
    return (w[addr] >> ((w_id % 18ull) * bits)) & ((1ull << bits) - 1ull);
  }

  [[maybe_unused]] int sanity{};

  __device__ __host__ void push_back(Bit_t x) {
    if (out == nullptr) {
      out = w;
    }
    (*out) |= (x << (local_id * bits));
    if (local_id == addr_per_block - 1) {
      *(++out) = 0;
      local_id = 0;
      sanity++;
    } else {
      local_id++;
    }
  }

  __device__ __host__ void pad(int times) {
    // Move to the next block if necessary
    for (int i = 0; i < times; i++) {
      *(++out) = 0;
      sanity++;
    }
    local_id = 0;
  }

  __device__ __host__ void pad_maybe() {
    // Move to the next block if necessary
    if (local_id != 0) {
      pad(1);
    }
  }
};

union ColVal {
  uint32_t _;
  struct {
    short c;
    half v;
  } members;
};

union SecondOrder {
  uint64_t v;
  struct SO {
    half2 ss;
    half2 zz;
  } members;


  __device__ __forceinline__ half2 get_sws2() const { return members.ss; }

  __device__ __forceinline__ half2 get_swz2() const { return members.zz; }
};
