#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "engine/cuda/cuda_defs.h"

namespace CUDA
{
    // __clz: Returns the number of consecutive high - order zero bits
    __device__ __forceinline__ int countLeadingZeros(int x) { return __clz(x); }
    __device__ __forceinline__ int countLeadingZeros(long long int x) { return __clzll(x); }
    __device__ __forceinline__ int countLeadingZeros(uint32_t x) { return __clz(x); }
    __device__ __forceinline__ int countLeadingZeros(uint64_t x) { return __clzll(x); }

    /*
    * Quantization of given floating point x in [0,1] to an integer in [0,...,n)
    */
    inline HOST_AND_DEVICE uint32_t quantize(float x, uint32_t n)
    {
        return max(min(static_cast<int32_t>(x * n), static_cast<int32_t>(n - 1)), int32_t(0));
    }

    template<typename T>
    __device__ inline void swap(T& v0, T& v1)
    {
        T tmp = v0;
        v0 = v1;
        v1 = tmp;
    }
}