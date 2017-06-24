#pragma once
#include <algorithm>

#define CUDA_ERROR_CHECK(errCode) { cudaAssert((errCode), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t errCode, const char *file, int line, bool abort = false)
{
	if (errCode != cudaSuccess)
	{
		fprintf(stderr, "\nCUDA ERROR: %s %s %d\n", cudaGetErrorString(errCode), file, line);
		if (abort) exit(errCode);
	}
}

namespace CUDA
{
    /**
     * Creates an instance of the given type on host and copies it over to device memory.
     */
    template<typename T, typename... Args>
    T* create(Args&&... args)
    {
        T hostData(std::forward<Args>(args)...);
        T* deviceData;
        cudaMalloc(&deviceData, sizeof(T));
        cudaMemcpy(deviceData, &hostData, sizeof(T), cudaMemcpyHostToDevice);
        return deviceData;
    }
}
