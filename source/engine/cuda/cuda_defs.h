#pragma once

//#ifdef __CUDACC__
//#define GLM_FORCE_CUDA
//#endif

// CUDA + C++ (mark as __device__ and __host__):
// http://stackoverflow.com/questions/6978643/cuda-and-classes
#ifdef __CUDACC__
#define HOST_AND_DEVICE __host__ __device__
#else
#define HOST_AND_DEVICE
#endif 

#define CUDA_EPSILON 1e-7f
#define CUDA_EPSILON6 1e-6f
#define CUDA_EPSILON5 1e-5f
#define CUDA_EPSILON4 1e-4f
#define CUDA_EPSILON3 1e-3f
#define CUDA_EPSILON2 1e-2f
#define MAX_TRAVERSAL_STACK_SIZE 64
#define CUDA_PI 3.14159265359f
#define CUDA_2PI 6.28318530718f
#define CUDA_PI_INV 0.31830988618f
#define CUDA_2PI_INV 0.15915494309f
