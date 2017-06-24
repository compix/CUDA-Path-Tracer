#pragma once
#include "TriangleInfo.h"

// Convenience structure to be passed to device functions
struct ShadingData
{
    const CUDA::TriangleInfo* triangleInfo;
    cudaTextureObject_t* textures;

    ShadingData(const CUDA::TriangleInfo* triangleInfo)
        :triangleInfo(triangleInfo) {}

    HOST_AND_DEVICE ShadingData() {}
};
