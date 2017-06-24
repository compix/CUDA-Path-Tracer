#pragma once
#include "cuda_runtime.h"
#include <glm/ext.hpp>

namespace CUDA
{
    /**
    * Given values of vertices of a triangle v0, v1, v2 and barycentric coordinates uv,
    * the barycentric interpolation computes v0 * (1.0f - uv.x - uv.y) + v1 * uv.x + v2 * uv.y
    */
    template<class T>
    __device__ T barycentricLerp(
        const T& v0, const T& v1, const T& v2,
        const glm::vec2& uv)
    {
        return v0 * (1.0f - uv.x - uv.y) + v1 * uv.x + v2 * uv.y;
    }

    /**
    * Input: values v0, v1, v2, v3 of corners of a rectangle:
    * v3 ___ v2
    * |       |
    * |       |
    * v0 ___ v1
    * x,y in [0,1]
    */
    template<class T>
    __device__ T bilinearLerp(const T& v0, const T& v1, const T& v2, const T& v3, float x, float y)
    {
        return glm::lerp(glm::lerp(v0, v1, x), glm::lerp(v3, v2, x), y);
    }
}
