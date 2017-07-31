#pragma once
#include "cuda_runtime.h"
#include <glm/ext.hpp>

namespace CUDA
{
    /*
    * Quantization of given floating point x in [0,1] to an integer in [0,...,n)
    */
    inline HOST_AND_DEVICE uint32_t quantize(float x, uint32_t n)
    {
        return glm::max(glm::min(static_cast<int32_t>(x * n), static_cast<int32_t>(n - 1)), int32_t(0));
    }

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

    /**
    * theta = azimuth
    * phi = elevation
    */
    __device__ inline glm::vec3 sphericalToDirection(float sintheta, float costheta, float phi)
    {
        return glm::vec3(
            -sintheta * cosf(phi),
            sinf(phi),
            costheta * cosf(phi));
    }

    /**
    * Transform a given vector to the hemisphere given by normal.
    */
    __device__ inline glm::vec3 toHemisphere(const glm::vec3& vector, const glm::vec3& normal)
    {
        glm::vec3 x;
        if (glm::abs(normal.x) < FLT_EPSILON)
            x = glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f));
        else
            x = glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f));

        x = glm::normalize(x);
        glm::vec3 y = glm::normalize(glm::cross(normal, x));
        return vector.x * x + vector.y * y + vector.z * normal;
    }

    /**
    * Transform a given vector to the hemisphere given by normal.
    */
    __device__ inline glm::vec3 toHemisphere(const glm::vec3& vector, const glm::vec3& normal, const glm::vec3& tangent, const glm::vec3& bitangent)
    {
        return vector.x * tangent + vector.y * bitangent + vector.z * normal;
    }
}
