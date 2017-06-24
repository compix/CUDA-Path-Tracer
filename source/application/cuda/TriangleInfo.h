#pragma once
#include <glm/detail/type_vec3.hpp>
#include <glm/detail/type_vec2.hpp>
#include <limits>
#include <stdint.h>
#include <cuda_runtime.h>
#include "math_util.h"

const uint32_t INVALID_CUDA_TEX_INDEX = std::numeric_limits<uint32_t>().max();
#define MAX_SPECULARITY 1024.0f // = Perfect mirror

namespace CUDA
{
    struct Vertex
    {
        glm::vec3 color;
        glm::vec2 uv;
        glm::vec3 normal;
        glm::vec3 tangent;
        glm::vec3 bitangent;
    };

    struct Material
    {
        glm::vec3 emission;
        float specularity;
        uint32_t albedoIdx{ INVALID_CUDA_TEX_INDEX };
        uint32_t normalMapIdx{ INVALID_CUDA_TEX_INDEX };
    };

    struct TriangleInfo
    {
        Vertex vertices[3];
        Material material;

        /**
        * Expecting barycentric coordinates.
        */
        __device__ glm::vec3 computeColor(const glm::vec2& triangleUV) const
        {
            return glm::normalize(barycentricLerp(vertices[0].color, vertices[1].color, vertices[2].color, triangleUV));
        }

        /**
         * Expecting barycentric coordinates.
         */
        __device__ glm::vec3 computeNormal(const glm::vec2& triangleUV) const
        {
            return glm::normalize(barycentricLerp(vertices[0].normal, vertices[1].normal, vertices[2].normal, triangleUV));
        }

        /**
        * Expecting barycentric coordinates.
        */
        __device__ glm::vec3 computeTangent(const glm::vec2& triangleUV) const
        {
            return glm::normalize(barycentricLerp(vertices[0].tangent, vertices[1].tangent, vertices[2].tangent, triangleUV));
        }

        /**
        * Expecting barycentric coordinates.
        */
        __device__ glm::vec3 computeBitangent(const glm::vec2& triangleUV) const
        {
            return glm::normalize(barycentricLerp(vertices[0].bitangent, vertices[1].bitangent, vertices[2].bitangent, triangleUV));
        }

        /**
        * Expecting barycentric coordinates.
        */
        __device__ glm::vec2 computeUV(const glm::vec2& triangleUV) const
        {
            return barycentricLerp(vertices[0].uv, vertices[1].uv, vertices[2].uv, triangleUV);
        }
    };
}
