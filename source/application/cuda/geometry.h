#pragma once
#include <cstdint>
#include <glm/detail/type_vec3.hpp>
#include "engine/cuda/cuda_defs.h"
#include "engine/geometry/BBox.h"

using TriangleIndex = uint32_t;

namespace CUDA
{
    struct Ray
    {
        glm::vec3 origin;
        glm::vec3 direction;

        HOST_AND_DEVICE Ray() {}
        HOST_AND_DEVICE Ray(const glm::vec3& origin, const glm::vec3& dir) : origin(origin), direction(dir) {}
    };

    struct Triangle
    {
        HOST_AND_DEVICE Triangle() {}
        HOST_AND_DEVICE Triangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, TriangleIndex idx)
            :v0(v0), v1(v1), v2(v2), infoIdx(idx) {}

        HOST_AND_DEVICE BBox bbox() const { return BBox::from({v0, v1, v2}); }
        HOST_AND_DEVICE glm::vec3 computeNormal() const { return glm::normalize(glm::cross(v1 - v0, v2 - v0)); }
        glm::vec3 v0, v1, v2;
        TriangleIndex infoIdx{ 0 };
    };
}
