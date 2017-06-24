#pragma once
#include "Lights.h"
#include "trace.h"

namespace CUDA
{
    __device__ inline glm::vec3 computeL(const glm::vec3& point, const DirectionalLight& dirLight);
}

namespace CUDA
{
    __device__ inline glm::vec3 computeL(const glm::vec3& point, const DirectionalLight& dirLight)
    {
        float distance = CUDA::shadowRay(CUDA::Ray(point - dirLight.direction * CUDA_EPSILON3, -dirLight.direction));
        bool inShadow = distance < FLT_MAX;
        if (inShadow)
            return glm::vec3(0.0f);

        return dirLight.color * dirLight.intensity;
    }
}
