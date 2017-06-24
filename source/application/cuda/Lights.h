#pragma once
#include <glm/glm.hpp>

namespace CUDA
{
    struct DirectionalLight
    {
        glm::vec3 color;
        glm::vec3 direction;
        float intensity;
    };
}
