#pragma once
#include <cstdio>
#include "engine/cuda/cuda_defs.h"
#include <glm/detail/type_vec2.hpp>
#include <glm/detail/type_vec3.hpp>
#include <glm/detail/type_vec4.hpp>

HOST_AND_DEVICE inline void print(const glm::vec2& v)
{
    printf("(%5.2f, %5.2f)\n", v.x, v.y);
}

HOST_AND_DEVICE inline void print(const glm::vec3& v)
{
    printf("(%5.2f, %5.2f, %5.2f)\n", v.x, v.y, v.z);
}

HOST_AND_DEVICE inline void print(const glm::vec4& v)
{
    printf("(%5.2f, %5.2f, %5.2f, %5.2f)\n", v.x, v.y, v.z, v.w);
}
