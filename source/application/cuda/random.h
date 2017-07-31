#pragma once
#include <glm/detail/type_vec3.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/detail/type_vec4.hpp>
#include <glm/detail/type_vec2.hpp>

namespace rng
{
    /**
     * In [0,1].
     */
    __device__ inline float getFloat(curandState* state)
    {
        return curand_uniform(state);
    }

    /**
     * In [-1,1]^2
     */
    __device__ inline glm::vec2 getVec2(curandState* state)
    {
        return glm::vec2(curand_uniform(state), curand_uniform(state)) * 2.0f - 1.0f;
    }

    /**
    * In [-1,1]^3
    */
    __device__ inline glm::vec3 getVec3(curandState* state)
    {
        return glm::vec3(curand_uniform(state), curand_uniform(state), curand_uniform(state)) * 2.0f - 1.0f;
    }

    /**
    * In [-1,1]^4
    */
    __device__ inline glm::vec4 getVec4(curandState* state)
    {
        return glm::vec4(curand_uniform(state), curand_uniform(state), curand_uniform(state), curand_uniform(state)) * 2.0f - 1.0f;
    }

    /**
    * In [-1,1]^2
    */
    __device__ inline glm::vec2 getDirection2D(curandState* state)
    {
        float theta = CUDA_2PI * getFloat(state);
        return glm::vec2(cosf(theta), sinf(theta));
    }

    /**
    * In [-1,1]^3
    */
    __device__ inline glm::vec3 getDirection3D(curandState* state)
    {
        float theta = CUDA_2PI * getFloat(state);
        float z = getFloat(state) * 2.0f - 1.0f;
        float sq = sqrtf(1.0f - z * z);
        return glm::vec3(cosf(theta) * sq, sinf(theta) * sq, z);
    }

    __device__ inline glm::vec3 getDirection3DInPlane(curandState* state, const glm::vec3& normal)
    {
        glm::vec3 r = getDirection3D(state);
        glm::vec3 p = glm::dot(normal, r) * normal;
        return glm::normalize(r - p);
    }

    namespace stratified
    {
        /**
        * Returns a random point (in [0,1)^2) in the strata given by strataPos in [0, dim.x - 1], [0, dim.y - 1].
        */
        __device__ inline glm::vec2 getVec2(curandState* state, const glm::ivec2& dim, const glm::ivec2& strataPos)
        {
            return glm::vec2(
                (strataPos.x + rng::getFloat(state)) / dim.x,
                (strataPos.y + rng::getFloat(state)) / dim.y);
        }

        /**
        * Returns a random float (in [0,1)) in the strata given by strataPos in [0, dim - 1].
        */
        __device__ inline float getFloat(curandState* state, int dim, int strataPos)
        {
            return (strataPos + rng::getFloat(state)) / dim;
        }
    }

    /*
     * This distribution is not uniform but cosine weighted: (pdf(x) = cos(theta)/pi)
     * Lower probability for rays in the lower part of the hemisphere.
     * Explained in: Physically Based Rendering 2nd edition Page 668
     */
    namespace cosDistribution
    {
        __device__ inline glm::vec3 getHemisphereDirection3D(curandState* state)
        {
            glm::vec2 rand(getFloat(state), getFloat(state));
            float r = sqrtf(rand.x);
            float theta = CUDA_2PI * rand.y;
            return glm::vec3(r * cosf(theta), r * sin(theta), sqrtf(1.0f - rand.x));
        }

        namespace stratified
        {
            __device__ inline glm::vec3 getHemisphereDirection3D(curandState* state, const glm::ivec2& dim, const glm::ivec2& strataPos)
            {
                glm::vec2 rand = rng::stratified::getVec2(state, dim, strataPos);
                float r = sqrtf(rand.x);
                float theta = CUDA_2PI * rand.y;
                return glm::vec3(r * cosf(theta), r * sin(theta), sqrtf(1.0f - rand.x));
            }
        }
    }
}
