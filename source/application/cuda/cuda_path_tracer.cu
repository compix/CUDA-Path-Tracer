#include "cuda_path_tracer.h"
#include <device_launch_parameters.h>
#include "cuda_surface_types.h"
#include "cuda_runtime.h"
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <cstdio>
#include "cuda_common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "engine/geometry/BBox.h"
#include "engine/util/morton/morton.h"
#include <thrust/sort.h>
#include <thrust/copy.h>
#include "BVHBuilder.h"
#include "intersection.h"
#include <curand.h>
#include <curand_kernel.h>
#include "random.h"
#include "Stack.h"
#include "visualization.h"
#include "trace.h"
#include "Globals.h"
#include "Lights.h"
#include "LightAlgorithms.h"
#include "math_util.h"
#include "logging.h"
#include "engine/util/Random.h"

#define CHECK_FOR_FULL_STACK
//#define CHECK_TRIANGLE_IDX_OUT_OF_BOUNDS

#pragma region Shading
__device__ glm::vec3 blinnPhongBRDF(
    glm::vec3 Id, glm::vec3 kd, glm::vec3 Is, glm::vec3 ks,
    glm::vec3 normal, glm::vec3 lightVec, glm::vec3 halfway, float shininess)
{
    float diffuseFactor = glm::max(glm::dot(normal, lightVec), 0.0f);
    glm::vec3 diffuse = Id * kd * diffuseFactor;

    float specFactor = pow(glm::max(glm::dot(normal, halfway), 0.0f), shininess);
    glm::vec3 specular = glm::vec3(0.0f);

    if (shininess > 0.0 && diffuseFactor > 0.0f)
        specular = Is * ks * specFactor;

    return diffuse + specular;
}
#pragma endregion

__device__ const CUDA::TriangleInfo& getTriangleInfo(TriangleIndex triangleIdx)
{
    return g_shadingData.triangleInfo[g_geometryData.triangles[triangleIdx].infoIdx];
}

__device__ glm::vec3 blinnPhongShadingTrace(
    const CUDA::Ray& primaryRay,
    const glm::vec3& lightVec,
    const glm::vec3& eyePos)
{
    glm::vec3 view = -primaryRay.direction;
    glm::vec3 halfway = 0.5f * (view + lightVec);
    NodeIndex triangleIdx = INVALID_NODE_INDEX;
    glm::vec2 minUV;

    CUDA::trace(primaryRay, triangleIdx, minUV);

    if (triangleIdx != INVALID_NODE_INDEX)
    {
        // Compute shading
        CUDA::TriangleInfo info = g_shadingData.triangleInfo[g_geometryData.triangles[triangleIdx].infoIdx];
        glm::vec3 color = CUDA::barycentricLerp(info.vertices[0].color, info.vertices[1].color, info.vertices[2].color, minUV);
        glm::vec3 normal = glm::normalize(
            glm::cross(g_geometryData.triangles[triangleIdx].v1 - g_geometryData.triangles[triangleIdx].v0,
                g_geometryData.triangles[triangleIdx].v2 - g_geometryData.triangles[triangleIdx].v0));
        return blinnPhongBRDF(glm::vec3(1.0f), color, glm::vec3(1.0f), glm::vec3(0.1f), normal, lightVec, halfway, 32.0f);
    }

    return glm::vec3(0.0f);
}

__device__ glm::vec3 unpackNormal(const CUDA::TriangleInfo& info, const glm::vec2& triangleUV, 
    const glm::vec2& texCoords, glm::vec3& outTangent, glm::vec3& outBitangent)
{
    glm::vec3 normal = info.computeNormal(triangleUV);

    if (info.material.normalMapIdx == INVALID_CUDA_TEX_INDEX)
        return normal;

    float4 normalSample = tex2D<float4>(g_shadingData.textures[info.material.normalMapIdx], texCoords.x, texCoords.y);
    normalSample.x = 2.0f * normalSample.x - 1.0f;
    normalSample.y = 2.0f * normalSample.y - 1.0f;
    normalSample.z = 2.0f * normalSample.z - 1.0f;

    outTangent = info.computeTangent(triangleUV);
    outBitangent = info.computeBitangent(triangleUV);
    
    glm::vec3 outNormal = normalSample.x * outTangent + normalSample.y * outBitangent + normalSample.z * normal;
    return glm::normalize(outNormal);
}

__device__ glm::vec3 unpackNormal(const CUDA::TriangleInfo& info, const glm::vec2& triangleUV,
    const glm::vec2& texCoords)
{
    glm::vec3 t, b;
    return unpackNormal(info, triangleUV, texCoords, t, b);
}

/**
 * Expecting triangleIdx to be valid.
 */
__device__ void extractInfo(
    TriangleIndex triangleIdx, 
    const glm::vec2& triangleUV,
    glm::vec3& outNormal,
    glm::vec3& outTangent,
    glm::vec3& outBitangent,
    glm::vec3& outAlbedo,
    glm::vec2& texCoords)
{
    auto& triangleInfo = getTriangleInfo(triangleIdx);
    texCoords = triangleInfo.computeUV(triangleUV);

    if (triangleInfo.material.albedoIdx != INVALID_NODE_INDEX)
    {
        float4 fColor = tex2D<float4>(g_shadingData.textures[triangleInfo.material.albedoIdx], texCoords.x, texCoords.y);
        outAlbedo = glm::vec3(fColor.x, fColor.y, fColor.z);
    }
    else
    {
        outAlbedo = triangleInfo.computeColor(triangleUV);
    }

    outNormal = unpackNormal(triangleInfo, triangleUV, texCoords, outTangent, outBitangent);
}

__device__ void extractInfo(
    TriangleIndex triangleIdx,
    const glm::vec2& triangleUV,
    glm::vec3& outNormal,
    glm::vec3& outAlbedo)
{
    glm::vec2 texCoords;
    glm::vec3 t, b;
    extractInfo(triangleIdx, triangleUV, outNormal, t, b, outAlbedo, texCoords);
}

__device__ void extractInfo(
    TriangleIndex triangleIdx,
    const glm::vec2& triangleUV,
    glm::vec3& outNormal,
    glm::vec3& outTangent,
    glm::vec3& outBitangent,
    glm::vec3& outAlbedo)
{
    glm::vec2 texCoords;
    extractInfo(triangleIdx, triangleUV, outNormal, outTangent, outBitangent, outAlbedo, texCoords);
}

__device__ glm::vec3 blinnPhongShadingShadowTrace(
    const CUDA::Ray& primaryRay,
    const glm::vec3& eyePos,
    curandState* randState,
    TriangleIndex& outTriangleIdx,
    glm::vec3& outHitPos,
    glm::vec3& outNormal,
    glm::vec3& outTangent,
    glm::vec3& outBitangent,
    glm::vec3& outAlbedo)
{
    glm::vec2 triangleUV;

    float t = CUDA::trace(primaryRay, outTriangleIdx, triangleUV);

    if (outTriangleIdx != INVALID_NODE_INDEX)
    {
#ifdef CHECK_TRIANGLE_IDX_OUT_OF_BOUNDS
        if (outTriangleIdx >= g_geometryData.triangleCount)
        {
            printf("Triangle index out of bounds!\n");
            return glm::vec3(1.0f);
        }
#endif

        // Compute shading
        auto& triangleInfo = getTriangleInfo(outTriangleIdx);
        outHitPos = primaryRay.direction * t + primaryRay.origin;
        extractInfo(outTriangleIdx, triangleUV, outNormal, outTangent, outBitangent, outAlbedo);

        // Check if it's an emissive material
        glm::vec3 emission = triangleInfo.material.emission;
        if ((emission.r + emission.g + emission.b) > CUDA_EPSILON)
        {
            outAlbedo = glm::vec3(1.0f);
            return emission;
        }

        glm::vec3 L(0.0f);

        glm::vec3 color(0.0f);

        for (int i = 0; i < g_lightingData.dirLightCount; ++i)
        {
            auto dirLight = g_lightingData.dirLights[i];

            L += CUDA::computeL(outHitPos, dirLight);
            glm::vec3 view = -primaryRay.direction;
            glm::vec3 halfway = 0.5f * (view - dirLight.direction);
            color += blinnPhongBRDF(L, outAlbedo, L, glm::vec3(0.5f), outNormal, 
                -dirLight.direction, halfway, triangleInfo.material.specularity);
        }

        return color;
    }

    return glm::vec3(0.0f);
}

__device__ glm::vec3 blinnPhongShadingShadowTrace(
    const CUDA::Ray& primaryRay,
    const glm::vec3& eyePos,
    curandState* randState,
    TriangleIndex& outTriangleIdx,
    glm::vec3& outHitPos,
    glm::vec3& outNormal,
    glm::vec3& outAlbedo)
{
    glm::vec3 t, b;
    return blinnPhongShadingShadowTrace(primaryRay, eyePos, randState, outTriangleIdx, outHitPos, outNormal, t, b, outAlbedo);
}

__device__ glm::vec3 blinnPhongShadingShadowTrace(
    const CUDA::Ray& primaryRay,
    const glm::vec3& eyePos,
    curandState* randState,
    TriangleIndex& outTriangleIdx)
{
    glm::vec3 hitPos, normal, outTangent, outBitangent, albedo;
    return blinnPhongShadingShadowTrace(primaryRay, eyePos, randState, outTriangleIdx, hitPos, normal, outTangent, outBitangent, albedo);
}

/**
 * Transform a given vector to the hemisphere given by normal.
 */
__device__ glm::vec3 toHemisphere(const glm::vec3& vector, const glm::vec3& normal)
{
    glm::vec3 x;
    if (glm::abs(normal.x) < 0.5f)
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
__device__ glm::vec3 toHemisphere(const glm::vec3& vector, const glm::vec3& normal, const glm::vec3& tangent, const glm::vec3& bitangent)
{
    return vector.x * tangent + vector.y * bitangent + vector.z * normal;
}

__device__ glm::vec3 blinnPhongShadingPathTrace(
    const CUDA::Ray& primaryRay,
    const glm::vec3& eyePos,
    curandState* randState)
{
    glm::vec2 triangleUV;
    glm::vec3 albedo;
    glm::vec3 normal;
    TriangleIndex triangleIdx;
    CUDA::Ray curRay = primaryRay;
    glm::vec3 hitPos;
    // Compute direct contribution
    glm::vec3 accumulatedColor = blinnPhongShadingShadowTrace(curRay, eyePos, randState, triangleIdx, hitPos, normal, albedo);
    glm::vec3 absorbed = albedo;

    if (triangleIdx == INVALID_NODE_INDEX)
    {
        return glm::vec3(0.0f);
    }

    // Compute diffuse indirect lighting
    for (uint32_t i = 0; i < g_lightingData.indirectBounceCount; ++i)
    {
        curRay.origin = hitPos + normal * CUDA_EPSILON5;
        // Using importance sampling with a cosine weighted distribution -> No need to multiply with cos(theta)
        curRay.direction = rng::cosDistribution::getHemisphereDirection3D(randState);
        curRay.direction = toHemisphere(curRay.direction, normal);

        float t = trace(curRay, triangleIdx, triangleUV);

        if (triangleIdx != INVALID_NODE_INDEX)
        {
#ifdef CHECK_TRIANGLE_IDX_OUT_OF_BOUNDS
            if (triangleIdx >= g_geometryData.triangleCount)
            {
                printf("Triangle index out of bounds!\n");
                return glm::vec3(1.0f);
            }
#endif

            hitPos = curRay.direction * t + curRay.origin;
            extractInfo(triangleIdx, triangleUV, normal, albedo);
            auto& triangleInfo = getTriangleInfo(triangleIdx);
            glm::vec3 L(0.0f);
            // Check if it's an emissive material
            glm::vec3 emission = triangleInfo.material.emission;
            if ((emission.r + emission.g + emission.b) > CUDA_EPSILON)
            {
                L = emission * glm::dot(-curRay.direction, normal);
                albedo = glm::vec3(1.0f);
            }
            else
            {
                // Send rays to light sources - in this case cos(theta) is necessary
                for (int i = 0; i < g_lightingData.dirLightCount; ++i)
                {
                    auto dirLight = g_lightingData.dirLights[i];
                    L += CUDA::computeL(hitPos, dirLight) * glm::dot(-dirLight.direction, normal);
                }
            }
            
            absorbed *= albedo;
            accumulatedColor += absorbed * L * g_lightingData.indirectIntensity;
        }
        else
        {
            break;
        }
    }

    return accumulatedColor;
}

#pragma region PathTracer

CUDAPathTracer::CUDAPathTracer()
{
    m_scene = std::make_shared<CUDA::Scene>();
    CUDA::Scene::setActiveScene(m_scene.get());
}

CUDAPathTracer::~CUDAPathTracer()
{
    if (m_screenTexture != 0)
    {
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(m_screenTexture));
    }
}

void CUDAPathTracer::setScreenTexture(const cudaArray_t* screenArray)
{
    if (m_screenTexture != 0)
    {
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(m_screenTexture));
    }

    cudaResourceDesc viewCudaArrayResourceDesc;
    memset(&viewCudaArrayResourceDesc, 0, sizeof(cudaResourceDesc));
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = *screenArray;

    CUDA_ERROR_CHECK(cudaCreateSurfaceObject(&m_screenTexture, &viewCudaArrayResourceDesc));
}

void CUDAPathTracer::setPrimaryRays(const CUDA::Ray& ray00, const CUDA::Ray& ray10, const CUDA::Ray& ray11, const CUDA::Ray& ray01)
{
    m_ray00 = ray00;
    m_ray10 = ray10;
    m_ray11 = ray11;
    m_ray01 = ray01;
}

// 0: bottom left, 1: bottom right, 2: top right, 3: top left
// t0: interpolation horizontally, t1 vertically
__device__ glm::vec3 lerpDirection(glm::vec3 d0, glm::vec3 d1, glm::vec3 d2, glm::vec3 d3, float t0, float t1)
{
    return glm::lerp(glm::lerp(d0, d1, t0), glm::lerp(d3, d2, t0), t1);
}

/*
* Global functions are also called "kernels". It's the functions that you may call from the host side using CUDA kernel call semantics (<<<...>>>).
* Device functions can only be called from other device or global functions. __device__ functions cannot be called from host code.
*/
__global__ void raytrace_kernel(cudaSurfaceObject_t surfObj, glm::vec3 eyePos,
    glm::vec3 ray00, glm::vec3 ray10, glm::vec3 ray11, glm::vec3 ray01, glm::ivec2 screenSize, uint32_t frameNumber, uint32_t randomSeed)
{
    uint32_t imgX = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t imgY = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // Some seeding tests that yield interesting convergance/image computation patterns.
    //uint64_t seed = uint64_t(frameNumber) + uint64_t((imgX << 16) | imgY);
    //uint64_t seed = (frameNumber << 16) | ((imgX << 10) | imgY);
    //uint64_t seed = frameNumber; // Seems to yield faster visually appealing (to me) results

    // Note: curand_init is slow if using subsequence and/or offset.
    // Using seeds with an obvious pattern like 0,1,2... yields correlated
    // pseudo random number sequences. Solution: Use a random seed provided as parameter.
    uint64_t seed = uint64_t(randomSeed) + threadId;
    curandState randState;
    curand_init(seed, 0, 0, &randState);

    glm::vec2 texCoords(imgX / (float)screenSize.x, imgY / (float)screenSize.y);
    CUDA::Ray primaryRay;
    primaryRay.direction = lerpDirection(ray00, ray10, ray11, ray01, texCoords.x, texCoords.y);
    primaryRay.origin = eyePos;

    glm::vec3 raytracedColor = blinnPhongShadingPathTrace(primaryRay, eyePos, &randState);

    glm::vec3 hitColor = raytracedColor;
    float4 readColor;
    surf2Dread(&readColor, surfObj, imgX * sizeof(float4), imgY, cudaBoundaryModeClamp);
    glm::vec3 finalColor(readColor.x, readColor.y, readColor.z);

    float t = float(frameNumber) / (frameNumber + 1.0f);
    finalColor = glm::lerp(hitColor, finalColor, t);

    float4 color = make_float4(finalColor.r, finalColor.g, finalColor.b, 1.0f);
    surf2Dwrite(color, surfObj, imgX * sizeof(float4), imgY, cudaBoundaryModeClamp);
}

void CUDAPathTracer::raytrace(int screenWidth, int screenHeight, uint32_t frameNumber)
{
    if (!m_scene->isReady())
        return;

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(screenWidth / threadsPerBlock.x, screenHeight / threadsPerBlock.y);

    // Currently using progressive path tracing: 
    // Computing an image every frame and adding its contribution to the previously
    // computed image. Camera jittering is used for anti-aliasing.
    // This means that caching primary ray results is currently not possible because they are different
    // every frame -> TODO: Implement full screen post process anti-aliasing and cache primary ray results.
    raytrace_kernel << <numBlocks, threadsPerBlock >> > (
        m_screenTexture,
        m_eyePos,
        m_ray00.direction, m_ray10.direction, m_ray11.direction, m_ray01.direction,
        glm::ivec2(screenWidth, screenHeight),
        m_localFrameNumber,
        Random::getUInt(0, std::numeric_limits<uint32_t>().max()));

    CUDA_ERROR_CHECK(cudaGetLastError());
    ++m_localFrameNumber;
}

void CUDAPathTracer::upload(const CUDA::HostSceneDescription& sceneDesc)
{
    m_scene->upload(sceneDesc);
    m_localFrameNumber = 0;
}

#pragma endregion 
