#include "Scene.h"
#include "HostSceneDescription.h"
#include "cuda_common.h"
#include "Globals.h"
#include "engine/util/Logger.h"
#include "application/CUDAPathTracer/GUI/PathTracingSettings.h"

__constant__ GeometryData g_geometryData;
__constant__ ShadingData g_shadingData;
__constant__ LightingData g_lightingData;
__constant__ MiscData g_miscData;

CUDA::Scene* CUDA::Scene::m_activeScene{nullptr};

CUDA::Scene::Scene()
{
    m_bvhBuilder = std::make_shared<BVHBuilder>();
}

CUDA::Scene::~Scene()
{
    for (auto tex : m_hostTextures)
    {
        cudaDestroyTextureObject(tex);
    }
}

void CUDA::Scene::upload(const CUDA::HostSceneDescription& sceneDesc)
{
    m_ready = true;
    m_desc.triangleInfo = sceneDesc.triangleInfo;
    m_desc.triangles = sceneDesc.triangles;
    m_desc.enclosingBBox = sceneDesc.enclosingBBox;

    Logger::stringStream(CUDA_STRING_STREAM) << "Uploaded triangle info." << std::endl;

    m_bvhBuilder->build(m_desc, sceneDesc.triangles.size());

    GeometryData geometryData(getInternalNodeBBoxes(), getChildren(), getTriangles(), sceneDesc.triangles.size());
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(g_geometryData, &geometryData, sizeof(GeometryData)));
    updateShadingData();

    LightingData lightingData;
    m_dirLights = sceneDesc.dirLights;
    lightingData.dirLightCount = sceneDesc.dirLights.size();
    lightingData.dirLights = thrust::raw_pointer_cast(m_dirLights.data());
    lightingData.indirectIntensity = PathTracerSettings::GI.indirectIntensity;
    lightingData.indirectBounceCount = static_cast<uint8_t>(PathTracerSettings::GI.indirectBounceCount);

    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(g_lightingData, &lightingData, sizeof(LightingData)));

    MiscData miscData;
    miscData.useStratifiedSampling = PathTracerSettings::GI.useStratifiedSampling;
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(g_miscData, &miscData, sizeof(MiscData)));
}

void CUDA::Scene::upload(const thrust::host_vector<cudaMipmappedArray_t>& textureArrays)
{
    for (size_t i = 0; i < textureArrays.size(); ++i)
    {
        cudaTextureObject_t tex;
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeMipmappedArray;
        resDesc.res.mipmap.mipmap = textureArrays[i];
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.normalizedCoords = 1;
        texDesc.maxMipmapLevelClamp = FLT_MAX;
        texDesc.mipmapFilterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;

        texDesc.filterMode = cudaFilterModeLinear;
        CUDA_ERROR_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

        m_hostTextures.push_back(tex);
    }

    m_textures = m_hostTextures;
    updateShadingData();
}

void CUDA::Scene::updateShadingData()
{
    ShadingData shadingData(getTriangleInfo());
    shadingData.textures = thrust::raw_pointer_cast(m_textures.data());
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(g_shadingData, &shadingData, sizeof(ShadingData)));
}
