#pragma once
#include "engine/cuda/cuda_defs.h"
#include "TriangleInfo.h"
#include <thrust/detail/raw_pointer_cast.h>
#include "geometry.h"
#include "engine/geometry/BBox.h"
#include "Node.h"
#include <memory>
#include "BVHBuilder.h"
#include "DeviceSceneDescription.h"
#include "GeometryData.h"
#include "ShadingData.h"
#include "Lights.h"

namespace CUDA
{
    struct HostSceneDescription;

    class Scene
    {
    public:
        Scene();
        ~Scene();

        void upload(const CUDA::HostSceneDescription& sceneDesc);
        HOST_AND_DEVICE const TriangleInfo* getTriangleInfo() { return thrust::raw_pointer_cast(m_desc.triangleInfo.data()); }
        HOST_AND_DEVICE const CUDA::Triangle* getTriangles() { return thrust::raw_pointer_cast(m_desc.triangles.data()); }
        HOST_AND_DEVICE const BBox* getInternalNodeBBoxes() { return thrust::raw_pointer_cast(m_desc.internalNodeBBoxes.data()); }
        HOST_AND_DEVICE const Node* getChildren() { return thrust::raw_pointer_cast(m_desc.children.data()); }

        static const Scene* activeScene() { return m_activeScene; }
        static void setActiveScene(Scene* scene) { m_activeScene = scene; }

        void upload(const thrust::host_vector<cudaMipmappedArray_t>& textureArrays);

        bool isReady() const { return m_ready; }
    private:
        void updateShadingData();

    private:
        std::shared_ptr<BVHBuilder> m_bvhBuilder{ nullptr };
        CUDA::DeviceSceneDescription m_desc;

        static Scene* m_activeScene;

        thrust::device_vector<CUDA::DirectionalLight> m_dirLights;
        thrust::host_vector<cudaTextureObject_t> m_hostTextures;
        thrust::device_vector<cudaTextureObject_t> m_textures;
        bool m_ready{ false };
    };
}
