#pragma once
#include "engine/cuda/cuda_defs.h"
#include <cstdint>
#include "DeviceSceneDescription.h"
#include <engine/util/morton/morton.h>

// Supported types are uint32_t and uint64_t
using MortonIntType = uint32_t;

namespace CUDA
{
    class BVHBuilder
    {
    public:
        void build(CUDA::DeviceSceneDescription& outSceneDesc, size_t triangleCount);
    private:
        thrust::device_vector<MortonCode3D<MortonIntType>> m_mortonCodes;
        thrust::device_vector<int> m_internalNodeAtomicCounters;
    };
}
