#pragma once
#include "engine/geometry/BBox.h"
#include "TriangleInfo.h"
#include <thrust/host_vector.h>
#include "geometry.h"
#include "Lights.h"

namespace CUDA
{
    struct HostSceneDescription
    {
        thrust::host_vector<CUDA::Triangle> triangles;
        thrust::host_vector<CUDA::TriangleInfo> triangleInfo;
        BBox enclosingBBox;

        thrust::host_vector<CUDA::DirectionalLight> dirLights;
    };
}
