#pragma once
#include "engine/geometry/BBox.h"
#include "TriangleInfo.h"
#include <thrust/device_vector.h>
#include "geometry.h"
#include "Node.h"

#define LEAF_PARENTS_IDX 0
#define INTERNAL_PARENTS_IDX 1

namespace CUDA
{
    struct DeviceSceneDescription
    {
        thrust::device_vector<Node> children;
        thrust::device_vector<NodeIndex> parents[2];
        thrust::device_vector<Triangle> triangles;
        thrust::device_vector<BBox> internalNodeBBoxes;
        thrust::device_vector<TriangleInfo> triangleInfo;
        BBox enclosingBBox;
    };
}
