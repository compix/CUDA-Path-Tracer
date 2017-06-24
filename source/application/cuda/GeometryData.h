#pragma once
#include "Node.h"
#include "engine/geometry/BBox.h"
#include "geometry.h"

// Convenience structure to be passed to device functions
struct GeometryData
{
    const BBox* internalNodeBBoxes;
    const Node* children;
    const CUDA::Triangle* triangles;
    uint32_t triangleCount;

    GeometryData(const BBox* internalNodeBBoxes, const Node* children, const CUDA::Triangle* triangles, uint32_t triangleCount)
        :internalNodeBBoxes(internalNodeBBoxes), children(children), triangles(triangles), triangleCount(triangleCount) {}

    HOST_AND_DEVICE GeometryData() {}
};
