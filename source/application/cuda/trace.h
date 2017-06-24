#pragma once
#include <cuda_runtime.h>
#include "geometry.h"
#include "engine/geometry/BBox.h"
#include "Node.h"
#include "Stack.h"
#include "intersection.h"
#include "util.h"
#include "GeometryData.h"
#include "Globals.h"

//#define CHECK_FOR_FULL_STACK

struct TraversalInfo
{
    NodeIndex idx{0};
    float t{0.0f};

    __device__ TraversalInfo() {}
    __device__ TraversalInfo(NodeIndex idx, float t)
        :idx(idx), t(t) {}
};

namespace CUDA
{
    /**
    * Computes the first intersection with a triangle for a given ray.
    * If there is no intersection then outTriangleIdx will be INVALID_NODE_INDEX
    * Returns the distance from the origin of the given ray to the hit.
    * If there is no hit FLT_MAX will be returned.
    */
    __device__ inline float trace(
        const CUDA::Ray& ray,
        TriangleIndex& outTriangleIdx,
        glm::vec2& outUV);

    /**
    * Returns the distance from the origin of the given ray to the hit.
    * If there is no hit FLT_MAX will be returned.
    */
    __device__ inline float shadowRay(const CUDA::Ray& ray);
}

__device__ float CUDA::trace(
    const CUDA::Ray& ray,
    TriangleIndex& outTriangleIdx,
    glm::vec2& outUV)
{
    Stack<TraversalInfo, MAX_TRAVERSAL_STACK_SIZE> stack;
    stack.push(TraversalInfo(0, 0.0f));
    float t;
    glm::vec2 uv;
    float minTriangleHit = FLT_MAX;
    outTriangleIdx = INVALID_NODE_INDEX;

    while (!stack.isEmpty())
    {
        // Find the closest hit
        int minHitIdx = 0;
        for (int i = 1; i <= stack.top; ++i)
        {
            if (stack.elements[i].t < stack.elements[minHitIdx].t)
            {
                minHitIdx = i;
            }
        }

        // If the currently hit triangle is closer than the computed minHit -> Found closest triangle hit -> leave loop
        if (minTriangleHit < stack.elements[minHitIdx].t)
            break;

        // Put the closest hit on top of the stack and pop
        CUDA::swap(stack.elements[minHitIdx], stack.elements[stack.top]);
        Node cNode = g_geometryData.children[stack.pop().idx];

        if (cNode.isLeftChildInternal())
        {
            if (CUDA::rayAABBIntersection(ray.origin, ray.direction, g_geometryData.internalNodeBBoxes[cNode.leftChildIndex()], t))
            {
                stack.push(TraversalInfo(cNode.leftChildIndex(), t));
            }
        }
        else
        {
            NodeIndex i = cNode.leftChildIndex();
            if (CUDA::rayTriangleIntersection(ray.origin, ray.direction,
                g_geometryData.triangles[i].v0, g_geometryData.triangles[i].v1, g_geometryData.triangles[i].v2, uv, t) && t < minTriangleHit)
            {
                minTriangleHit = t;
                outTriangleIdx = i;
                outUV = uv;
            }
        }

#ifdef CHECK_FOR_FULL_STACK
        if (stack.isFull())
        {
            printf("Full Stack!\n");
            return minTriangleHit;
        }
#endif

        if (cNode.isRightChildInternal())
        {
            if (CUDA::rayAABBIntersection(ray.origin, ray.direction, g_geometryData.internalNodeBBoxes[cNode.rightChildIndex()], t))
            {
                stack.push(TraversalInfo(cNode.rightChildIndex(), t));
            }
        }
        else
        {
            NodeIndex i = cNode.rightChildIndex();
            if (CUDA::rayTriangleIntersection(ray.origin, ray.direction,
                g_geometryData.triangles[i].v0, g_geometryData.triangles[i].v1, g_geometryData.triangles[i].v2, uv, t) && t < minTriangleHit)
            {
                minTriangleHit = t;
                outTriangleIdx = i;
                outUV = uv;
            }
        }

#ifdef CHECK_FOR_FULL_STACK
        if (stack.isFull())
        {
            printf("Full Stack!\n");
            return minTriangleHit;
        }
#endif
    }

    return minTriangleHit;
}

__device__ float CUDA::shadowRay(const CUDA::Ray& ray)
{
    TriangleIndex idx; // Not used: Compiler will remove this during optimization
    glm::vec2 uv; // Same as above
    return trace(ray, idx, uv);
}
