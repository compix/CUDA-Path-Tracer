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

    __device__ inline bool rayTriangleIntersection(const CUDA::Ray& ray, NodeIndex triangleIdx, glm::vec2& outUV, float& outT);

    __device__ inline bool rayAABBIntersection(const CUDA::Ray& ray, NodeIndex nodeIdx, float& outT);

    __device__ inline bool intersectsNode(const CUDA::Ray& ray, const Node& node, float& outHit, glm::vec2& outUV);

    /**
    * Note: outHit will be FLT_MAX if there is no intersection.
    */
    __device__ inline void computeNodeIntersection(const CUDA::Ray& ray, const Node& node, float& outHit, glm::vec2& outUV);

    /**
    * Returns the distance from the origin of the given ray to the hit.
    * If there is no hit FLT_MAX will be returned.
    */
    __device__ inline float shadowRay(const CUDA::Ray& ray);
}

__device__ float CUDA::shadowRay(const CUDA::Ray& ray)
{
    TriangleIndex idx; // Not used: Compiler will remove this during optimization
    glm::vec2 uv; // Same as above
    return trace(ray, idx, uv);
}

__device__ bool CUDA::intersectsNode(const CUDA::Ray& ray, const Node& node, float& outHit, glm::vec2& outUV)
{
    if (node.isInternal())
        return rayAABBIntersection(ray, node.index(), outHit);

    return rayTriangleIntersection(ray, node.index(), outUV, outHit);
}

__device__ void CUDA::computeNodeIntersection(const CUDA::Ray& ray, const Node& node, float& outHit, glm::vec2& outUV)
{
    if (node.isInternal())
    {
        if (rayAABBIntersection(ray, node.index(), outHit))
            return;
    }
    else if (rayTriangleIntersection(ray, node.index(), outUV, outHit))
        return;

    outHit = FLT_MAX;
}

__device__ float CUDA::trace(const CUDA::Ray & ray, TriangleIndex & outTriangleIdx, glm::vec2 & outUV)
{
    // Algorithm description:
    // Traverse the BVH along the ray starting at the root by
    // maintaining a stack of internal nodes that need to be processed. 
    // Also track the closest intersecting triangle (called minTriangle).

    // Processing an internal node:
    // Check for intersections with children.
    // Both are behind minTriangle: Done processing this node - Proceed processing the stack
    // Otherwise at least one is closer:
    // If the closer node is a leaf (triangle): 
    //     Update minTriangle and proceed processing the stack
    // otherwise it's an internal node: Process it next
    // If the other node is also in front of the triangle:
    //     If it's a triangle: Update minTriangle otherwise push the internal node on the stack

    // Using a constant stack size to store it in thread local memory.
    // Choosing a big enough stack size works quite well in practice.
    Stack<Node, MAX_TRAVERSAL_STACK_SIZE> stack;
    // Starting at the root which is always an internal node
    stack.push(Node::makeInternal(0));
    outTriangleIdx = INVALID_NODE_INDEX;
    // Record the closest triangle hit
    float minTriangleHit = FLT_MAX;

    while (!stack.isEmpty())
    {
        Node cNode = stack.pop();
        bool processingInternalnode = true;

        while (processingInternalnode)
        {
            cNode = g_geometryData.children[cNode.index()];

            float hits[2];
            glm::vec2 uv[2];
            Node children[] = { cNode.leftNode(), cNode.rightNode() };

            // Note: tFirst, tSecond will be FLT_MAX if there is no hit.
            // Relying on this fact allows to save some extra if (hit) statements
            // and improves code readability.
            computeNodeIntersection(ray, children[0], hits[0], uv[0]);
            computeNodeIntersection(ray, children[1], hits[1], uv[1]);

            // If both children are farther away than minTriangleHit then continue with the stack
            if (hits[0] >= minTriangleHit && hits[1] >= minTriangleHit)
                break;

            uint32_t idxClose{ 0 }, idxFar{ 1 };

            // One is closer: Check which one
            if (hits[1] < hits[0])
                CUDA::swap(idxClose, idxFar);

            if (children[idxClose].isInternal())
            {
                // Process this internal node next
                cNode = children[idxClose];

                // Check if the other node was also closer than minTriangleHit
                if (hits[idxFar] < minTriangleHit)
                {
                    if (children[idxFar].isInternal())
                    {
                        stack.push(children[idxFar]);
                    }
                    else
                    {
                        // Found closer triangle hit: update
                        minTriangleHit = hits[idxFar];
                        outUV = uv[idxFar];
                        outTriangleIdx = children[idxFar].index();
                    }
                }
            }
            else
            {
                // Closer hit is a triangle -> Update min hit
                minTriangleHit = hits[idxClose];
                outUV = uv[idxClose];
                outTriangleIdx = children[idxClose].index();
                processingInternalnode = false;
            }

#ifdef CHECK_FOR_FULL_STACK
            if (stack.isFull())
            {
                printf("tracen: Stack is full.");
                return minTriangleHit;
            }
#endif
        }
    }

    return minTriangleHit;
}

__device__ bool CUDA::rayTriangleIntersection(const CUDA::Ray& ray, NodeIndex triangleIdx, glm::vec2& outUV, float& outT)
{
    return CUDA::rayTriangleIntersection(ray.origin, ray.direction,
        g_geometryData.triangles[triangleIdx].v0, g_geometryData.triangles[triangleIdx].v1, g_geometryData.triangles[triangleIdx].v2, outUV, outT);
}

__device__ bool CUDA::rayAABBIntersection(const CUDA::Ray& ray, NodeIndex nodeIdx, float& outT)
{
    return CUDA::rayAABBIntersection(ray.origin, ray.direction, g_geometryData.internalNodeBBoxes[nodeIdx], outT);
}
