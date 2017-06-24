#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "geometry.h"
#include "engine/geometry/BBox.h"
#include "Node.h"
#include "Stack.h"
#include "intersection.h"

namespace CUDA
{
    __device__ inline glm::vec3 computeBVHColor(
        const glm::vec3& baseColor,
        const CUDA::Ray& primaryRay,
        const BBox* internalNodeBBoxes,
        const Node* children);
}

__device__ glm::vec3 CUDA::computeBVHColor(
    const glm::vec3& baseColor,
    const CUDA::Ray& primaryRay,
    const BBox* internalNodeBBoxes,
    const Node* children)
{
    glm::vec3 accumulatedColor;
    Stack<NodeIndex, MAX_TRAVERSAL_STACK_SIZE> stack;
    stack.push(0);
    while (!stack.isEmpty())
    {
        // Find the closest hit
        NodeIndex curNode = stack.pop();

        Node cNode = children[curNode];
        if (cNode.isLeftChildInternal())
        {
            if (CUDA::rayAABBIntersection(primaryRay.origin, primaryRay.direction, internalNodeBBoxes[cNode.leftChildIndex()]))
            {
                stack.push(cNode.leftChildIndex());
                accumulatedColor += baseColor;
            }
        }

        if (stack.isFull())
        {
            printf("Full Stack!\n");
            return accumulatedColor;
        }

        if (cNode.isRightChildInternal())
        {
            if (CUDA::rayAABBIntersection(primaryRay.origin, primaryRay.direction, internalNodeBBoxes[cNode.rightChildIndex()]))
            {
                stack.push(cNode.rightChildIndex());
                accumulatedColor += baseColor;
            }
        }

        if (stack.isFull())
        {
            printf("Full Stack!\n");
            return accumulatedColor;
        }
    }

    return accumulatedColor;
}
