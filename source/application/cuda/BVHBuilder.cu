#include "BVHBuilder.h"
#include <thrust/sort.h>
#include "util.h"
#include "glm/glm.hpp"
#include <glm/ext.hpp>
#include "engine/util/QueryManager.h"
#include <engine/util/math.h>

#pragma region Morton
const uint32_t QUANTIZATION_CLUSTER_SIZE = 1 << MortonCode3D<MortonIntType>::order;

template<typename IntType>
struct morton_encoder
{
    HOST_AND_DEVICE morton_encoder(const BBox& enclosingBBox) :
        m_min(enclosingBBox.min()),
        m_invScale(1.0f / enclosingBBox.scale()) {}

    HOST_AND_DEVICE MortonCode3D<IntType> operator() (const CUDA::Triangle& triangle)
    {
        glm::vec3 centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;

        // Map point to [0,1]^3 in the enclosing bbox, quantize and compute morton encoding
        return MortonCode3D<IntType>(
            CUDA::quantize((centroid[0] - m_min[0]) * m_invScale[0], QUANTIZATION_CLUSTER_SIZE),
            CUDA::quantize((centroid[1] - m_min[1]) * m_invScale[1], QUANTIZATION_CLUSTER_SIZE),
            CUDA::quantize((centroid[2] - m_min[2]) * m_invScale[2], QUANTIZATION_CLUSTER_SIZE));
    }

    const glm::vec3 m_min;
    const glm::vec3 m_invScale;
};
#pragma endregion 

#pragma region BBox computation
/**
* Code based on: "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" by Tero Karras
*/
struct compute_internal_node_bbox_kernel
{
    HOST_AND_DEVICE compute_internal_node_bbox_kernel(int* internalNodeAtomicCounters, NodeIndex** parents,
        BBox* internalNodeBBoxes, Node* children, CUDA::Triangle* triangles)
        :internalNodeAtomicCounters(internalNodeAtomicCounters), parents(parents),
        internalNodeBBoxes(internalNodeBBoxes), children(children), triangles(triangles) {}

    /*
    * From the mentioned paper:
    * "Each thread starts from one leaf node and walks up the tree using 
    *  parent pointers that we record during radix tree construction.
    *  We track how many threads have visited each internal node
    *  using atomic counters, the first thread terminates immediately 
    *  while the second one gets to process the node."
    */
    __device__ void operator()(NodeIndex leafIndex)
    {
        // Go up the parent chain
        NodeIndex curNodeIdx = parents[LEAF_PARENTS_IDX][leafIndex];

        while (curNodeIdx != INVALID_NODE_INDEX)
        {
            // Only the second node getting here gets to compute the bbox
            int* counterAddr = &internalNodeAtomicCounters[curNodeIdx];
            if (atomicAdd(counterAddr, 1) == 0)
                return;

            // Compute bbox
            BBox leftBBox;
            BBox rightBBox;

            NodeIndex leftChildIdx = children[curNodeIdx].leftChildIndex();
            NodeIndex rightChildIdx = children[curNodeIdx].rightChildIndex();

            if (children[curNodeIdx].leftChildType() == 0) // If child is a leaf then compute the bbox from triangles
                leftBBox.unite({ triangles[leftChildIdx].v0, triangles[leftChildIdx].v1, triangles[leftChildIdx].v2 });
            else // If child is an internal node then the bbox was already computed in the chain
                leftBBox = internalNodeBBoxes[leftChildIdx];

            // Same for right child
            if (children[curNodeIdx].rightChildType() == 0) // If child is a leaf then compute the bbox from triangles
                rightBBox.unite({ triangles[rightChildIdx].v0, triangles[rightChildIdx].v1, triangles[rightChildIdx].v2 });
            else // If child is an internal node then the bbox was already computed in the chain
                rightBBox = internalNodeBBoxes[rightChildIdx];

            // Now compute the final BBox and set it
            leftBBox.unite(rightBBox);
            internalNodeBBoxes[curNodeIdx] = leftBBox;

            // Proceed going up the chain
            curNodeIdx = parents[INTERNAL_PARENTS_IDX][curNodeIdx];
        }
    }

    int* internalNodeAtomicCounters;
    NodeIndex** parents;
    BBox* internalNodeBBoxes;
    Node* children;
    CUDA::Triangle* triangles;
};
#pragma endregion


#pragma region Radix Tree Construction
/*
* Use this functor for each internal node i to compute the two children.
* Code based on: "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" by Tero Karras
*/
template<typename IntType>
struct radix_tree_construction_kernel
{
    __device__ __host__ radix_tree_construction_kernel(MortonCode3D<IntType>* mortonCodes, uint32_t mortonCodeCount, NodeIndex** parents)
        :mortonCodes(mortonCodes), mortonCodeCount(mortonCodeCount), parents(parents) {}

    /*
    * Computes the length of the longest common prefix between keys ki, kj
    */
    __device__ int delta(int i, int j)
    {
        if (j >= mortonCodeCount || j < 0)
            return -1;

        MortonCode3D<IntType> ki = mortonCodes[i];
        MortonCode3D<IntType> kj = mortonCodes[j];

        // To handle duplicate morton codes an augmentation of keys is necessary.
        // This is achieved by concatenating ki with i and kj with j, respectively.
        // Note: CUDA::countLeadingZeros(ki ^ kj) = MortonCode3D<IntType>::totalBitCount in this case.
        if (ki == kj)
        {
            return CUDA::countLeadingZeros(i ^ j) + MortonCode3D<IntType>::totalBitCount;
        }

        return CUDA::countLeadingZeros(ki ^ kj);
    }

    __device__ Node operator()(const NodeIndex nodeIdx)
    {
        int i = static_cast<int>(nodeIdx);
        // Determine direction of the range (+1 or -1)
        int d = glm::sign(delta(i, i + 1) - delta(i, i - 1));

        // Compute upper bound for the length of the range
        int dMin = delta(i, i - d);
        int lMax = 2;
        while (delta(i, i + lMax * d) > dMin)
            lMax *= 2;

        // Find the other end using binary search
        int l = 0;
        while (lMax > 1)
        {
            lMax /= 2;
            if (delta(i, i + (l + lMax) * d) > dMin)
                l += lMax;
        }
        const int j = i + l * d;

        // Find the split position using binary search
        int dNode = delta(i, j);
        int s = 0;
        while (l > 1)
        {
            l = (l + 1) / 2;
            if (delta(i, i + (s + l) * d) > dNode)
                s += l;
        }
        s = i + s * d + min(d, 0);

        // Set children
        NodeIndex typeCode = 0;

        if (min(i, j) != s)
        {
            // Internal Node
            typeCode |= 1;
            parents[INTERNAL_PARENTS_IDX][s] = i;
        }
        else
        {
            // Leaf Node
            parents[LEAF_PARENTS_IDX][s] = i;
        }

        if (max(i, j) != s + 1)
        {
            // Internal Node
            typeCode |= 2;
            parents[INTERNAL_PARENTS_IDX][s + 1] = i;
        }
        else
        {
            // Leaf Node
            parents[LEAF_PARENTS_IDX][s + 1] = i;
        }

        return Node(s, typeCode);
    }

    MortonCode3D<IntType>* mortonCodes;
    uint32_t mortonCodeCount;
    NodeIndex** parents;
};
#pragma endregion 


#pragma region BVH Builder

void CUDA::BVHBuilder::build(CUDA::DeviceSceneDescription& outSceneDesc, size_t triangleCount)
{
    // 1. Assign a morton code for each primitive according to its centroid.
    // 2. Sort the morton codes.
    // 3. Construct a binary radix tree.
    // 4. Assign a bounding box for each internal node.

    QueryManager::beginElapsedTime(QueryTarget::CPU, "BVH Construction");

    uint32_t pointCount = static_cast<uint32_t>(triangleCount);
    uint32_t internalNodeCount = pointCount - 1;

    thrust::counting_iterator<NodeIndex> countingIter(0);
    m_mortonCodes.resize(pointCount);
    outSceneDesc.children.resize(internalNodeCount);
    outSceneDesc.parents[0].resize(pointCount);
    outSceneDesc.parents[1].resize(internalNodeCount);
    outSceneDesc.parents[1][0] = INVALID_NODE_INDEX;
    outSceneDesc.internalNodeBBoxes.resize(internalNodeCount);
    thrust::device_vector<glm::vec3> bboxCenterPoints(pointCount);

    // 1. Assign a morton code for each primitive according to its centroid.
    thrust::transform(outSceneDesc.triangles.begin(), outSceneDesc.triangles.end(),
        m_mortonCodes.begin(), morton_encoder<MortonIntType>(outSceneDesc.enclosingBBox));

    // 2. Sort the morton codes.
    // Note: sort_by_key sorts keys and values
    thrust::sort_by_key(m_mortonCodes.begin(), m_mortonCodes.end(), outSceneDesc.triangles.begin());

    // 3. Construct a binary radix tree
    MortonCode3D<MortonIntType>* mortonCodesPtr = thrust::raw_pointer_cast(m_mortonCodes.data());
    void* parentsPtr;
    cudaMalloc(&parentsPtr, sizeof(NodeIndex*) * 2);
    NodeIndex* hostParentsPtr[] = { thrust::raw_pointer_cast(outSceneDesc.parents[0].data()),
        thrust::raw_pointer_cast(outSceneDesc.parents[1].data()) };
    cudaMemcpy(parentsPtr, hostParentsPtr, sizeof(NodeIndex*) * 2, cudaMemcpyHostToDevice);

    thrust::transform(countingIter, countingIter + internalNodeCount, outSceneDesc.children.begin(),
        radix_tree_construction_kernel<MortonIntType>(mortonCodesPtr, pointCount, (NodeIndex**)parentsPtr));

    // 4. Assign a bounding box for each internal node.
    m_internalNodeAtomicCounters.resize(internalNodeCount);
    thrust::fill(m_internalNodeAtomicCounters.begin(), m_internalNodeAtomicCounters.end(), 0);
    thrust::for_each(countingIter, countingIter + pointCount,
        compute_internal_node_bbox_kernel(
            thrust::raw_pointer_cast(m_internalNodeAtomicCounters.data()),
            (NodeIndex**)parentsPtr,
            thrust::raw_pointer_cast(outSceneDesc.internalNodeBBoxes.data()),
            thrust::raw_pointer_cast(outSceneDesc.children.data()), thrust::raw_pointer_cast(outSceneDesc.triangles.data())
        ));

    // Verify result:
    // The computed AABB should have the same volume as the user provided enclosing AABB
    BBox rootBBox;
    cudaMemcpy(&rootBBox, thrust::raw_pointer_cast(outSceneDesc.internalNodeBBoxes.data()), sizeof(BBox), cudaMemcpyDeviceToHost);
    
    std::cout << "Original volume: " << outSceneDesc.enclosingBBox.volume();
    std::cout << " - Computed volume: " << rootBBox.volume() << std::endl;
    assert(math::nearEq(outSceneDesc.enclosingBBox.volume(), rootBBox.volume()));
    cudaFree(parentsPtr);

    QueryManager::endElapsedTime(QueryTarget::CPU, "BVH Construction");
}

#pragma endregion
