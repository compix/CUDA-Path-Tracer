#pragma once
#include "engine/cuda/cuda_defs.h"
#include <cstdint>

using NodeIndex = uint32_t;
const NodeIndex INVALID_NODE_INDEX = std::numeric_limits<NodeIndex>().max();

struct Node
{
    /*
    * Note: This node representation is used for a compact radix tree, i.e. every internal node has two children.
    * The right child is given by left child idx + 1 thus one node can encode two nodes.
    *
    * The following low order bits are added for encoding:
    * AB - A refers to right, B to current/left node
    * Where
    * 00: this node is a leaf, right node is a leaf
    * 01: this is an internal node and the right sibling is a leaf node
    * 10: this node is a leaf, right node is internal
    * 11: this is an internal node and the right sibling is an internal node
    *
    * Since 2 bits are used to encode additional information,
    * the actual index is in [0,2^30-1] -> ~1 billion primitives are supported
    */

    /**
    * typeCode in [0, 3]: See above.
    */
    HOST_AND_DEVICE Node(NodeIndex i, NodeIndex typeCode)
        :idx((i << 2) | typeCode) {}

    HOST_AND_DEVICE Node() {}

    HOST_AND_DEVICE static Node makeInternal(NodeIndex i) { return Node(i, 1); }

    HOST_AND_DEVICE void toLeafRightLeaf(NodeIndex i)
    {
        idx = (i << 2);
    }

    HOST_AND_DEVICE void toLeafRightInternal(NodeIndex i)
    {
        idx = (i << 2) | 2;
    }

    HOST_AND_DEVICE void toInternalRightInternal(NodeIndex i)
    {
        idx = (i << 2) | 3;
    }

    HOST_AND_DEVICE void toInternalRightLeaf(NodeIndex i)
    {
        idx = (i << 2) | 1;
    }

    /**
    * typeCode in [0, 3]: See above.
    */
    HOST_AND_DEVICE void encode(NodeIndex i, NodeIndex typeCode)
    {
        idx = (i << 2) | typeCode;
    }

    /**
    * Returns 0 if it is a leaf node otherwise 1.
    */
    HOST_AND_DEVICE uint32_t type() const
    {
        return (idx & 1);
    }

    /**
    * Returns 0 if it is a leaf node otherwise 1.
    */
    HOST_AND_DEVICE uint32_t leftChildType() const
    {
        return (idx & 1);
    }

    HOST_AND_DEVICE bool isLeftChildInternal() const
    {
        return (idx & 1) == 1;
    }

    /**
    * If this node encodes two nodes then the index of the left child is returned.
    */
    HOST_AND_DEVICE NodeIndex index() const
    {
        return idx >> 2;
    }

    HOST_AND_DEVICE NodeIndex leftChildIndex() const
    {
        return idx >> 2;
    }

    HOST_AND_DEVICE NodeIndex rightChildIndex() const
    {
        return (idx >> 2) + 1;
    }

    /**
    * Returns 0 if it is a leaf node otherwise 1.
    */
    HOST_AND_DEVICE NodeIndex rightChildType() const
    {
        return (idx & 2) >> 1;
    }

    HOST_AND_DEVICE bool isRightChildInternal() const
    {
        return (idx & 2) == 2;
    }

    HOST_AND_DEVICE bool isInternal() const
    {
        return (idx & 1) == 1;
    }

    HOST_AND_DEVICE NodeIndex typeCode() const
    {
        return idx & 3;
    }

    HOST_AND_DEVICE Node leftNode() const
    {
        return Node(leftChildIndex(), leftChildType());
    }

    HOST_AND_DEVICE Node rightNode() const
    {
        return Node(rightChildIndex(), rightChildType());
    }

    NodeIndex idx{ 0 };
};
