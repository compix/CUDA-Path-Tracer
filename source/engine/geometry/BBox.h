#pragma once
#include <glm/glm.hpp>
#include "engine/cuda/cuda_defs.h"
#include <stdint.h>

class BBox
{
public:
	HOST_AND_DEVICE BBox();
	HOST_AND_DEVICE BBox(const glm::vec3& min, const glm::vec3& max);

    HOST_AND_DEVICE static BBox from(const std::initializer_list<glm::vec3>& points);
	HOST_AND_DEVICE void unite(const std::initializer_list<glm::vec3>& ps);
	HOST_AND_DEVICE void unite(const glm::vec3& p);
	HOST_AND_DEVICE void unite(const BBox& b);

    HOST_AND_DEVICE bool overlaps(const BBox& b) const;
    HOST_AND_DEVICE bool inside(const glm::vec3& p) const;

    HOST_AND_DEVICE void expand(float delta);
    HOST_AND_DEVICE float surfaceArea() const;
    HOST_AND_DEVICE float volume() const;

    // Returns the number of the longest axis
    // 0 for x, 1 for y and 2 for z
	HOST_AND_DEVICE uint8_t maxExtentIdx() const;

    // Returns the number of the shortest axis
    // 0 for x, 1 for y and 2 for z
	HOST_AND_DEVICE uint8_t minExtentIdx() const;

    HOST_AND_DEVICE float maxExtent() const;
    HOST_AND_DEVICE float minExtent() const;

	HOST_AND_DEVICE glm::vec3 center() const { return m_min * 0.5f + m_max * 0.5f; }

    // Returns width(x), height(y), depth(z)
	HOST_AND_DEVICE glm::vec3 scale() const { return m_max - m_min; }

    // Returns the transformed BBox.
	HOST_AND_DEVICE BBox toWorld(const glm::mat4& world) const;

	HOST_AND_DEVICE void transform(const glm::mat4& m);

	HOST_AND_DEVICE glm::mat4 world() const;

	HOST_AND_DEVICE const glm::vec3& min() const { return m_min; }

	HOST_AND_DEVICE const glm::vec3& max() const { return m_max; }

private:
    glm::vec3 m_min, m_max;
};
