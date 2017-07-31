#pragma once
#include <glm/glm.hpp>
#include "geometry.h"
#include <engine/geometry/BBox.h>
#include <engine/util/math.h>

namespace CUDA
{
    // Returns a negative number if the ray missed 
    __device__ inline float intersectsSphere(CUDA::Ray ray, glm::vec3 sPos, float radius)
    {
        glm::vec3 toSphere = ray.origin - sPos;
        float a = glm::dot(ray.direction, ray.direction);
        float b = glm::dot(ray.direction, (2.0f * toSphere));
        float c = glm::dot(toSphere, toSphere) - radius * radius;

        float d = b * b - 4.0f * a * c;

        if (d > 0.0f)
            return (-b - sqrt(d)) / (2.0f * a);

        return -1.0f;
    }

    /*
    * Triangle intersection test from "Fast Minimum Storage RayTriangle Intersection" by M�ller et al. 2005
    */
    __device__ inline bool rayTriangleIntersection(const glm::vec3& origin, const glm::vec3& direction,
        const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, glm::vec2& uv, float& t)
    {
        glm::vec3 e1 = p1 - p0;
        glm::vec3 e2 = p2 - p0;

        glm::vec3 q = glm::cross(direction, e2);

        float det = glm::dot(e1, q);
        
        if (det > -FLT_EPSILON && det < FLT_EPSILON)
            return false;

        float inv_det = 1.f / det;

        glm::vec3 s = origin - p0;
        uv.x = inv_det * glm::dot(s, q);

        if (uv.x < 0.f)
            return false;

        glm::vec3 r = glm::cross(s, e1);

        uv.y = inv_det * glm::dot(direction, r);

        if (uv.y < 0.f || (uv.x + uv.y) > 1.f)
            return false;

        t = inv_det * glm::dot(e2, r);

        return t >= 0.0f;
    }

    /*
    * Triangle intersection test from "Fast Minimum Storage RayTriangle Intersection" by M�ller et al. 2005
    */
    __device__ inline bool rayTriangleIntersection(const glm::vec3& origin, const glm::vec3& direction,
        const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, glm::vec2& uv, float& t, glm::vec3& triangleNormal)
    {
        glm::vec3 e1 = p1 - p0;
        glm::vec3 e2 = p2 - p0;

        glm::vec3 q = glm::cross(direction, e2);

        float det = glm::dot(e1, q);

        if (det > -FLT_EPSILON && det < FLT_EPSILON)
            return false;

        float inv_det = 1.f / det;

        glm::vec3 s = origin - p0;
        uv.x = inv_det * glm::dot(s, q);

        if (uv.x < 0.f)
            return false;

        glm::vec3 r = glm::cross(s, e1);

        uv.y = inv_det * glm::dot(direction, r);

        if (uv.y < 0.f || (uv.x + uv.y) > 1.f)
            return false;

        t = inv_det * glm::dot(e2, r);
        triangleNormal = glm::normalize(glm::cross(e1, e2));

        return t >= 0.0f;
    }

    __device__ inline bool rayTriangleIntersection(const glm::vec3& origin, const glm::vec3& direction,
        const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, glm::vec3& n)
    {
        glm::vec2 uv;
        float t;
        return rayTriangleIntersection(origin, direction, p0, p1, p2, uv, t, n);
    }

    __device__ inline bool rayAABBIntersection(const glm::vec3& o, const glm::vec3& d, const BBox& aabb, float& t)
    {
        glm::vec3 dInv = 1.0f / d;
        glm::vec3 tMin = (aabb.min() - o) * dInv;
        glm::vec3 tMax = (aabb.max() - o) * dInv;
        glm::vec3 t1 = min(tMin, tMax);
        glm::vec3 t2 = max(tMin, tMax);
        t = glm::max(glm::max(t1.x, t1.y), t1.z); // near hit
        float farT = glm::min(glm::min(t2.x, t2.y), t2.z); // far hit

        return t <= farT && farT > 0.0f;
    }

    __device__ inline bool rayAABBIntersection(const glm::vec3& o, const glm::vec3& d, const BBox& aabb)
    {
        float t;
        return rayAABBIntersection(o, d, aabb, t);
    }

    // Note that the assumption, that precomputing dInv to compute multiple ray/bbox tests for the same ray
    // improves performance doesn't necessarily apply because inlining the computation leads to compiler optimizations
    // that do it automatically resulting in the same assembly output.
    __device__ inline bool rayAABBIntersectionInvDirection(const glm::vec3& o, const glm::vec3& dInv, const BBox& aabb, float& t)
    {
        glm::vec3 tMin = (aabb.min() - o) * dInv;
        glm::vec3 tMax = (aabb.max() - o) * dInv;
        glm::vec3 t1 = min(tMin, tMax);
        glm::vec3 t2 = max(tMin, tMax);
        t = glm::max(glm::max(t1.x, t1.y), t1.z); // near hit
        float farT = glm::min(glm::min(t2.x, t2.y), t2.z); // far hit

        return t <= farT && farT > 0.0f;
    }

    // Based on "An Efficient and Robust Ray–Box Intersection Algorithm" by Amy Williams et al.
    __device__ inline bool segmentAABBIntersectionWilliams(const glm::vec3& o, const glm::ivec3& sign, 
        const glm::vec3& dInv, float t0, float t1, const glm::vec3 bounds[2], float& tmin)
    {
        tmin = (bounds[sign[0]].x - o.x) * dInv.x;
        float tmax = (bounds[1 - sign[0]].x - o.x) * dInv.x;
        float tymin = (bounds[sign[1]].y - o.y) * dInv.y;
        float tymax = (bounds[1 - sign[1]].y - o.y) * dInv.y;

        if ((tmin > tymax) || (tymin > tmax))
            return false;

        if (tymin > tmin)
            tmin = tymin;

        if (tymax < tmax)
            tmax = tymax;

        float tzmin = (bounds[sign[2]].z - o.z) * dInv.z;
        float tzmax = (bounds[1 - sign[2]].z - o.z) * dInv.z;

        if ((tmin > tzmax) || (tzmin > tmax))
            return false;

        if (tzmin > tmin)
            tmin = tzmin;

        if (tzmax < tmax)
            tmax = tzmax;

        return ((tmin < t1) && (tmax > t0));
    }

    __device__ inline bool rayAABBIntersectionWilliams(const glm::vec3& o, const glm::ivec3& sign,
        const glm::vec3& dInv, const glm::vec3 bounds[2], float& tmin)
    {
        return segmentAABBIntersectionWilliams(o, sign, dInv, 0.0f, FLT_MAX, bounds, tmin);
    }
}
