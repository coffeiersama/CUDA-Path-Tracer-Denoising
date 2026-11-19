#pragma once
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <vector>

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

class BoundingBox {
public:
	glm::vec3 minCorner, maxCorner;

	BoundingBox();
	BoundingBox(const glm::vec3& min, const glm::vec3& max);
	BoundingBox(const BoundingBox &b);

	int CalculatethelongestAxis();
	
	bool isPointInsideAABB(const glm::vec3 &p) const;
	
	float Area() const;	// for sah
	glm::vec3 getOffset(const glm::vec3 &p) const;

	__host__ __device__ BoundingBox& operator = (const BoundingBox& b) {
		minCorner = b.minCorner;
		maxCorner = b.maxCorner;
		return *this;
	}

	const glm::vec3& operator[](int i) const {
		return (i == 0) ? minCorner : maxCorner;
	}

	BoundingBox operator || (const BoundingBox& b2) const {
		if (this->minCorner.x == 0.f && this->maxCorner.x == 0.f &&
			this->minCorner.y == 0.f && this->maxCorner.y == 0.f &&
			this->minCorner.z == 0.f && this->maxCorner.z == 0.f) {
			return b2;
		}
		else {
			return BoundingBox(glm::vec3(glm::min(this->minCorner.x, b2.minCorner.x),
										 glm::min(this->minCorner.y, b2.minCorner.y),
										 glm::min(this->minCorner.z, b2.minCorner.z)),
							   glm::vec3(glm::max(this->maxCorner.x, b2.maxCorner.x),
										 glm::max(this->maxCorner.y, b2.maxCorner.y),
										 glm::max(this->maxCorner.z, b2.maxCorner.z)));
		}
	}

	BoundingBox operator || (const glm::vec3& p) const {
		return BoundingBox(glm::vec3(glm::min(this->minCorner.x, p.x),
									 glm::min(this->minCorner.y, p.y),
									 glm::min(this->minCorner.z, p.z)),
						   glm::vec3(glm::max(this->maxCorner.x, p.x),
						   			 glm::max(this->maxCorner.y, p.y),
						   			 glm::max(this->maxCorner.z, p.z)));
	}

	//https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
	__host__ __device__ bool intersectBefore(const Ray &ray, const glm::vec3 &invDir, float limit) const {
		float nearDistance = 0.0f, farDistance = limit;
		for (int axis = 0; axis < 3; ++axis) {
			if (ray.direction[axis] == 0.0f) {
				if (ray.origin[axis] < minCorner[axis] || ray.origin[axis] > maxCorner[axis]) return false;
			} else {
				float a = (minCorner[axis] - ray.origin[axis]) * invDir[axis];
				float b = (maxCorner[axis] - ray.origin[axis]) * invDir[axis];
				nearDistance = fmaxf(nearDistance, fminf(a, b));
				farDistance = fminf(farDistance, fmaxf(a, b));
				if (nearDistance > farDistance) return false;
			}
		}
		return nearDistance < limit;
	}

	__host__ __device__ bool AABBIntersect2(const Ray &ray, const glm::vec3 &invDir) const {
		float txMin = (minCorner.x - ray.origin.x) * invDir.x; // 0-1 
		float txMax = (maxCorner.x - ray.origin.x) * invDir.x;
		float tyMin = (minCorner.y - ray.origin.y) * invDir.y;
		float tyMax = (maxCorner.y - ray.origin.y) * invDir.y;
		float tzMin = (minCorner.z - ray.origin.z) * invDir.z;
		float tzMax = (maxCorner.z - ray.origin.z) * invDir.z;

		float tmin = glm::max(glm::max(glm::min(txMin, txMax), glm::min(tyMin, tyMax)), glm::min(tzMin, tzMax));
		float tmax = glm::min(glm::min(glm::max(txMin, txMax), glm::max(tyMin, tyMax)), glm::max(tzMin, tzMax));

		// if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
		if (tmax < 0) { return false; }
		// if tmin > tmax, ray doesn't intersect AABB
		if (tmin > tmax) { return false; }

		return true;
	}
};
