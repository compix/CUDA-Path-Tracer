#pragma once
#ifdef __CUDACC__
#include <cuda.h>
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>
#include "geometry.h"
#include "HostSceneDescription.h"
#include "Scene.h"

class BBox;

class CUDAPathTracer
{
public:
	CUDAPathTracer();
    ~CUDAPathTracer();
    void setScreenTexture(const cudaArray_t* screenArray);

	void setPrimaryRays(const CUDA::Ray& ray00, const CUDA::Ray& ray10, const CUDA::Ray& ray11, const CUDA::Ray& ray01);
	void setEyePos(const glm::vec3& eyePos) { m_eyePos = eyePos; }

	void raytrace(int screenWidth, int screenHeight);

	void upload(const CUDA::HostSceneDescription& sceneDesc);
    CUDA::Scene* scene() const { return m_scene.get(); }

    void resetFrames() { m_localFrameNumber = 0; }
private:
	// 0: (0,0), 1: (1,0), 2: (1,1), 3: (0,1)
	CUDA::Ray m_ray00, m_ray10, m_ray11, m_ray01;
	glm::vec3 m_eyePos;
	
	std::shared_ptr<CUDA::Scene> m_scene{nullptr};

    float m_startTime = 0.0f;
    bool m_displayedTimeForFrame{ false };
    cudaSurfaceObject_t m_screenTexture{0};
    uint32_t m_localFrameNumber;
};
