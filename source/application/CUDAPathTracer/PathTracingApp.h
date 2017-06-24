#pragma once
#include <engine/Application.h>
#include <engine/input/Input.h>
#include "engine/util/Timer.h"
#include "engine/ecs/EntityManager.h"
#include "application/cuda/cuda_path_tracer.h"
#include "engine/rendering/Texture2D.h"
#include <cuda_gl_interop.h>
#include "engine/rendering/renderer/SimpleMeshRenderer.h"
#include <memory>
#include "engine/rendering/shader/Shader.h"
#include "GUI/PathTracingGUI.h"
#include "engine/rendering/architecture/RenderPipeline.h"

#define SCENE_COUNT 2

class PathTracingApp : public Application, InputHandler
{
    struct SceneDesc
    {
        std::string path;
        std::string texturePath;
        float scale{0.0f};

        SceneDesc() {}
        SceneDesc(const std::string& path, const std::string& texPath, float scale)
            :path(path), texturePath(texPath), scale(scale) {}
    };
public:
	PathTracingApp();

    void update() override;
    void initUpdate() override;

    bool moveCamera(Seconds deltaTime) const;

    void startPathTracing();
    void setStandardSurfaceShadingActive();
    bool isPathTracing() const { return m_usePathTracer; }
protected:
    void onKeyDown(SDL_Keycode keyCode) override;

private:
	void resize(int width, int height);
	void linkToCUDA(cudaGraphicsResource_t& cudaTexture, GLuint image, cudaArray_t& cudaArray);
    void linkToCUDA(cudaGraphicsResource_t& cudaTexture, GLuint image, cudaMipmappedArray_t& cudaArray) const;
    void createDemoScene();
    void uploadSceneToPathTracer();
    void uploadTexturesToPathTracer();
    void extractScene(
        ComponentPtr<Transform> transform, 
        bool processChildren, 
        CUDA::HostSceneDescription& sceneDesc, 
        uint32_t& triangleIdx);

    CUDA::Ray screenToRay(const glm::vec3& p) const;
    void loadScene(const SceneDesc& sceneDesc);
protected:
	void quit() override;
	void onWindowEvent(const SDL_WindowEvent& windowEvent) override;

private:
    std::unique_ptr<RenderPipeline> m_renderPipeline;
	unsigned char* m_textureBuffer{ nullptr };
	CUDAPathTracer m_cudaPathTracer;

	Texture2D m_texture;
	cudaGraphicsResource_t m_cudaTexture2D{ nullptr };
	cudaArray_t m_cudaScreenArray;
	std::shared_ptr<SimpleMeshRenderer> m_fullscreenQuadRenderer;
	std::shared_ptr<Shader> m_fullscreenQuadShader;
	std::unique_ptr<PathTracingGUI> m_gui;
    bool m_usePathTracer{false};
    Entity m_directionalLight;
    ComponentPtr<Transform> m_sceneRoot;
    bool m_guiEnabled{ true };
    std::vector<cudaGraphicsResource_t> m_texResources;
    std::unordered_map<TextureID, uint32_t> m_cudaTexIds; // Mapping OpenGL tex ids to (own) cuda tex ids
    std::shared_ptr<Shader> m_forwardShader;

    bool m_allowPause{ false };
    bool m_jitter{ true };
    int m_screenTexDiv{ 1 };
    int m_selectedSceneIdx{ 0 };
    SceneDesc m_scenes[SCENE_COUNT];
};
