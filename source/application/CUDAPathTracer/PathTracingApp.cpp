#include "PathTracingApp.h"
#include <engine/util/Random.h>
#include <engine/util/Timer.h>
#include <engine/rendering/util/GLUtil.h>
#include <engine/Engine.h>
#include <engine/resource/ResourceManager.h>
#include <engine/rendering/architecture/RenderPipeline.h>
#include <engine/ecs/ecs.h>
#include "engine/rendering/debug/DebugRenderer.h"
#include "engine/util/ECSUtil/ECSUtil.h"
#include "engine/rendering/renderPasses/ShadowMapPass.h"
#include "engine/util/commands/RotationCommand.h"
#include "engine/rendering/renderer/MeshRenderers.h"
#include "engine/rendering/Screen.h"
#include <cuda_runtime_api.h>
#include "application/cuda/cuda_common.h"
#include "engine/rendering/renderPasses/ForwardScenePass.h"
#include "engine/rendering/lights/DirectionalLight.h"
#include "engine/globals.h"
#include "engine/util/NamingConvention.h"
#include "GUI/PathTracingSettings.h"
#include "engine/rendering/settings/RenderingSettings.h"

PathTracingApp::PathTracingApp()
{
    Input::subscribe(this);

    Random::randomize();
}

void PathTracingApp::extractScene(ComponentPtr<Transform> transform, bool processChildren, CUDA::HostSceneDescription& sceneDesc, uint32_t& triangleIdx)
{
    // Go through MeshRenderers, apply transform to triangles, store triangles in sceneDesc
    // Also: store meterial information
    auto meshRenderer = transform->getComponent<MeshRenderer>();

    if (meshRenderer)
    {
        auto mesh = meshRenderer->getMesh();

        for (int meshIdx = 0; meshIdx < mesh->getSubMeshes().size(); ++meshIdx)
        {
            auto& subMesh = mesh->getSubMeshes()[meshIdx];
            auto material = meshRenderer->getMaterial(meshIdx);

            for (size_t i = 0; i < subMesh.indices.size(); i += 3, ++triangleIdx)
            {
                CUDA::Triangle triangle(
                    transform->transformPointToWorld(subMesh.vertices[subMesh.indices[i]]),
                    transform->transformPointToWorld(subMesh.vertices[subMesh.indices[i + 1]]),
                    transform->transformPointToWorld(subMesh.vertices[subMesh.indices[i + 2]]),
                    static_cast<TriangleIndex>(triangleIdx));

                sceneDesc.triangles.push_back(triangle);

                CUDA::TriangleInfo info;
                glm::vec3 normal = triangle.computeNormal();

                for (int j = 0; j < 3; ++j)
                {
                    auto& v = info.vertices[j];
                    auto idx = subMesh.indices[i + j];

                    if (subMesh.uvs.size() > 0)
                        v.uv = subMesh.uvs[idx];

                    if (subMesh.colors.size() > 0)
                        v.color = subMesh.colors[idx];
                    else
                    {
                        v.color = material->getColor(NC::color());
                    }

                    if (subMesh.normals.size() > 0)
                        v.normal = subMesh.normals[idx];
                    else
                        v.normal = triangle.computeNormal();

                    if (subMesh.tangents.size() > 0)
                    {
                        v.tangent = subMesh.tangents[idx];

                        if (subMesh.bitangents.size() > 0)
                            v.bitangent = subMesh.bitangents[idx];
                        else
                            v.bitangent = glm::cross(info.vertices[j].tangent, normal);
                    }
                }
                TextureID texID;
                if (material->tryGetTexture2D(NC::diffuseTexture(0), texID))
                {
                    info.material.albedoIdx = m_cudaTexIds[texID];
                }

                if (material->tryGetTexture2D(NC::normalMap(0), texID))
                {
                    info.material.normalMapIdx = m_cudaTexIds[texID];
                }

                glm::vec3 emission;
                if (material->tryGetColor3(NC::emissionColor(), emission))
                {
                    info.material.emission = emission;
                }

                float shininess;
                if (material->tryGetFloat(NC::shininess(), shininess))
                {
                    info.material.specularity = shininess;
                }
                
                sceneDesc.triangleInfo.push_back(info);
                sceneDesc.enclosingBBox.unite(triangle.bbox());
            }
        }
    }

    // Continue with children
    if (processChildren)
    {
        for (auto& child : transform->getChildren())
            extractScene(child, processChildren, sceneDesc, triangleIdx);
    }
}

void PathTracingApp::initUpdate()
{
    ResourceManager::setShaderIncludePath("shaders");        
	m_fullscreenQuadRenderer = MeshRenderers::fullscreenQuad();
	m_fullscreenQuadShader = ResourceManager::getShader("shaders/simple/fullscreenQuad.vert", "shaders/simple/fullscreenQuad.frag");

    createDemoScene();

    DebugRenderer::init();
    m_engine->setAllowPause(m_allowPause);

	m_textureBuffer = new unsigned char[Screen::getWidth() * Screen::getHeight() * 4];

	// Initialize CUDA - OpenGL interop
	CUDA_ERROR_CHECK(cudaSetDevice(0));
	CUDA_ERROR_CHECK(cudaGLSetGLDevice(0));
	cudaDeviceProp prop;
	CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, 0));

	// Create a 2D OpenGL floating point texture
	m_texture.create(Screen::getWidth() / m_screenTexDiv, Screen::getHeight() / m_screenTexDiv, 
	    GL_RGBA32F, GL_RGBA, GL_FLOAT, Texture2DSettings::S_T_CLAMP_TO_BORDER_MIN_MAX_NEAREST, nullptr);
	GL_ERROR_CHECK();
	linkToCUDA(m_cudaTexture2D, m_texture, m_cudaScreenArray);
    m_cudaPathTracer.setScreenTexture(&m_cudaScreenArray);

    m_renderPipeline = std::make_unique<RenderPipeline>(MainCamera);

    // Add render passes to the pipeline
    m_renderPipeline->addRenderPasses(
        std::make_shared<ShadowMapPass>(SHADOW_SETTINGS.shadowMapResolution),
        std::make_shared<ForwardScenePass>());

    m_gui = std::make_unique<PathTracingGUI>(m_renderPipeline.get(), this);

    uploadTexturesToPathTracer();

    m_initializing = false;
}

void PathTracingApp::resize(int width, int height)
{
	// Note: the order is important here:
	// 1. Unregister resource
	// 2. Resize OpenGL texture
	// 3. Reregister with CUDA
	cudaGraphicsUnregisterResource(m_cudaTexture2D);
	m_texture.resize(glm::max(width / m_screenTexDiv, 1), glm::max(height / m_screenTexDiv, 1));
	linkToCUDA(m_cudaTexture2D, m_texture, m_cudaScreenArray);
    m_cudaPathTracer.setScreenTexture(&m_cudaScreenArray);
    m_cudaPathTracer.resetFrames();
}

void PathTracingApp::linkToCUDA(cudaGraphicsResource_t& cudaTexture, GLuint image, cudaArray_t& cudaArray)
{
	CUDA_ERROR_CHECK(cudaGraphicsGLRegisterImage(&cudaTexture, image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CUDA_ERROR_CHECK(cudaGraphicsMapResources(1, &cudaTexture));
	CUDA_ERROR_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaTexture, 0, 0));
	CUDA_ERROR_CHECK(cudaGraphicsUnmapResources(1, &cudaTexture));
}

void PathTracingApp::linkToCUDA(cudaGraphicsResource_t& cudaTexture, GLuint image, cudaMipmappedArray_t& cudaArray) const
{
    CUDA_ERROR_CHECK(cudaGraphicsGLRegisterImage(&cudaTexture, image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    CUDA_ERROR_CHECK(cudaGraphicsMapResources(1, &cudaTexture));
    CUDA_ERROR_CHECK(cudaGraphicsResourceGetMappedMipmappedArray(&cudaArray, cudaTexture));
    CUDA_ERROR_CHECK(cudaGraphicsUnmapResources(1, &cudaTexture));
}

void PathTracingApp::onWindowEvent(const SDL_WindowEvent& windowEvent)
{
	switch (windowEvent.event)
	{
	case SDL_WINDOWEVENT_RESIZED:
		resize(windowEvent.data1, windowEvent.data2);
		break;
	default: break;
	}
}

void PathTracingApp::quit()
{
	if (m_cudaTexture2D)
	{
		cudaGraphicsUnregisterResource(m_cudaTexture2D);
	}

    for (auto texR : m_texResources)
        cudaGraphicsUnregisterResource(texR);
}

CUDA::Ray PathTracingApp::screenToRay(const glm::vec3& p) const
{
	// Convert to NDC space
	glm::vec3 ndcP = MainCamera->screenToNDC(p);
	glm::vec4 start(ndcP, 1.f);
	glm::vec4 end(ndcP.x, ndcP.y, 1.0f, 1.f);

	// Jittering is used for antialiasing
	float wi = 0.0f;
	float hi = 0.0f;
    if (m_jitter)
    {
        wi = 2.0f / m_texture.getWidth();
        hi = 2.0f / m_texture.getHeight();
    }
	glm::mat4 inv = glm::inverse(glm::translate(glm::vec3(Random::getFloat(-wi, wi), Random::getFloat(-hi, hi), 0.0f)) * MainCamera->viewProj());

	// Convert to world space
	start = inv * start;
	start /= start.w;

	end = inv * end;
	end /= end.w;

	return CUDA::Ray(glm::vec3(start), glm::normalize(glm::vec3(end - start)));
}

void PathTracingApp::update()
{
    if (moveCamera(Time::deltaTime()))
        m_cudaPathTracer.resetFrames();

    glFrontFace(GL_CW);

    for (auto c : ECS::getEntitiesWithComponents<CameraComponent>())
    {
        c.getComponent<CameraComponent>()->updateViewMatrix();
    }

    GL::setViewport(MainCamera->getViewport());

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    DebugRenderInfo info(MainCamera->view(), MainCamera->proj(), MainCamera->getPosition());
    DebugRenderer::begin(info);
    DebugRenderer::update();

	float w = MainCamera->getScreenWidth();
	float h = MainCamera->getScreenHeight();
	float nc = MainCamera->getNearClipPlane();

	m_cudaPathTracer.setPrimaryRays(
		screenToRay(glm::vec3(0.0f, 0.0f, nc)),
		screenToRay(glm::vec3(w, 0.0f, nc)),
		screenToRay(glm::vec3(w, h, nc)),
		screenToRay(glm::vec3(0.0f, h, nc)));
	m_cudaPathTracer.setEyePos(MainCamera->getPosition());

    glEnable(GL_DEPTH_TEST);

    ECS::lateUpdate();

    DebugRenderer::end();

    if (m_usePathTracer)
    {
        m_cudaPathTracer.raytrace(m_texture.getWidth(), m_texture.getHeight(), m_engine->getFrameNumber());
        CUDA_ERROR_CHECK(cudaStreamSynchronize(0));

        m_fullscreenQuadShader->bind();
        m_fullscreenQuadShader->bindTexture2D(m_texture, "u_textureDiffuse");
        m_fullscreenQuadRenderer->bindAndRender();
    }
    else
    {
        m_renderPipeline->update();
    }

    if (m_guiEnabled)
	    m_gui->update();
}

bool PathTracingApp::moveCamera(Seconds deltaTime) const
{
    bool moved = false;
    float speed = PathTracerSettings::DEMO.cameraSpeed * deltaTime;

    if (Input::isKeyDown(SDL_SCANCODE_W))
    {
        MainCamera->walk(speed);
        moved = true;
    }

    if (Input::isKeyDown(SDL_SCANCODE_S))
    {
        MainCamera->walk(-speed);
        moved = true;
    }

    if (Input::isKeyDown(SDL_SCANCODE_A))
    {
        MainCamera->strafe(-speed);
        moved = true;
    }

    if (Input::isKeyDown(SDL_SCANCODE_D))
    {
        MainCamera->strafe(speed);
        moved = true;
    }

    if (Input::rightDrag().isDragging())
    {
        float dx = -Input::rightDrag().getDragDelta().x;
        float dy = Input::rightDrag().getDragDelta().y;

        MainCamera->pitch(math::toRadians(dy * 0.1f));
        MainCamera->rotateY(math::toRadians(dx * 0.1f));
        SDL_SetRelativeMouseMode(SDL_TRUE);
        moved = true;
    }
    else
    {
        SDL_SetRelativeMouseMode(SDL_FALSE);
    }

    return moved;
}

void PathTracingApp::startPathTracing()
{
    uploadSceneToPathTracer();
    m_usePathTracer = true;
    m_cudaPathTracer.resetFrames();
}

void PathTracingApp::setStandardSurfaceShadingActive()
{
    m_usePathTracer = false;
}

void PathTracingApp::onKeyDown(SDL_Keycode keyCode)
{
    int lastSceneIdx = m_selectedSceneIdx;
    switch (keyCode)
    {
    case SDLK_F1:
        m_usePathTracer = !m_usePathTracer;

        if (m_usePathTracer)
            startPathTracing();
        else
            setStandardSurfaceShadingActive();
        break;
    case SDLK_F2:
        if (m_usePathTracer)
            startPathTracing();
        break;
    case SDLK_F3:
        m_guiEnabled = !m_guiEnabled;
        break;
    case SDLK_F5:
        m_engine->requestScreenshot();
        break;
    case SDLK_j:
        m_jitter = !m_jitter;
        break;
    case SDLK_KP_MINUS:
    case SDLK_MINUS:
        m_screenTexDiv /= 2;
        if (m_screenTexDiv == 0)
            m_screenTexDiv = 1;
        resize(Screen::getWidth(), Screen::getHeight());
        break;
    case SDLK_KP_PLUS:
    case SDLK_PLUS:
        m_screenTexDiv *= 2;
        resize(Screen::getWidth(), Screen::getHeight());
        break;
    default: break;
    }

    if (lastSceneIdx != m_selectedSceneIdx)
    {
        loadScene(m_scenes[m_selectedSceneIdx]);
    }
}

void PathTracingApp::loadScene(const SceneDesc& sceneDesc)
{
    if (m_sceneRoot)
    {
        m_sceneRoot.getOwner().setActive(false);
    }

    m_sceneRoot = ECSUtil::loadMeshEntities(sceneDesc.path, m_forwardShader, sceneDesc.texturePath, glm::vec3(sceneDesc.scale), true);
}

void PathTracingApp::createDemoScene()
{
    Entity camera = ECS::createEntity("Camera");
    camera.addComponent<Transform>();
    camera.addComponent<CameraComponent>();

    auto camComponent = camera.getComponent<CameraComponent>();
    auto camTransform = camera.getComponent<Transform>();

    MainCamera = camComponent;

    camComponent->setPerspective(45.0f, float(Screen::getWidth()), float(Screen::getHeight()), 0.3f, 30.0f);
    glm::vec3 cameraPositionOffset(8.625f, 6.593f, -0.456f);
    //glm::vec3 cameraPositionOffset(-0.324f, 4.428f, -14.886f);
    camTransform->setPosition(cameraPositionOffset);
    camTransform->setEulerAngles(glm::vec3(math::toRadians(10.236f), math::toRadians(-66.0f), 0.0f));
    //camTransform->setEulerAngles(glm::vec3(0.0f, 0.0, 0.0f));

    m_engine->registerCamera(camComponent);

    m_forwardShader = ResourceManager::getShader("shaders/forwardShadingPass.vert", "shaders/forwardShadingPass.frag",
        { "in_pos", "in_normal", "in_tangent", "in_bitangent", "in_uv" });
    m_scenes[0] = SceneDesc("meshes/sponza_obj/sponza.obj", "textures/sponza_textures/", 0.01f);
    m_scenes[1] = SceneDesc("meshes/cornell-box/CornellBox-Original.obj", "", 5.0f);
    //m_scenes[2] = SceneDesc("meshes/dragon.obj", "", 5.0f);
    //m_scenes[3] = SceneDesc("meshes/hairball.obj", "", 1.0f);
    loadScene(m_scenes[m_selectedSceneIdx]);

    if (m_sceneRoot)
        m_sceneRoot->setPosition(glm::vec3());

    m_directionalLight = ECS::createEntity("Directional Light");
    m_directionalLight.addComponent<DirectionalLight>();
    m_directionalLight.addComponent<Transform>();
    m_directionalLight.getComponent<Transform>()->setPosition(glm::vec3(0.0f, 20.0f, 0.f));
    m_directionalLight.getComponent<Transform>()->setEulerAngles(glm::vec3(math::toRadians(72.0f), 0.0f, 0.0f));
    m_directionalLight.getComponent<DirectionalLight>()->intensity = 1.5f;
    m_directionalLight.getComponent<DirectionalLight>()->shadowsEnabled = false;
}

void PathTracingApp::uploadSceneToPathTracer()
{
    TriangleIndex triangleIdx = 0;
    CUDA::HostSceneDescription sceneDesc;

    for (auto e : ECS::getEntitiesWithComponents<Transform, MeshRenderer>())
    {
        extractScene(e.getComponent<Transform>(), false, sceneDesc, triangleIdx);
    }

    for (auto e : ECS::getEntitiesWithComponents<DirectionalLight>())
    {
        auto dirLight = e.getComponent<DirectionalLight>();

        CUDA::DirectionalLight cudaDirLight;
        cudaDirLight.direction = e.getComponent<Transform>()->getForward();
        cudaDirLight.color = dirLight->color;
        cudaDirLight.intensity = dirLight->intensity;
        sceneDesc.dirLights.push_back(cudaDirLight);
    }

    m_cudaPathTracer.upload(sceneDesc);
}

void PathTracingApp::uploadTexturesToPathTracer()
{
    thrust::host_vector<cudaMipmappedArray_t> texArrays;
    auto textures2D = ResourceManager::getTextures2D();
    uint32_t cudaTexID = 0;
    for (auto& tex : textures2D)
    {
        tex->bind();
        cudaMipmappedArray_t texArr;
        cudaGraphicsResource_t resource;
        linkToCUDA(resource, *tex, texArr);

        m_texResources.push_back(resource);
        texArrays.push_back(texArr);
        m_cudaTexIds.insert({ *tex, cudaTexID });
        ++cudaTexID;
    }

    m_cudaPathTracer.scene()->upload(texArrays);
}
