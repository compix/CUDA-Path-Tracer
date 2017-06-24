#pragma once
#include "engine/input/Input.h"
#include "engine/gui/GUIElements.h"
#include "engine/util/commands/MoveCommand.h"
#include "engine/rendering/architecture/RenderPipeline.h"
#include "engine/util/ECSUtil/EntityPicker.h"
#include "engine/gui/GUITexture.h"
#include "engine/gui/StatsWindow.h"
#include "application/cuda/cuda_path_tracer.h"

class PathTracingApp;
struct GUISettings;

class PathTracingGUI : public InputHandler
{
    struct EntityPickRequest
    {
        EntityPickRequest() {}
        EntityPickRequest(int x, int y)
            :x(x), y(y) {}

        int x{ 0 };
        int y{ 0 };

        bool requested{ true };
    };
public:
    PathTracingGUI::PathTracingGUI(RenderPipeline* renderPipeline, PathTracingApp* pathTracingApp);
    void update();

    void showEntityTree();

    void moveCameraToEntity(ComponentPtr<CameraComponent>& camera, Entity& entity);

private:
    void subTree(const ComponentPtr<Transform>& transform);
    void guiShowFPS();
    void showComponents(const Entity& entity) const;
    void showEntityWindow(const Entity& entity);
    void onEntityClicked(Entity& clickedEntity);
    void showSettings(const std::string& label, const GUISettings* settings) const;
    void showTextures(const ImVec2& canvasSize, std::initializer_list<GUITexture> textures) const;
    void showViewMenu();
    void showSceneMenu();
    void showEditorMenu();
    void showHelpMenu() const;
    void showControlWindow();

protected:
    void onMouseDown(const SDL_MouseButtonEvent& e) override;
    void onWindowEvent(const SDL_WindowEvent& windowEvent) override;

private:
    Entity m_selectedEntity;
    int m_treeNodeIdx{ 0 };
    MoveCommand m_cameraMoveCommand;
    RenderPipeline* m_renderPipeline;
    PathTracingApp* m_pathTracingApp;

    bool m_showGBuffers{ false };
    GUIWindow m_mainWindow{ "Main Window" };
    GUIWindow m_fpsWindow{ "FPS" };
    GUIWindow m_entityWindow{ "Entity Window" };
    GUIWindow m_consoleWindow{ "Console" };
    GUIWindow m_controlWindow{ "Control Window" };
    GUIWindow m_helpWindow{ "Help Window" };

    std::unique_ptr<EntityPicker> m_entityPicker;
    std::unique_ptr<StatsWindow> m_statsWindow;
    bool m_showSettings{ true };

    EntityPickRequest m_entityPickRequest;

    bool m_visualizeTexture{ false };

    bool m_showObjectCoordinateSystem{ true };
    bool m_showObjectBBox{ true };
};
