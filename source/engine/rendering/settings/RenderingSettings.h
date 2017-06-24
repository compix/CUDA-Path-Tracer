#pragma once
#include <vector>
#include "engine/gui/GUIElements.h"
#include "engine/gui/GUISettings.h"

struct ShadowSettings : GUISettings
{
    ShadowSettings() { guiElements.insert(guiElements.end(), {&usePoissonFilter, &depthBias}); }

    CheckBox usePoissonFilter{"Use Poisson Filter", true};
    uint32_t shadowMapResolution{4096};
    SliderFloat depthBias{"Depth Bias", 0.013f, 0.0001f, 0.1f, "%.6f"};
};

struct RenderingSettings : GUISettings
{
    RenderingSettings() { guiElements.insert(guiElements.end(), {&wireFrame, &pipeline, &cullBackFaces, &brdfMode }); }

    CheckBox wireFrame{"Wireframe", false};
    ComboBox pipeline = ComboBox("Pipeline", {"GI", "Forward"}, 0);
    CheckBox cullBackFaces{ "Cull Back Faces", false };
    ComboBox brdfMode = ComboBox("BRDF", { "Blinn-Phong", "Cook-Torrance"}, 1);
};

extern ShadowSettings SHADOW_SETTINGS;
extern RenderingSettings RENDERING_SETTINGS;
