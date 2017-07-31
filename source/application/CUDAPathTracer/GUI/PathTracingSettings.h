#pragma once
#include "engine/gui/GUISettings.h"

namespace PathTracerSettings
{
    struct Demo : GUISettings
    {
        Demo()
        {
            guiElements.insert(guiElements.end(), { &cameraSpeed, &stopAtFrame });
        }

        SliderFloat cameraSpeed{ "Camera Speed", 5.0f, 1.0f, 15.0f };
        uint32_t frameNum{ 0 };
        SliderInt stopAtFrame{ "Stop At Frame", -1, -1, 65536 };

    };

    struct GISettings : GUISettings
    {
        GISettings()
        {
            guiElements.insert(guiElements.end(), {&indirectIntensity, &indirectBounceCount, &useStratifiedSampling });
        }

        SliderFloat indirectIntensity{ "Indirect Intensity", 3.0f, 0.0f, 30.0f };
        SliderInt indirectBounceCount{ "Indirect Bounces", 1, 0, 5 };
        CheckBox useStratifiedSampling{ "Use Stratified Sampling", true };
    };

    extern Demo DEMO;
    extern GISettings GI;
}
