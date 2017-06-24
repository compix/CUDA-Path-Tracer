#pragma once
#include "Lights.h"

struct LightingData
{
    HOST_AND_DEVICE LightingData() {}

    uint32_t dirLightCount;
    CUDA::DirectionalLight* dirLights;

    uint8_t indirectBounceCount;
    float indirectIntensity;
};
