#pragma once
#include <host_defines.h>
#include "GeometryData.h"
#include "ShadingData.h"
#include "LightingData.h"

// Define commonly used data in constant memory
// Note that these data structures just store device pointers to the actual data which
// is not in constant memory
extern __constant__ GeometryData g_geometryData;
extern __constant__ ShadingData g_shadingData;
extern __constant__ LightingData g_lightingData;
