#pragma once
#include "engine/cuda/cuda_defs.h"

struct MiscData
{
    bool useStratifiedSampling;

    HOST_AND_DEVICE MiscData() {}
};
