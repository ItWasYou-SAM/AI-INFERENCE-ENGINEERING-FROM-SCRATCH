// Shared helpers for the Module 7 labs: the Module 6 helpers (and through them,
// Modules 4 and 5), plus NVTX ranges so the profilers can show named phases.
#pragma once
#include <nvtx3/nvToolsExt.h>
#include "../m06/common.cuh"

// An NVTX range lasts for the enclosing scope and shows up as a named bar on the
// Nsight Systems timeline:  { Range r("step"); ... }
struct Range {
    explicit Range(const char* name) { nvtxRangePushA(name); }
    ~Range() { nvtxRangePop(); }
};
