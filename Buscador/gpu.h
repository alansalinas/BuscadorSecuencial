

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t evaluarCuda(float *c, float vi, int rl, float vd, size_t size);