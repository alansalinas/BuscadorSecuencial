

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t bandasCuda(float *c, float vi, int rl, float vd, size_t size);

cudaError_t ohmsCuda(float *c, float vi, int rl, float vd, int r1, int r2, int m1, int m2, size_t size);