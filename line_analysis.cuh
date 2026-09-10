#include <cuda_runtime.h>
#include <stdint.h>

void submitLineAnalysis(uint8_t *d_gray, uint8_t *d_canny, int width, int height, cudaStream_t &stream, int *d_result);
