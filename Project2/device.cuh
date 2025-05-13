#ifndef DEF_DEVICE
#define DEF_DEVICE

#include "preprocessing.cuh"
#include <cuda_runtime.h>

__device__ int matches(char pattern, char text);
__global__ void simple_gpu_re(char *text, int text_len, char *formatted_patterns, Pattern *patterns, int* num_patterns, unsigned int matches_found[], Match match_arr[], RegEx *regexes, Token* tokens);

#endif