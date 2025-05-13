#ifndef DEF_DEVICE
#define DEF_DEVICE

#include "preprocessing.cuh"
#include <cuda_runtime.h>

__device__ int matches(Token *pattern, char text, int *token_off, int text_len, int text_start, int *text_off);
__global__ void simple_gpu_re(char *text, int text_len, RegEx *regexes, Token* tokens, int* num_patterns, unsigned int matches_found[], Match match_arr[]);

#endif