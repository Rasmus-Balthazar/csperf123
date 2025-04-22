#include <cuda_runtime.h>

// Input, how much is left of input, Pattern, Pattern length
/** Progression
 * Literal Search (ignorant) https://github.com/cli117/thesis_work/blob/main/literal_match_normal/literal_match.cu
 * Begin adding rules - only extract relevant RE patterns as Literal Search
 * Wildcards
 * Repetitions
 * Ranges/Sets 
 * Ors/Options
 */

 /**
  * How you doing? 
  */

  __global__ void simple_gpu_re(char *text, int text_len, char **patterns, int[] patterns_len) {
    char *pattern = patterns+blockIdx.x;
    int pattern_len = patterns_len[blockIdx.x];
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < text_len; i += stride)
    {
        int pattern_off = 0;
        int text_off = 0;
        while (matches(pattern[pattern_off], text[i+text_off])) {
            pattern_off++;
            text_off++;
            if (pattern_off > pattern_len)
                break;
        }
    }
  }
  