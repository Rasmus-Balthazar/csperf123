#include "device.cuh"

__global__ void simple_gpu_re(char *text, int text_len, char *formatted_patterns, Pattern *patterns, int* num_patterns, unsigned int matches_found[], Match match_arr[]) {
    int stride = blockDim.x;
    //loop over patterns
    for (int pattern_index = blockIdx.x; pattern_index < *num_patterns; pattern_index += gridDim.x) {
        Pattern pattern = patterns[pattern_index];
        int pattern_len = pattern.pattern_len;
        char *pattern_text = formatted_patterns + pattern.pattern_text_offset;
        //find earliest pattern match
        for (int i = threadIdx.x; i < text_len; i += stride) {
            int pattern_off = 0;
            int text_off = 0;
            int does_match;
            do
            {
                does_match = matches(pattern_text[pattern_off], text[i + text_off]);
                pattern_off+= does_match;
                text_off+= does_match;
                // If the offset is longer than the pattern length we have found it
                if (pattern_off >= pattern_len) {
                    unsigned int val = matches_found[pattern_index];
                    // We are relying on the checks not being exhaustive by doing val > i before atomicCAS
                    while (val > i && atomicCAS(matches_found + pattern_index, val, i) > i) {
                        val = matches_found[pattern_index];
                        // Compares b to a, and if true then writes c into a.
                    }
                    break;
                }
            } while (does_match);
            
            if ((i + stride) > matches_found[pattern_index] || (i+stride) >= text_len) 
            {
                // If match here, collection process can start,
                __syncthreads(); // Synchronize threads in the block
                if (threadIdx.x == matches_found[pattern_index]%stride) {
                    match_arr[pattern_index].start_index = i;
                    match_arr[pattern_index].length = text_off;
                    match_arr[pattern_index].pattern_idx = pattern_index;
                    // printf("Match for pattern \"%s\" found at %i\n", patterns[pattern_index], i);
                }
                break;
            }
        }
    }
    __syncthreads();
}

// Update this to work with tokens, and return how much of text was consumed
__device__ int matches(char pattern, char text) {
    // printf("Trying to match %c and %c\n", pattern, text);
    return pattern == text;
}


