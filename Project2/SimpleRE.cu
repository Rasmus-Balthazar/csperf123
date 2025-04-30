#include <stdbool.h>
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

typedef struct {
    int start_index;
    int length;
    int pattern_idx;
} Match;

bool matches(char pattern, char text);

__global__ void simple_gpu_re(char *text, int text_len, int pattern_count, char **patterns, int patterns_len[], unsigned int matches_found[], Match match_arr[]) {
    int pattern_len = patterns_len[blockIdx.x];
    int stride = blockDim.x;
    for (int pattern_index = blockIdx.x; pattern_index < gridDim.x; pattern_index += gridDim.x) {
        char *pattern = patterns + pattern_index;
        for (int i = threadIdx.x; i < text_len; i += stride) {
            int pattern_off = 0;
            int text_off = 0;
            while (matches(pattern[pattern_off], text[i + text_off])) {
                pattern_off++;
                text_off++;
                // If the offset is longer than the pattern length we have found it
                if (pattern_off > pattern_len) {
                    unsigned int val = matches_found[pattern_index];
                    // We are relying on the checks not being exhaustive by doing val > i before atomicCAS
                    while (val > i && atomicCAS(matches_found + pattern_index, val, i) > i) {
                        val = matches_found[pattern_index];
                        // Compares b to a, and if true then writes c into a.
                    }
                    break;
                }
            }
            if ((i + stride) > matches_found[pattern_index] || (i+stride) > text_len) 
            {
                __syncthreads(); // Synchronize threads in the block
                if (threadIdx.x == matches_found[pattern_index]%stride) {
                    match_arr[pattern_index] = {
                        start_index = i,
                        length = text_off,
                        pattern_idx = pattern_index
                    };
                }
                break;
            }
                // If match here, collection process can start,
            
        }
    }
    
    // collect within block to find first position of pattern matched

    // Collect accross blocks, whith the match for each pattern
}

// Update this to work with tokens, and return how much of text was consumed
bool matches(char pattern, char text)
{
    return pattern == text;
}