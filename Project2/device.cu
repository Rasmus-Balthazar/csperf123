#include "device.cuh"

__global__ void simple_gpu_re(char *text, int text_len, RegEx *regexes, Token *tokens, int* num_patterns, unsigned int matches_found[], Match match_arr[]) {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     for (int i = 0; i < regexes[*num_patterns-1].token_offset+regexes[*num_patterns-1].token_count; i++)
    //     {
    //         printf("%c", tokens[i].to_match);
    //     }
    // }
    

    int stride = blockDim.x;
    //loop over regexes
    for (int pattern_index = blockIdx.x; pattern_index < *num_patterns; pattern_index += gridDim.x) {
        RegEx re = regexes[pattern_index];
        //find earliest pattern match
        for (int text_start = threadIdx.x; text_start < text_len; text_start += stride) {
            int token_off = 0;
            int text_off = 0;
            int does_match;
            do {
                does_match = matches(tokens+re.token_offset+token_off, text[text_start + text_off]);
                token_off += does_match;
                text_off += does_match;
                // If the token offset is longer than the amount of token we have then we have found it
                if (token_off >= re.token_count && does_match) {
                    unsigned int last_val = matches_found[pattern_index];
                    // We are relying on the checks not being exhaustive by doing val > i before atomicCAS
                    while (last_val > text_start && atomicCAS(matches_found + pattern_index, last_val, text_start) > text_start) {
                        last_val = matches_found[pattern_index];
                        // Compares b to a, and if true then writes c into a.
                    }
                    break;
                }
                if (text_start+text_off >= text_len)
                    does_match = 0;
            } while (does_match);
            
            if ((text_start + stride) > matches_found[pattern_index] || (text_start+stride) >= text_len) 
            {
                // If match here, collection process can start,
                __syncthreads(); // Synchronize threads in the block
                if (threadIdx.x == matches_found[pattern_index]%stride) {
                    match_arr[pattern_index].start_index = text_start;
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
__device__ int matches(Token *token, char text) {
    // printf("Trying to match %c and %c\n", pattern, text);
    if (token->mode)
        return 1;
    return token->to_match == text;
}


