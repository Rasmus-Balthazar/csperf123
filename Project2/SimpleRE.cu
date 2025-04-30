#include <stdbool.h>
#include <string.h>
#include <cuda_runtime.h>
#include <stdio.h>

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

__device__ int matches(char pattern, char text);

__global__ void simple_gpu_re(char *text, int text_len, int pattern_count, char *patterns[], int patterns_len[], unsigned int matches_found[], Match match_arr[]) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; i++)
        {
            printf("pattern %i: %s", i, patterns[i]);
        }
    }

    int pattern_len = patterns_len[blockIdx.x];
    int stride = blockDim.x;
    for (int pattern_index = blockIdx.x; pattern_index < gridDim.x; pattern_index += gridDim.x) {
        char *pattern = patterns[pattern_index];
        for (int i = threadIdx.x; i < text_len; i += stride) {
            int pattern_off = 0;
            int text_off = 0;
            int does_match;
            do
            {
                does_match = matches(pattern[pattern_off], text[i + text_off]);
                pattern_off+= does_match;
                text_off+= does_match;
                // If the offset is longer than the pattern length we have found it
                if (pattern_off >= pattern_len) {
                    printf("Matched pattern \"%s\" on thread %i\n", pattern, threadIdx.x);
                    unsigned int val = matches_found[pattern_index];
                    // We are relying on the checks not being exhaustive by doing val > i before atomicCAS
                    while (val > i && atomicCAS(matches_found + pattern_index, val, i) > i) {
                        val = matches_found[pattern_index];
                        // Compares b to a, and if true then writes c into a.
                    }
                    break;
                }
            } while (does_match);
            
            if ((i + stride) > matches_found[pattern_index] || (i+stride) > text_len) 
            {
                // If match here, collection process can start,
                __syncthreads(); // Synchronize threads in the block
                if (threadIdx.x == matches_found[pattern_index]%stride) {
                    match_arr[pattern_index].start_index = i;
                    match_arr[pattern_index].length = text_off;
                    match_arr[pattern_index].pattern_idx = pattern_index;

                    printf("Match for pattern \"%s\" found at %i\n", patterns[pattern_index], i);
                }
                break;
            }
        }
    }
}

// Update this to work with tokens, and return how much of text was consumed
__device__ int matches(char pattern, char text) {
    printf("Trying to match %c and %c\n", pattern, text);
    return pattern == text;
}

__global__ void print_char_array(char* array, int array_size) {
    for(int i = 0; i < array_size; i++) {
        printf("Char at index: %i was %c", i, array[i]);
    }
}

#define BLOCK_SIZE 256  // Number of threads per block
#define ARRAY_SIZE 1024  // Size of the input arrays

int main() {
    //h_ for host 
    char* h_text = "dette er en lang test tekst xD";
    int text_len = strlen(h_text);
    char* h_patterns[] = { "test", "er", "nope" };
    int h_pattern_lens[] = {4, 2, 4}; //because of terminating char
    unsigned int h_matches_found[] = {-1u, -1u, -1u};
    Match* h_match_arr = (Match*)calloc(3, sizeof(Match)); 

    for(int i = 0; i )

    // Device data allocation
    // d_ for device 
    char* d_text;
    char** d_patterns;
    int* d_pattern_lengths;
    unsigned int* d_matches_found;
    Match* d_match_arr;

    cudaMalloc((void **)&d_text, text_len * sizeof(char));
    cudaMalloc((void **)&d_patterns, (5+3+5)*sizeof(char));
    cudaMalloc((void **)&d_pattern_lengths, 3*sizeof(int));
    cudaMalloc((void **)&d_matches_found, 3*sizeof(unsigned int));
    cudaMalloc((void **)&d_match_arr, 3*sizeof(Match));

    print_char_array<<<1,1>>>(d_patterns, 3);
    cudaDeviceSynchronize();
    // Copy input arrays to device
    cudaMemcpy(d_text, h_text, text_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns, h_patterns, (5+3+5)*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern_lengths, h_pattern_lens, 3*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matches_found, h_matches_found, 3*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_match_arr, h_match_arr, 3*sizeof(Match), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    simple_gpu_re<<<blocksPerGrid, threadsPerBlock>>>(d_text, text_len, 3, d_patterns, d_pattern_lengths, d_matches_found, d_match_arr);

    cudaMemcpy(h_match_arr, d_match_arr, 3*sizeof(Match), cudaMemcpyDeviceToHost);
}