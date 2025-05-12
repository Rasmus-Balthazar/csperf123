#include <stdbool.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

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

typedef struct {
    int pattern_text_offset;
    int pattern_len;
} Pattern;

typedef struct {
    char* formatted_patterns;
    int formatted_length;
    Pattern* patterns;
    int num_patterns;
} PatternsInformation;

__device__ int matches(char pattern, char text);

__host__ PatternsInformation process_patterns(char* file_path);

__device__ int matches(char pattern, char text);

__global__ void simple_gpu_re(char *text, int text_len, char *formatted_patterns, Pattern *patterns, int* num_patterns, unsigned int matches_found[], Match match_arr[]) {
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        printf("text: %s\ntext len: %d\nform patterns:%s %s %s\nnum patterns: %d\n",text, text_len, formatted_patterns, formatted_patterns + 5, formatted_patterns + 8, 4);
    }
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

/* New */
__host__ PatternsInformation process_patterns(const char *file_path) {
        std::string pattern;

        std::ifstream RegexFile(file_path);

        std::string pattern_collection;
        while (std::getline(RegexFile, pattern)) {
                /* TODO: process file to get the information the way we want */
                pattern_collection = pattern_collection + pattern + "\0";
        }
        RegexFile.close(); 

        /* TODO: return the correct thing */
        
        Pattern* arr = (Pattern*)calloc(3,sizeof(Pattern)); 
        arr[0].pattern_text_offset = 0;
        arr[0].pattern_len = 4;
        arr[1].pattern_text_offset = 5;
        arr[1].pattern_len = 2;
        arr[2].pattern_text_offset = 8;
        arr[2].pattern_len = 4;
        PatternsInformation p = {"test\0er\0nope\0", 13, arr, 3};
        return p;
}

__host__ int count_lines_in_file(const char *file_path) {
    std::ifstream inFile(file_path);  
    int line_count = std::count(std::istreambuf_iterator<char>(inFile), 
             std::istreambuf_iterator<char>(), '\n');
    return line_count;
}

#define BLOCK_SIZE 8  // Number of threads per block
#define ARRAY_SIZE 64  // Size of the input arrays
/* get file path from input arg  */
int main(int argc, const char * argv[]) {
    //h_ for host 
    PatternsInformation p = process_patterns(argv[1]);
    /* TODO: use the info from this p to inisialise things */ 
    char* h_text = "dette er en lang test tekst xD!!";
    int text_len = strlen(h_text);
    unsigned int h_matches_found[] = {-1u, -1u, -1u};
    Match* h_match_arr = (Match*)calloc(3, sizeof(Match)); 


    // Device data allocation
    // d_ for device 
    Pattern* d_patterns;
    char* d_text;
    char* d_patterns_text;
    unsigned int* d_matches_found;
    Match* d_match_arr;
    int* d_num_patterns;
    
    cudaMalloc((void **)&d_num_patterns, sizeof(int));
    cudaMalloc((void **)&d_text, text_len * sizeof(char));
    cudaMalloc((void **)&d_patterns, p.num_patterns*sizeof(Pattern));
    cudaMalloc((void **)&d_patterns_text, p.formatted_length*sizeof(char));
    cudaMalloc((void **)&d_matches_found, p.num_patterns*sizeof(unsigned int));
    cudaMalloc((void **)&d_match_arr, p.num_patterns*sizeof(Match));

    // Copy input arrays to device
    cudaMemcpy(d_num_patterns, &( p.num_patterns ), sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_text, h_text, text_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns, p.patterns, p.num_patterns*sizeof(Pattern), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matches_found, h_matches_found, p.num_patterns*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_match_arr, h_match_arr, p.num_patterns*sizeof(Match), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns_text, p.formatted_patterns, p.formatted_length * sizeof(char), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    simple_gpu_re<<<blocksPerGrid, threadsPerBlock>>>(d_text, text_len, d_patterns_text, d_patterns, d_num_patterns, d_matches_found, d_match_arr);

    cudaMemcpy(h_match_arr, d_match_arr, p.num_patterns*sizeof(Match), cudaMemcpyDeviceToHost);
    for(int i = 0; i < p.num_patterns; i++) {
        char* pattern_at_index_i = p.formatted_patterns + p.patterns[i].pattern_text_offset;
        if (!h_match_arr[i].length) {
            printf("no match found for pattern: \"%s\"\n", pattern_at_index_i);
        } else {
            printf("match found for pattern: \"%s\" at position %i\n", pattern_at_index_i, h_match_arr[i].start_index);
        }
    }
}