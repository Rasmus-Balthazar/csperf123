#include "main.cuh"

// Input, how much is left of input, Pattern, Pattern length
/** Progression
 * Literal Search (ignorant) https://github.com/cli117/thesis_work/blob/main/literal_match_normal/literal_match.cu
 * Begin adding rules - only extract relevant RE patterns as Literal Search
 * Wildcards
 * Repetitions
 * Ranges/Sets
 * Ors/Options
 */

#define BLOCK_SIZE 256  // Number of threads per block

using Clock = std::chrono::high_resolution_clock;


int main(int argc, const char * argv[]) {
    auto start_total = Clock::now();

    auto start_host_preprocess = Clock::now();
    PatternsInformation p = process_patterns(argv[2]);
    RegEx* h_regexes = tokenize_regex(p);
    int num_tokens = h_regexes[p.num_patterns-1].token_offset + h_regexes[p.num_patterns-1].token_count;

    char* h_text = read_file(argv[1]);
    int text_len = strlen(h_text);
    unsigned int *h_matches_found = (unsigned int *)calloc(p.num_patterns, sizeof(unsigned int));
    for (int i = 0; i < p.num_patterns; i++) {
        h_matches_found[i] = -1u;
    }

    Match* h_match_arr = (Match*)calloc(p.num_patterns, sizeof(Match));
    auto end_host_preprocess = Clock::now();

    auto start_device_alloc = Clock::now();
    char* d_text;
    unsigned int* d_matches_found;
    Match* d_match_arr;
    int* d_num_patterns;
    RegEx* d_regexes;
    Token* d_tokens;

    cudaMalloc((void **)&d_num_patterns, sizeof(int));
    cudaMalloc((void **)&d_text, text_len * sizeof(char));
    cudaMalloc((void **)&d_matches_found, p.num_patterns * sizeof(unsigned int));
    cudaMalloc((void **)&d_match_arr, p.num_patterns * sizeof(Match));
    cudaMalloc((void **)&d_regexes, p.num_patterns * sizeof(RegEx));
    cudaMalloc((void **)&d_tokens, num_tokens * sizeof(Token));
    auto end_device_alloc = Clock::now();

    auto start_data_transfer = Clock::now();
    cudaMemcpy(d_num_patterns, &(p.num_patterns), sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_text, h_text, text_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matches_found, h_matches_found, p.num_patterns * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_match_arr, h_match_arr, p.num_patterns * sizeof(Match), cudaMemcpyHostToDevice);
    cudaMemcpy(d_regexes, h_regexes, p.num_patterns * sizeof(RegEx), cudaMemcpyHostToDevice);
    for (int i = 0; i < p.num_patterns; i++) {
        cudaMemcpy(d_tokens + h_regexes[i].token_offset, h_regexes[i].tokens, h_regexes[i].token_count * sizeof(Token), cudaMemcpyHostToDevice);
    }
    auto end_data_transfer = Clock::now();

    // GPU Kernel timing using CUDA events
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid(p.num_patterns);
    simple_gpu_re<<<blocksPerGrid, threadsPerBlock>>>(d_text, text_len, d_regexes, d_tokens, d_num_patterns, d_matches_found, d_match_arr);
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);

    auto start_post_gpu = Clock::now();
    cudaMemcpy(h_match_arr, d_match_arr, p.num_patterns * sizeof(Match), cudaMemcpyDeviceToHost);

    for(int i = 0; i < p.num_patterns; i++) {
        char* pattern_at_index_i = p.formatted_patterns + p.patterns[i].pattern_text_offset;
        if (!h_match_arr[i].length) {
            printf("no match found for pattern: \"%s\"\n", pattern_at_index_i);
        } else {
            printf("match found for pattern: \"%s\" at position %i\n", pattern_at_index_i, h_match_arr[i].start_index);
        }
    }
    auto end_post_gpu = Clock::now();

    auto end_total = Clock::now();

    // Print durations
    auto duration_ms = [](auto start, auto end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };

    printf("\n--- Timing Summary ---\n");
    printf("Host preprocessing:    %lld ms\n", duration_ms(start_host_preprocess, end_host_preprocess));
    printf("Device memory alloc:   %lld ms\n", duration_ms(start_device_alloc, end_device_alloc));
    printf("Data transfer to GPU:  %lld ms\n", duration_ms(start_data_transfer, end_data_transfer));
    printf("Kernel execution:      %.3f ms\n", kernel_ms);
    printf("Post GPU processing:   %lld ms\n", duration_ms(start_post_gpu, end_post_gpu));
    printf("Total runtime:         %lld ms\n", duration_ms(start_total, end_total));

    return 0;
}