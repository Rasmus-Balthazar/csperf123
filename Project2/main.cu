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

#define BLOCK_SIZE 8  // Number of threads per block
#define ARRAY_SIZE 64  // Size of the input arrays

int main(int argc, const char * argv[]) {
    //h_ for host 
    PatternsInformation p = process_patterns(argv[2]);
    RegEx* h_regexes = tokenize_regex(p);
    int num_tokens = h_regexes[p.num_patterns-1].token_offset+h_regexes[p.num_patterns-1].token_count;
    
    char* h_text = read_file(argv[1]);
    int text_len = strlen(h_text);
    unsigned int *h_matches_found = (unsigned int *)calloc(p.num_patterns, sizeof(unsigned int));
    for (int i = 0; i < p.num_patterns; i++)
    {
        h_matches_found[i]=-1u;
    }
    
    Match* h_match_arr = (Match*)calloc(p.num_patterns, sizeof(Match)); 


    // Device data allocation
    // d_ for device 
    Pattern* d_patterns;
    char* d_text;
    char* d_patterns_text;
    unsigned int* d_matches_found;
    Match* d_match_arr;
    int* d_num_patterns;
    RegEx* d_regexes;
    Token* d_tokens;
    
    cudaMalloc((void **)&d_num_patterns, sizeof(int));
    cudaMalloc((void **)&d_text, text_len * sizeof(char));
    cudaMalloc((void **)&d_patterns, p.num_patterns*sizeof(Pattern));
    cudaMalloc((void **)&d_patterns_text, p.formatted_length*sizeof(char));
    cudaMalloc((void **)&d_matches_found, p.num_patterns*sizeof(unsigned int));
    cudaMalloc((void **)&d_match_arr, p.num_patterns*sizeof(Match));
    cudaMalloc((void **)&d_regexes, p.num_patterns*sizeof(RegEx));
    cudaMalloc((void **)&d_tokens, num_tokens*sizeof(Token));


    // Copy input arrays to device
    cudaMemcpy(d_num_patterns, &( p.num_patterns ), sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_text, h_text, text_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns, p.patterns, p.num_patterns*sizeof(Pattern), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matches_found, h_matches_found, p.num_patterns*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_match_arr, h_match_arr, p.num_patterns*sizeof(Match), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns_text, p.formatted_patterns, p.formatted_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_regexes, h_regexes, p.num_patterns*sizeof(RegEx),cudaMemcpyHostToDevice);
    for (int i = 0; i < p.num_patterns; i++)
    {
        cudaMemcpy(d_tokens+h_regexes[i].token_offset, h_regexes[i].tokens, h_regexes[i].token_count*sizeof(Token),cudaMemcpyHostToDevice);
    }
    


    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    simple_gpu_re<<<blocksPerGrid, threadsPerBlock>>>(d_text, text_len, d_patterns_text, d_patterns, d_num_patterns, d_matches_found, d_match_arr, d_regexes, d_tokens);

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