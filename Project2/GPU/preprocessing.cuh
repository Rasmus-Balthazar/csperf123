#ifndef DEF_PREPROCESS
#define DEF_PREPROCESS

#include <algorithm>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

typedef struct {
    int start_index;
    unsigned int length;
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

typedef struct {
    int mode; // 0 = literal, 1 = wildcard
    unsigned int min_count, max_count;
    char to_match;

    int backtracing; // 0 = no, 1 = yes
    int match_count;
} Token;

typedef struct {
    Token* tokens;
    int token_offset;
    int token_count;
} RegEx;

__host__ char* read_file(const char* file_path);
__host__ PatternsInformation process_patterns(const char* file_path);
__host__ int count_lines_in_file(const char *file_path);
__host__ RegEx* tokenize_regex(PatternsInformation p);
__host__ int tokenize_helper(PatternsInformation, int, Token*);

#endif