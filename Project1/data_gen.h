#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    uint64_t partitionKey;
    uint64_t payload;
} Tuple;

uint64_t rand_64bit();
Tuple *new_tuple(uint64_t key);
void shuffle(Tuple **array, int n);
void gen_input(Tuple **result, uint64_t n);
Tuple** gen_data(int n);