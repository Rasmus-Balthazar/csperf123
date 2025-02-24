#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifndef DG_H
#define DG_H

typedef struct {
    uint64_t partitionKey;
    uint64_t payload;
} Tuple;

uint64_t rand_64bit();
Tuple *new_tuple(uint64_t key);
void shuffle(Tuple **array, uint64_t n);
void gen_input(Tuple **result, uint64_t n);
Tuple** gen_data(uint64_t n);

#endif