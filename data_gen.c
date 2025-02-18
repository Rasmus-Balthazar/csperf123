#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define MAX_N 1000
// https://stackoverflow.com/questions/22727404/making-a-tuple-in-c
typedef struct {
    uint64_t partition_key;
    uint64_t payload;
} Tuple;

uint64_t rand_64bit() {
    // call rand once, cast it to 64 bits,
    // move the first 32 bits to the left
    // bitwise OR  with another rand call
    // to ensure that lower 32 bits are filled
    // bam, random 64 bit unsigned int
    return ((uint64_t)rand()) << 32 | rand();
}

Tuple *new_tuple(uint64_t key) {
    Tuple *new = (Tuple*)malloc(sizeof(Tuple));
    new->partition_key = key;
    new->payload = rand_64bit();
    return new;
}

// shuffle
void shuffle(Tuple **array, int n) {
    if(n > 1) {
        for(int i = 0; i < n -1; i++) {
            int j = rand() % n;
            Tuple *temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}

// gen_input
void gen_input(Tuple **result, uint64_t n) {
    for(int i = 0; i < n; i++) {
        *(result+i) = new_tuple(i);
        /* uint64_t payload = rand_64bit(); */
        /* result[i].partition_key = i; */
        /* result[i].payload = payload; */
    }
    shuffle(result,n);
}


int main() {
    srand(4617929);
    int n = 10;
    Tuple **res = (Tuple**)calloc(n, sizeof(Tuple*));

    gen_input(res, n);
    for(int i = 0; i < n; i++) {
        printf("key: %ld payload: %ld\n", (long)res[i]->partition_key, (long)res[i]->payload);
    }
    free(res);
    return 0;
    }
