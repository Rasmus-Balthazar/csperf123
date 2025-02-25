#include "data_gen.h"

#define MAX_N 1000
// https://stackoverflow.com/questions/22727404/making-a-tuple-in-c

int main1() {
    Tuple **res = gen_data(10);    

    for(int i = 0; i < 10; i++) {
        printf("key: %ld payload: %ld\n", (long)res[i]->partitionKey, (long)res[i]->payload);
    }
    return 0;
}

Tuple** gen_data(uint64_t n) {
    Tuple **res = (Tuple**)calloc(n, sizeof(Tuple*));

    gen_input(res, n);
    return res;
}

uint64_t rand_64bit() {
    // call rand once, cast it to 64 bits,
    // move the first 32 bits to the left
    // bitwise OR  with another rand call
    // to ensure that lower 32 bits are filled
    // bam, random 64 bit unsigned int
    return ((uint64_t)rand()) << 32 | rand();
}

Tuple *new_tuple(uint64_t key, Tuple *new) {
    // Tuple *new = (Tuple*)malloc(sizeof(Tuple));
    new->partitionKey = key;
    new->payload = rand_64bit();
    return new;
}

// shuffle
void shuffle(Tuple **array, uint64_t n) {
    if(n > 1) {
        for(int i = 0; i < n; i++) {
            int j = rand() % n;
            Tuple *temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}

// gen_input
void gen_input(Tuple **result, uint64_t n) {
    Tuple *tuples = (Tuple*)calloc(n, sizeof(Tuple));
    for(int i = 0; i < n; i++) {
        *(result+i) = new_tuple(i, &tuples[i]);
        /* uint64_t payload = rand_64bit(); */
        /* result[i].partitionKey = i; */
        /* result[i].payload = payload; */
    }
    shuffle(result,n);
}
