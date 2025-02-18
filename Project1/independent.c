#include <pthread.h>
#include "data_gen.h"

#define NUM_THREADS 4
#define KEY_BITS 10
#define NUM_PARTITIONS (2 << KEY_BITS)
#define SAMPLE_SIZE (NUM_PARTITIONS << 4)
#define PARTITION_SIZE (SAMPLE_SIZE / NUM_PARTITIONS)

typedef struct {
    int startIndex;
    int endIndex;
    Tuple** data;
    Tuple*** partitions;
} Args;

int hash_key(uint64_t key);
void *run(void *args);

int main(){
    Tuple **data = gen_data(SAMPLE_SIZE);
    Tuple ****buffers = (Tuple****)calloc(NUM_THREADS, sizeof(Tuple***));
    for (int i = 0; i < NUM_THREADS; i++) {
        *(buffers + i) = (Tuple***)calloc(NUM_PARTITIONS, sizeof(Tuple**));
    }

    pthread_t threads[NUM_THREADS];
    // int args[] = {41,42,43,44};

    for (int i = 0; i < NUM_THREADS; i++)
    {
        Args *args = (Args*)malloc(sizeof(Args));
        args->startIndex = (SAMPLE_SIZE / NUM_THREADS) * i; 
        args->endIndex = (SAMPLE_SIZE / NUM_THREADS) * (i+1);
        args->data = data;
        args->partitions = *(buffers + i);

        pthread_create(&threads[i], NULL, run, args); // If no args, set last argument to NULL
        // Note that NULL is the pthread policy, it can be used e.g. to set core affinity
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    for(int i = 0; i < PARTITION_SIZE; i++) {
        printf("Printing partition key %ld\n", (long)(*(*(*(buffers+3)+4)+i))->partitionKey);
    }
    printf("huh? how did i get here?\n");
    return 0;
}
//method hash and find where it belongs
int hash_key(uint64_t key) {
    //we know amount of partitions, so get hashing
    return (int)(key % NUM_PARTITIONS);
}

void *run(void *args) {
    Args *input = (Args *)args;
    for(int i =  0; i < NUM_PARTITIONS; i++) {
        //Allocate every partition from null to a partion of the correct size.
        *(input->partitions+i) = (Tuple**)calloc(PARTITION_SIZE, sizeof(Tuple*));
    }
    int thread_number = 5;
    for (int i = input->startIndex; i < input->endIndex; i++) {
        Tuple *data = *(input->data + i);
        int hashedKey = hash_key(data->partitionKey);
        int *offset = (int*)calloc(NUM_PARTITIONS, sizeof(int));
        *(*(input->partitions + hashedKey) + *(offset + hashedKey)) = data;
        (*(offset + hashedKey))++;
    }
}