#include <pthread.h>
#include "data_gen.h"

typedef struct {
    int startIndex;
    int endIndex;
    Tuple** data;
    Tuple*** partitions;

    int numPartitions;
    int partitionSize;
} Args;

int hash_key(uint64_t key);
void *run(void *args);
Tuple ****patition_independent(int data_size, Tuple **data, int num_threads, int num_partitions, int partition_size);