#include <pthread.h>
#include "data_gen.h"

#ifndef INDE_H
#define INDE_H

typedef struct {
    int startIndex;
    int endIndex;
    Tuple** data;
    Tuple*** partitions;

    int numPartitions;
    int partitionSize;
} Args;

int hash_key(uint64_t key, int num_partitions);
void *run(void *args);
Tuple ****partition_independent(int data_size, Tuple **data, int num_threads, int num_partitions, int partition_size);

#endif