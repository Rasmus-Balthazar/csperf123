#include <pthread.h>
#include "data_gen.h"
#ifndef CTM_H
#define CTM_H

typedef struct {
    Tuple **data;
    int **offsets;
    int startIndex;
    int endIndex;
    int threadNum;
    int partitionCount;
} CountArgs;

typedef struct {
    Tuple **data;
    int **offsets;
    int startIndex;
    int endIndex;
    int threadNum;
    int partitionCount;
    Tuple ***output;
} MoveArgs;

int hash(uint64_t key, int num_partitions);
void *count(void *_args);
void *move(void *_args);
Tuple ***count_then_move_partition(uint64_t sample_size, Tuple **data, int num_threads, int num_partitions); 

#endif