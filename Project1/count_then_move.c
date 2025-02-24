#include "count_then_move.h"

Tuple ***count_then_move_partition(uint64_t sample_size, Tuple **data, int num_threads, int num_partitions) {
    int **offsets = (int**)calloc(num_partitions, sizeof(int*));
    for (int i = 0; i < num_partitions; i++)
    {
        offsets[i] = (int*)calloc(num_threads, sizeof(int));
    }
    

    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; i++)
    {
        CountArgs *args = (CountArgs*)malloc(sizeof(CountArgs));
        args->data = data;
        args->offsets = offsets;
        args->startIndex = (sample_size / num_threads) * i; 
        args->endIndex = (sample_size / num_threads) * (i+1);
        args->threadNum = i;
        args->partitionCount = num_partitions;
        pthread_create(&threads[i], NULL, count, args); // If no args, set last argument to NULL
    }
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    Tuple ***output = (Tuple***)calloc(num_partitions, sizeof(Tuple**));
    for (int i = 0; i < num_partitions; i++)
    {
        int partition_size = 0;
        for (int j = 0; j < num_threads; j++)
        {
            partition_size += offsets[i][j];
        }
        
        *(output+i) = (Tuple**)calloc(partition_size, sizeof(Tuple*));
    }
    
    for (int i = 0; i < num_threads; i++)
    {
        MoveArgs *args = (MoveArgs*)malloc(sizeof(MoveArgs)); // Could reduce duplicate work by reusing args from Count
        args->data = data;
        args->offsets = offsets;
        args->startIndex = (sample_size / num_threads) * i; 
        args->endIndex = (sample_size / num_threads) * (i+1);
        args->threadNum = i;
        args->partitionCount = num_partitions;
        args->output = output;
        pthread_create(&threads[i], NULL, move, args);
    }
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    return output;
}

int hash(uint64_t key, int num_partitions) {
    return key % num_partitions;
}


void *count(void *_args) {
    CountArgs *args = (CountArgs*)_args;
    for (int i = args->startIndex; i < args->endIndex; i++)
    {
        int partition = hash(args->data[i]->partitionKey, args->partitionCount);
        args->offsets[partition][args->threadNum]++;
    }
    return 0;
}


void *move(void *_args) {
    MoveArgs *args = (MoveArgs*)_args;
    int *offsets = calloc(args->partitionCount, sizeof(int));
    for (int i = 0; i < args->partitionCount; i++)
        for (int j = 0; j < args->threadNum; j++)
            offsets[i] += args->offsets[i][j];
    for(int i = args->startIndex; i < args->endIndex; i++) 
    {   
        int hashed_key = hash(args->data[i]->partitionKey, args->partitionCount);
        args->output[hashed_key][offsets[hashed_key]] = args->data[i];
        offsets[hashed_key]++;
    }
    return 0;
}
