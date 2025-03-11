#include "count_then_move.h"

Tuple ***count_then_move_partition(uint64_t sample_size, Tuple **data, int num_threads, int num_partitions) {
    int **offsets = (int**)calloc(num_partitions, sizeof(int*));
    for (int i = 0; i < num_partitions; i++)
    {
        offsets[i] = (int*)calloc(num_threads, sizeof(int));
    }
    

    pthread_t threads[num_threads];
    CountArgs *countArgsArray = (CountArgs*)calloc(num_threads,sizeof(CountArgs));
    for (int i = 0; i < num_threads; i++)
    {
        CountArgs *args = countArgsArray + i;
        args->data = data;
        args->offsets = offsets;
        args->startIndex = (sample_size / num_threads) * i; 
        args->endIndex = (sample_size / num_threads) * (i+1);
        if (args->endIndex >= sample_size)
        args->endIndex = sample_size - 1;
        args->threadNum = i;
        args->partitionCount = num_partitions;
        args->start = (struct timespec*)malloc(sizeof(struct timespec));
        args->end = (struct timespec*)malloc(sizeof(struct timespec));
        pthread_create(&threads[i], NULL, count, args);
        //Maybe print something here
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
    
    MoveArgs *moveArgsArray = (MoveArgs*)calloc(num_threads, sizeof(MoveArgs)); // Could reduce duplicate work by reusing args from Count
    for (int i = 0; i < num_threads; i++)
    {
        MoveArgs *args = moveArgsArray + i; 
        args->data = data;
        args->offsets = offsets;
        args->startIndex = (sample_size / num_threads) * i; 
        args->endIndex = (sample_size / num_threads) * (i+1);
        if (args->endIndex >= sample_size)
            args->endIndex = sample_size - 1;
        args->threadNum = i;
        args->partitionCount = num_partitions;
        args->output = output;
        args->start = (struct timespec*)malloc(sizeof(struct timespec));
        args->end = (struct timespec*)malloc(sizeof(struct timespec));
        pthread_create(&threads[i], NULL, move, args);
        //maybe print something here
    }
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    //collect data from moveArgsArray
    long count_max = 0;
    long count_min = __LONG_MAX__;
    long count_avg = 0;
    long move_max =  0;
    long move_min =  __LONG_MAX__;
    long move_avg =  0;
    long total_max = 0;
    long total_min = 0;
    long total_avg = 0;
    for (int i = 0; i < num_threads; i++)
    {
        //get count max, min, avg of each thread
        CountArgs countArgs = countArgsArray[i];
        MoveArgs moveArgs = moveArgsArray[i];
        long count_time = countArgs.end->tv_nsec - countArgs.start->tv_nsec;
        long move_time = moveArgs.end->tv_nsec - moveArgs.start->tv_nsec;
        if(count_time > count_max)
            count_max = count_time;
        if(count_time < count_min)
            count_min = count_time;
        count_avg += countArgs.end->tv_nsec;

        if (move_time > move_max)
            move_max = move_time;
        if (move_time < move_min)
            move_min = move_time;
        move_avg += move_time;

        if (moveArgs.end->tv_nsec > move_max)
            total_max = move_max + count_max;
        if (moveArgs.end->tv_nsec < move_min)
            total_min = move_min + count_min;
    }
    count_avg /= num_threads;
    move_avg /= num_threads;
    total_avg = count_avg + move_avg;

    long count_max_time = (count_max) * 1000 + (count_max) / 1000000;
    long move_max_time = (move_max) * 1000 + (move_max) / 1000000;
    long count_min_time = (count_min) * 1000 + (count_min) / 1000000;
    long move_min_time = (move_min) * 1000 + (move_min) / 1000000;
    long count_avg_time = (count_avg) * 1000 + (count_avg) / 1000000;
    long move_avg_time = (move_avg) * 1000 + (move_avg) / 1000000;
    long total_max_time = (total_max) * 1000 + (total_max) / 1000000;
    long total_min_time = (total_min) * 1000 + (total_min) / 1000000;
    long total_avg_time = (total_avg) * 1000 + (total_avg) / 1000000;

    printf("Count max: %ld\n", count_max_time);
    printf("Count min: %ld\n", count_min_time);
    printf("Count avg: %ld\n", count_avg_time);
    printf("Move max: %ld\n", move_max_time);
    printf("Move min: %ld\n", move_min_time);
    printf("Move avg: %ld\n", move_avg_time);
    printf("Total max: %ld\n", total_max_time);
    printf("Total min: %ld\n", total_min_time);
    printf("Total avg: %ld\n", total_avg_time);
    return output;
}

int hash(uint64_t key, int num_partitions) {
    return key % num_partitions;
}


void *count(void *_args) {
    CountArgs *args = (CountArgs*)_args;
    clock_gettime(CLOCK_MONOTONIC_RAW, args->start); //this breaks
    for (uint64_t i = args->startIndex; i < args->endIndex; i++)
    {
        int partition = hash(args->data[i]->partitionKey, args->partitionCount);
        args->offsets[partition][args->threadNum]++;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, args->end);
    return 0;
}


void *move(void *_args) {
    MoveArgs *args = (MoveArgs*)_args;
    int *offsets = calloc(args->partitionCount, sizeof(int));
    clock_gettime(CLOCK_MONOTONIC_RAW, args->start);
    for (int i = 0; i < args->partitionCount; i++)
        for (int j = 0; j < args->threadNum; j++)
            offsets[i] += args->offsets[i][j];
    for(int i = args->startIndex; i < args->endIndex; i++) 
    {   
        int hashed_key = hash(args->data[i]->partitionKey, args->partitionCount);
        args->output[hashed_key][offsets[hashed_key]] = args->data[i];
        offsets[hashed_key]++;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, args->end);
    return 0;
}
