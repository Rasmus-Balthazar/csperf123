#include "independent.h"


Tuple ****partition_independent(int data_size, Tuple **data, int num_threads, int num_partitions, int partition_size){
    // Tuple **data = gen_data(SAMPLE_SIZE);
    Tuple ****buffers = (Tuple****)calloc(num_threads, sizeof(Tuple***));
    for (int i = 0; i < num_threads; i++) {
        buffers[i] = (Tuple***)calloc(num_partitions, sizeof(Tuple**));
    }

    pthread_t threads[num_threads];
    // int args[] = {41,42,43,44};

    Args *argsArr = (Args*)calloc(num_threads, sizeof(Args));
    for (int i = 0; i < num_threads; i++)
    {
        Args *args = argsArr+i;
        args->startIndex = (data_size / num_threads) * i; 
        args->endIndex = (data_size / num_threads) * (i+1);
        if (args->endIndex >= data_size)
            args->endIndex = data_size - 1;
        args->data = data;
        args->partitions = buffers[i];
        args->numPartitions = num_partitions;
        args->partitionSize = partition_size;
        args->start = (struct timespec*)malloc(sizeof(struct timespec));
        args->end = (struct timespec*)malloc(sizeof(struct timespec));

        pthread_create(&threads[i], NULL, run, args); // If no args, set last argument to NULL
        // Note that NULL is the pthread policy, it can be used e.g. to set core affinity
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    unsigned long min_time = -1;
    unsigned long max_time = 0;
    unsigned long avg_time = 0;

    for (int i = 0; i < num_threads; i++)
    {
        long thread_time = (argsArr[i].end->tv_sec - argsArr[i].start->tv_sec) * 1000 + (argsArr[i].end->tv_nsec - argsArr[i].start->tv_nsec) / 1000000;
        if (thread_time < min_time)
            min_time = thread_time;
        if (max_time < thread_time)
            max_time = thread_time;
        avg_time += thread_time;
    }
    avg_time /= num_threads;

    printf("Total max: %lu\n", max_time);
    printf("Total min: %lu\n", min_time);
    printf("Total avg: %lu\n", avg_time);

    return buffers;
}
//method hash and find where it belongs
int hash_key(uint64_t key, int num_partitions) {
    //we know amount of partitions, so get hashing
    return key % num_partitions;
}

void *run(void *args) {
    Args *input = (Args *)args;
    Tuple **local_array = (Tuple**)calloc((input->partitionSize * input->numPartitions) << 1, sizeof(Tuple*));
    for (int i = 0; i < input->numPartitions; i++)
    {
        *(input->partitions+i) = &local_array[2*i*input->partitionSize];
        //Allocate every partition from null to a partion of the correct size.
    }
    int *offsets = (int*)calloc(input->numPartitions, sizeof(int));
    
    // THIS WILL BE ANGRY
    // its ok :D
    clock_gettime(CLOCK_MONOTONIC_RAW, input->start);
    for (int i = input->startIndex; i < input->endIndex; i++) {
        Tuple *data = input->data[i];
        int hashedKey = hash_key(data->partitionKey, input->numPartitions);
        local_array[(2*hashedKey*input->partitionSize)+offsets[hashedKey]] = data;
        offsets[hashedKey]++;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, input->end);
    return 0;
}
