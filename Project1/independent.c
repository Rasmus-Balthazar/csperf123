#include "independent.h"


Tuple ****partition_independent(int data_size, Tuple **data, int num_threads, int num_partitions, int partition_size){
    // Tuple **data = gen_data(SAMPLE_SIZE);
    Tuple ****buffers = (Tuple****)calloc(num_threads, sizeof(Tuple***));
    for (int i = 0; i < num_threads; i++) {
        *(buffers + i) = (Tuple***)calloc(num_partitions, sizeof(Tuple**));
    }

    pthread_t threads[num_threads];
    // int args[] = {41,42,43,44};

    for (int i = 0; i < num_threads; i++)
    {
        Args *args = (Args*)malloc(sizeof(Args));
        args->startIndex = (data_size / num_threads) * i; 
        args->endIndex = (data_size / num_threads) * (i+1);
        if (args->endIndex >= data_size)
            args->endIndex = data_size - 1;
        args->data = data;
        args->partitions = *(buffers + i);
        args->numPartitions = num_partitions;
        args->partitionSize = partition_size;

        pthread_create(&threads[i], NULL, run, args); // If no args, set last argument to NULL
        // Note that NULL is the pthread policy, it can be used e.g. to set core affinity
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    return buffers;
}
//method hash and find where it belongs
int hash_key(uint64_t key, int num_partitions) {
    //we know amount of partitions, so get hashing
    return (int)(key % num_partitions);
}

void *run(void *args) {
    Args *input = (Args *)args;
    for(int i =  0; i < input->numPartitions; i++) {
        //Allocate every partition from null to a partion of the correct size.
        *(input->partitions+i) = (Tuple**)calloc(input->partitionSize << 1, sizeof(Tuple*));
    }
    int *offset = (int*)calloc(input->numPartitions, sizeof(int));
    for (int i = input->startIndex; i < input->endIndex; i++) {
        Tuple *data = *(input->data + i);
        int hashedKey = hash_key(data->partitionKey, input->numPartitions);
        *(*(input->partitions + hashedKey) + *(offset + hashedKey)) = data;
        (*(offset + hashedKey))++;
    }
    return 0;
}
