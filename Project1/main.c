#include "main.h"
int main(int argc, char **argv) {
    // begin reading arguments!
    if(argc < 4)
    {
        printf("Incorrect amount of arguments, expected 3 and got %d\nIn the format:\nmain.exe <num_threads> <key_bits> <partition_size>\n", argc-1);
        return 1;
    }
    //FIXME in partitioning we need that the last thread takes the rest and not just the defined amount
    int num_threads = atoi(argv[1]);
    int key_bits = atoi(argv[2]);
    int partition_size = atoi(argv[3]);
    int num_partitions = 1 << key_bits;
    uint64_t sample_size = ((uint64_t)num_partitions * partition_size);
    Tuple **data = gen_data(sample_size);
    
    // Tuple ***count_then_move = run_ctm(data);
    if(strcmp(argv[4], "-i") == 0)
    {
        Tuple ****independently_partitioned = run_independent(data, sample_size, num_threads, num_partitions, partition_size);
        print_partition(independently_partitioned[0][0], partition_size);
        return 0;
    }
    if(strcmp(argv[4], "-c") == 0)
    {

        Tuple ***ctm_partitioned = run_ctm(data, sample_size, num_threads, num_partitions);
        print_partition(*(ctm_partitioned), partition_size);
        return 0;
    }
    if(strcmp(argv[4], "-d") == 0)
    {
        return 0;
    }
    return 1;
}

Tuple ****run_independent(Tuple **data, uint64_t sample_size, int num_threads, int num_partitions, int partition_size) {
    Tuple ****independently_partitioned = partition_independent(sample_size, data, num_threads, num_partitions, partition_size);
    return independently_partitioned;
}

Tuple ***run_ctm(Tuple **data, uint64_t sample_size, int num_threads, int num_partitions) {
    Tuple ***count_then_move = count_then_move_partition(sample_size, data, num_threads, num_partitions);
    return count_then_move;
}

void print_partition(Tuple **partition, int partition_size) {
    for (int i = 0; i < partition_size; i++)
    {
        printf("%llu\n", partition[i]->partitionKey);
    }
}