#include "main.h"
int main(int argc, char **argv) {
    // begin reading arguments!
    if(argc < 4)
    {
        printf("Incorrect amount of arguments, expected 3 and got %d\nIn the format:\nmain.exe <num_threads> <key_bits> <partition_size>\n", argc-1);
        return 1;
    }
    int algorithm = 0;
    int debug = 0;
    int num_threads = -1;           //atoi(argv[1]);
    int key_bits = -1;              //atoi(argv[2]);
    int data_bits = -1;        //atoi(argv[3]);
    //FIXME in partitioning we need that the last thread takes the rest and not just the defined amount
    for (int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-a") == 0)
        {
            if(strcmp(argv[i+1], "c") == 0 || strcmp(argv[i+1], "ctm"))
            {
                algorithm = 1;
            } else if (strcmp(argv[i+1], "i") == 0 || strcmp(argv[i+1], "independent"))
            {
                algorithm = 2;
            }
            i++; 
        } else if (strcmp(argv[i], "-d") == 0)
        {
            debug = 1;
        } else
        {
            if (num_threads == -1)
            {
                num_threads = atoi(argv[i]);
            }
            else if (key_bits == -1)
            {
                key_bits = atoi(argv[i]);
            }
            else if (data_bits == -1)
            {
                data_bits = atoi(argv[i]);
            }
        }
    }

    printf("starting\n");

    int num_partitions = 1 << key_bits;
    uint64_t sample_size = (1llu << data_bits);
    Tuple **data = gen_data(sample_size);

    
    // Tuple ***count_then_move = run_ctm(data);
    if(algorithm == 1)
    {
        Tuple ****independently_partitioned = run_independent(data, sample_size, num_threads, num_partitions, data_bits);
        print_partition(independently_partitioned[0][0], data_bits-key_bits);
        return 0;
    } else if (algorithm == 2)
    {

        printf("running algorithm\n");
        Tuple ***ctm_partitioned = run_ctm(data, sample_size, num_threads, num_partitions);
        printf("algorithm run\n");
        print_partition(*(ctm_partitioned), data_bits-key_bits);
        return 0;
    }
    return 0;
}

Tuple ****run_independent(Tuple **data, uint64_t sample_size, int num_threads, int num_partitions, int partition_size) {
    Tuple ****independently_partitioned = partition_independent(sample_size, data, num_threads, num_partitions, partition_size);
    return independently_partitioned;
}

Tuple ***run_ctm(Tuple **data, uint64_t sample_size, int num_threads, int num_partitions) {
    Tuple ***count_then_move = count_then_move_partition(sample_size, data, num_threads, num_partitions);
    return count_then_move;
}

void print_partition(Tuple **partition, int data_bits) {
    for (int i = 0; i < (1 << data_bits); i++)
    {
        printf("%llu\n", partition[i]->partitionKey);
    }
}