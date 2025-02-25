#include "independent.h"
#include "count_then_move.h"

#define NUM_THREADS 4
#define KEY_BITS 10
#define PARTITION_SIZE 16
#define NUM_PARTITIONS (2 << KEY_BITS)
#define DATA_PER_PARTITION (NUM_PARTITIONS * PARTITION_SIZE)
#define SAMPLE_SIZE (DATA_PER_PARTITION * NUM_THREADS)

Tuple ****run_independent(Tuple **data);
Tuple ***run_ctm(Tuple **data);
void print_partition(Tuple **partition, int partition_size);

int main(int argc, char **argv) {
    Tuple **data = gen_data(SAMPLE_SIZE);

    // Tuple ***count_then_move = run_ctm(data);
    Tuple ****independently_partitioned = run_independent(data);

    print_partition(independently_partitioned[0][0], 16);

    return 0;
}

Tuple ****run_independent(Tuple **data) {
    Tuple ****independently_partitioned = partition_independent(SAMPLE_SIZE, data, NUM_THREADS, NUM_PARTITIONS, PARTITION_SIZE);
    return independently_partitioned;
}

Tuple ***run_ctm(Tuple **data) {
    Tuple ***count_then_move = count_then_move_partition(SAMPLE_SIZE, data, NUM_THREADS, NUM_PARTITIONS);
    return count_then_move;
}

void print_partition(Tuple **partition, int partition_size) {
    for (int i = 0; i < partition_size; i++)
    {
        printf("%lu\n", partition[i]->partitionKey);
    }
}