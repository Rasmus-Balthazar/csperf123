// #include "independent.h"
#include "count_then_move.h"

#define NUM_THREADS 4
#define KEY_BITS 10
#define PARTITION_SIZE 16
#define NUM_PARTITIONS (2 << KEY_BITS)
#define DATA_PER_PARTITION (NUM_PARTITIONS * PARTITION_SIZE)
#define SAMPLE_SIZE (DATA_PER_PARTITION * NUM_THREADS)

int main() {
// int main(int argc, char **argv) {
    printf("hi");
    Tuple **data = gen_data(SAMPLE_SIZE);
    /* Tuple ****independently_partitioned = partition_independent(SAMPLE_SIZE, data, NUM_THREADS, NUM_PARTITIONS, PARTITION_SIZE); */
    Tuple ***count_then_move = count_then_move_partition(SAMPLE_SIZE, data, NUM_THREADS, NUM_PARTITIONS);
    for (int i = 0; i < 32; i++)
    {
        printf("%lu\n", (*(*(count_then_move)+i))->partitionKey);
        // printf("%lu\n", (*(*(*(independently_partitioned))+i))->partitionKey);
    }
    return 0;
}
