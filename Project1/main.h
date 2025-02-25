#include <string.h>
#include "independent.h"
#include "count_then_move.h"

#ifndef MAIN_H
#define MAIN_H


Tuple ****run_independent(Tuple **data, uint64_t sample_size, int num_threads, int num_partitions, int partition_size);
Tuple ***run_ctm(Tuple **data, uint64_t sample_size, int num_threads, int num_partitions);
void print_partition(Tuple **partition, int partition_size);
#endif