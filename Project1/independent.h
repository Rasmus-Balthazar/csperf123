#include <pthread.h>
#include "data_gen.h"

typedef struct {
    int startIndex;
    int endIndex;
    Tuple** data;
    Tuple*** partitions;
} Args;

int hash_key(uint64_t key);
void *run(void *args);