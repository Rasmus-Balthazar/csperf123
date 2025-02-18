#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

atomic_int counter = 0;
// int counter = 0; // You can try using this regular int, to see that the counter result is clearly the result of race conditions

void *inc(void *args)
{
    int myCoolArg = *(int *) args; // Just here to show how to get an argument
    printf("My arg was: %i\n", myCoolArg);
    for (int i = 0; i < 1000000; i++) 
    {
        counter++;
        // The following line is more C-like syntax, and there are more atomic operations like it, you might find useful
        // atomic_fetch_add(&counter, 1);
    }
};

int main(int argc, char *argv[])
{
    if (argc != 2) 
    {
        printf("Please enter the number of threads\n");
        return 1;
    }

    
    int num_threads = atoi(argv[1]);
    
    printf("Running with %d threads\n", num_threads);

    pthread_t threads[num_threads];
    int args[] = {41,42,43,44};

    for (int i = 0; i < num_threads; i++)
    {
        pthread_create(&threads[i], NULL, inc, args+i); // If no args, set last argument to NULL
        // Note that NULL is the pthread policy, it can be used e.g. to set core affinity
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    printf("The counter is now %d\n", counter);
    return 0;
}
