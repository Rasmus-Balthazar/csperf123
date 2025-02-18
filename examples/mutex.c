#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>


pthread_mutex_t mut; // Declare a mutex
int counter = 0;    //  Declare a counter

void *inc()         // In C++, we also passed a function by pointer, now its just explicit
{
    for (int i = 0; i < 1000000; i++)
    {
        pthread_mutex_lock(&mut);
        counter++;
        pthread_mutex_unlock(&mut);
    }
}

int main(int argc, char *argv[])
{
    pthread_mutex_init(&mut, NULL); // Initialize the mutex

    if (argc != 2) 
    {
        printf("Please enter the number of threads\n");
        return 1;
    }

    
    int num_threads = atoi(argv[1]);
    
    printf("Running with %d threads\n", num_threads);

    pthread_t threads[num_threads];
    
    for (int i = 0; i < num_threads; i++)
    {
        // Recall that we need a pointer, and the pthreads are stored in an array, so reference the threads[i]
        pthread_create(&threads[i], NULL, inc, NULL);
    }

    for (int i = 0; i < num_threads; i++)
    {
        // This will make all threads wait for work to be completed
        pthread_join(threads[i], NULL);
    }

    printf("The counter is now %d\n", counter);
    return 0;
}
