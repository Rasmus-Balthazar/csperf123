#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
// #include <Windows.h> // Sleep function for windows

int main()
{
    // method 1: don't do this
    // only second precision, and is affected by NTP
    time_t before = time(NULL);
    // Sleep(2500); // for windows
    usleep(2500000);
    time_t after = time(NULL);
    time_t elapsed = difftime(after, before);
    printf("Elapsed: %lu ms\n", elapsed * 1000);

    // method 2: don't do this either
    // still affected by NTP and system clock
    struct timeval stop, begin;
    gettimeofday(&begin, NULL);
    // Sleep(2500); // for windows
    usleep(2500000);
    gettimeofday(&stop, NULL);
    printf("Elapsed: %lu ms\n", (stop.tv_sec - begin.tv_sec) * 1000 + (stop.tv_usec - begin.tv_usec) / 1000);

    // method 3: do this or use time from perf
    // choose correct type of clock
    // https://www.man7.org/linux/man-pages/man3/clock_gettime.3.html
    // should not be affected by NTP, or count across threads
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    // Sleep(2500); // for windows
    usleep(2500000);
    clock_gettime(CLOCK_MONOTONIC_RAW, &finish);
    // need to sum seconds and nanoseconds
    long elapsed_time_ms = (finish.tv_sec - start.tv_sec) * 1000 + (finish.tv_nsec - start.tv_nsec) / 1000000;
    printf("Elapsed: %lu ms\n", elapsed_time_ms);
    return 0;
}
