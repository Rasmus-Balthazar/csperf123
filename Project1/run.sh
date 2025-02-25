#!/bin/bash

# run both algorithms

# example setups:
perf stat -e cycles,instructions,L1-icache-load-misses,L1-dcache-load-misses, \
LLC-load-misses,cache-misses,uops_retired.stall_cycles, \
branch-misses,iTLB-load-misses,dTLB-load-misses ./main.exe 1 1 1 -d

perf stat -e cycles,instructions,L1-icache-load-misses,L1-dcache-load-misses, \
LLC-load-misses,cache-misses,uops_retired.stall_cycles, \
branch-misses,iTLB-load-misses,dTLB-load-misses ./main.exe 1 1 1 -c

perf stat -e cycles,instructions,L1-icache-load-misses,L1-dcache-load-misses, \
LLC-load-misses,cache-misses,uops_retired.stall_cycles, \
branch-misses,iTLB-load-misses,dTLB-load-misses ./main.exe 1 1 1 -i



# loop over num threads, data sizes
perf stat -e cycles,instructions,L1-icache-load-misses,L1-dcache-load-misses, \
LLC-load-misses,cache-misses,uops_retired.stall_cycles, \
branch-misses,iTLB-load-misses,dTLB-load-misses ./main.exe 4 8 16 -i