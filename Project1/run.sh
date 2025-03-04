#!/bin/bash

# run both algorithms

# example setups:
# perf stat -e cycles,instructions,L1-icache-load-misses,L1-dcache-load-misses, \
# LLC-load-misses,cache-misses,uops_retired.stall_cycles, \
# branch-misses,iTLB-load-misses,dTLB-load-misses ./main.exe 1 1 1 -d

# perf stat -e cycles,instructions,L1-icache-load-misses,L1-dcache-load-misses, \
# LLC-load-misses,cache-misses,uops_retired.stall_cycles, \
# branch-misses,iTLB-load-misses,dTLB-load-misses ./main.exe 1 1 1 -c

# perf stat -e cycles,instructions,L1-icache-load-misses,L1-dcache-load-misses, \
# LLC-load-misses,cache-misses,uops_retired.stall_cycles, \
# branch-misses,iTLB-load-misses,dTLB-load-misses -o ./out/perf_dry.txt ./main.exe 1 1 1 -i



# loop over num threads, data sizes
# perf stat -e cycles,instructions,L1-icache-load-misses,L1-dcache-load-misses, \
# LLC-load-misses,cache-misses,uops_retired.stall_cycles, \
# branch-misses,iTLB-load-misses,dTLB-load-misses ./main.exe 4 8 16 -i

tuplespower=24

for keybits in {1..18}; do
    # echo $keybits
    for threads in {1,2,4,8,16,32}; do
    # for threads in 2; do
        for i in {1..10}; do
            perf stat -e \
            branch-misses,iTLB-load-misses,dTLB-load-misses -o "./out/ctm/perf_ctm-${keybits}-${threads}-${i}.txt" ./main.exe $threads $keybits $tuplespower -a ctm -i $i >> ./out/data_ctm.txt
            # ./main.exe $threads $keybits $tuplespower -a ctm >> data_ctm.txt
        done
        for i in {1..10}; do
            perf stat -e  \
            branch-misses,iTLB-load-misses,dTLB-load-misses -o "./out/independent/perf_independent-${keybits}-${threads}-${i}.txt" ./main.exe $threads $keybits $tuplespower -a i -i $i >> ./out/data_independent.txt
            # ./main.exe $threads $keybits $tuplespower -a i >> data_independent.txt
        done
    done
done
for i in {1..100}; do
    perf stat -e  \
    branch-misses,iTLB-load-misses,dTLB-load-misses -o "./out/dry/perf_dry-${i}.txt" ./main.exe 1 1 $tuplespower -i $i >> ./out/data_dry.txt
    # ./main.exe $threads $keybits $tuplespower >> data_dry.txt
done