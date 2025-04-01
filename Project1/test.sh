#!/bin/bash

tuplespower=24
for threads in {16,32}; do
    for keybits in {1..18}; do
    # echo $keybits
    # for threads in 2; do
        ./main.exe $threads $keybits $tuplespower -a ctm >> ./out/data_ctm.txt
        # ./main.exe $threads $keybits $tuplespower -a i >> data_independent.txt
    done
done