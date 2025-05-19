#!/bin/bash

g++ main.cpp 

inputs=("../Homemade_datasets/patterns.regex" "../Homemade_datasets/patterns_literal.regex")

for input in "${inputs[@]}"; do
    input_name=$(basename "$input" .regex)
    
    for iteration in {1..5}; do
        output_file="output_${input_name}_iteration_${iteration}.txt"
        output_perf_file="output_perf_${input_name}_iteration_${iteration}.txt"

        perf stat -o "$output_perf_file" -e cycles,instructions ./a.out "../Homemade_datasets/dataset_10mb.txt" "$input" > "$output_file"
 
    done
done

echo "All runs completed."