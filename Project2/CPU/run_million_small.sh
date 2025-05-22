#!/bin/bash

g++ main.cpp 

inputs=("../Homemade_datasets/patterns_small.regex" "../Homemade_datasets/patterns_literal_small.regex")

for input in "${inputs[@]}"; do
    input_name=$(basename "$input" .regex)
    
    for iteration in {1}; do
        output_file="perf_${input_name}_million_small.txt"
        output_perf_file="perf_${input_name}_million_small.txt"

        perf stat -o "$output_perf_file" -e cycles,instructions ./a.out "../Homemade_datasets/dataset_million_lines.txt" "$input" > "$output_file"
 
    done
done

echo "All runs completed."