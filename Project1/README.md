# csperf123
aight, so the questions in question is that we are working with the algorithms:
3.1 Independant output


3.4 Count-then-move


## Roadmap
- [x] Setup data to be used for partitioning
- [x] Make data accessible for usage
- [x] Setup threading for algorithms
- [x] Implement Independant ouput
- [x] Implement count-then-move
- [x] setup measurements
- [x] take measurements
- [x] visualize data

different types of algorithms:
- Independant output: i
- Count-then-move: c


But how do we run it?
Move to Project1 folder
./main.exe <number_of_threads> <key_bits> <partition_size> -a <algorithm> <debug_flag: -d>

./main.exe 1 2 4 -a i


in other words:
define the number of threads (mandatory)
define the number of key bits (mandatory)
define the number of partition size (mandatory)
Set the algorithm flag if we want something else than a dry run (optional)
    If we set the algorithm flag, we need to define the algorithm we want to use (mandatory)
Set the debug flag if we want to see the output (optional)


## Data
Running the script run.sh will generate the following folders: ctm, dry and indepebnant.
ctm contaisn perforamnce metrics for running the count-then-move algorithm with our defined
parameters. independent contains the same metrics but for the independant output algorithm.
dry contains the same metrics but for a dry run to give us the time for just generating the data.

This data will be placed in a folder called 'out' in the root of the Project1 folder.

## Visualizing the data
We can visualize the data by running our python script graph_creation.py without any commands. 
It is important you ensure that you have your own data in the out folder before running the 
script if you want you own data visualized.