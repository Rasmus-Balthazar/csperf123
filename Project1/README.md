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
- [ ] setup measurements
- [ ] take measurements

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