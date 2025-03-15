from collections import defaultdict
import os
import re
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

# === Where to find data ===
data_ctm = "out/data_ctm.txt" #file for stdout for cmt
data_independent = "out/data_independent.txt" #file for stdout for independent
data_dry = "out/data_dry.txt" #file for stdout for data generation

ctm_perf_dir = "out/ctm/"
independent_perf_dir = "out/independent/"
dry_perf_dir = "out/dry/"

# ==== Data Structures ====
#iterations = 10 

@dataclass
class Results:
    algo: str
    threads: int
    hashbits: int
    tuplepower: int
    time: list[float]

    def time_average(self):
        return sum(self.time) / len(self.time)


# ==== regex matchers ======
input_perm_matcher = re.compile(r"(Performance counter stats for '/home/group123/Project1/main\.exe )([\d]+).([\d]+).([\d]+)", re.MULTILINE)
time_matcher = re.compile(r"([0-9]*\.?[0-9]*)( seconds time elapsed)", re.MULTILINE)

ctm_stdout_matcher = re.compile(
    r"Count max: (\d+)\s*"                # Count max [0]
    r"Count min: (\d+)\s*"                # Count min [1]
    r"Count avg: (\d+)\s*"                # Count avg [2]
    r"Move max: (\d+)\s*"                 # Move max [3]
    r"Move min: (\d+)\s*"                 # Move min [4]
    r"Move avg: (\d+)\s*"                 # Move avg [5]
    r"Total max: (\d+)\s*"                # Total max [6]
    r"Total min: (\d+)\s*"                # Total min [7]
    r"Total avg: (\d+)\s*"                # Total avg [8]
    r"Iteration: (\d+)\s*"                # Iteration number [9]
    r"Algorithm: (\d+)\s*"                # Algorithm number [10]
    r"Num threads: (\d+)\s*"              # Number of threads [11]
    r"Num key bits: (\d+)\s*"             # Number of key bits/hash bits [12]
    r"Time to generate data: (\d+)\s*"    # Time to generate data [13]
    r"Time to partition in ms: (\d+)"     # Time to partition [14]
    , re.MULTILINE
)

independent_stdout_matcher = re.compile(
    r"Max time: (\d+)\s*"                 # Max time [0]
    r"Min time: (\d+)\s*"                 # Min time [1]
    r"Avg time: (\d+)\s*"                 # Avg time [2]
    r"Iteration: (\d+)\s*"                # Iteration number [3]
    r"Algorithm: (\d+)\s*"                # Algorithm number [4]
    r"Num threads: (\d+)\s*"              # Number of threads [5]
    r"Num key bits: (\d+)\s*"             # Number of key bits/hash bits [6]
    r"Time to generate data: (\d+)\s*"    # Time to generate data [7]
    r"Time to partition in ms: (\d+)"     # Time to partition [8]
    , re.MULTILINE
)

dry_stdout_matcher = re.compile(
    r"Iteration: (\d+)\s*"                # Iteration number [0]
    r"Algorithm: (\d+)\s*"                # Algorithm number [1]
    r"Num threads: (\d+)\s*"              # Number of threads [2]
    r"Num key bits: (\d+)\s*"             # Number of key bits/hash bits [3]
    r"Time to generate data: (\d+)\s*"    # Time to generate data [4]
    r"Time to partition in ms: (\d+)"     # Time to partition [5] always 0, it is not a partitioning algo
    , re.MULTILINE
)

def getPerfResultsFromDirS(algos, perf_dirs):
    results_dict: dict[str, dict[int, dict[int, Results]]] = defaultdict(lambda: defaultdict(lambda: dict()))

    for i in range(len(algos)):
        algo = algos[i]
        path = perf_dirs[i]
        folder = os.fsencode(path)
        for file in os.listdir(folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                with open(os.path.join(path, filename)) as myfile:
                    content = myfile.read()

                    #Regex matcher one, TODO make nicer
                    in_params = input_perm_matcher.findall(content)[0]
                    threads  = int(in_params[1])
                    hashbits = int(in_params[2])
                    tuplepower = int(in_params[3])

                    #Regex matcher two, TODO make nicer 
                    time = float(time_matcher.findall(content)[0][0])
                    
                    #print(f'filename: {filename} threads: {in_params[1]}, keybits: {in_params[2]}, power: {in_params[3]} time: {time}')
                    if hashbits not in results_dict[algo][threads]:
                        r = Results(algo, threads, hashbits, tuplepower, [time])
                        results_dict[algo][threads][hashbits] = r
                    else:
                        results_dict[algo][threads][hashbits].time.append(time)
    
    print(results_dict["ctm"][1][1].time_average())
    return results_dict


def getDataFileResults(algos, files, tuplepower):
    results_dict: dict[str, dict[int, dict[int, Results]]] = defaultdict(lambda: defaultdict(lambda: dict()))
    
    for i in range(len(algos)):
        algo = algos[i]
        path = files[i]
        content = open(path, "r").read()
        sections = content.split("-------------------")
        for s in sections:
            # ===== CTM =====
            if algo == "ctm":
                observations = ctm_stdout_matcher.findall(s)
                if len(observations) == 0:
                    continue
                if int(observations[0][9]) == 10:
                    threads = int(observations[0][11])
                    hashbits = int(observations[0][12])
                    time = int(observations[0][8])/1000 # To get seconds
                    r = Results(algo, threads, hashbits, tuplepower, [time])
                    results_dict[algo][threads][hashbits] = r
            # ===== Independent =====
            if algo == "i":
                observations = independent_stdout_matcher.findall(s)
                if len(observations) == 0:
                    continue
                if int(observations[0][3]) == 10:
                    threads = int(observations[0][5])
                    hashbits = int(observations[0][6])
                    time = int(observations[0][2])/1000 # To get seconds
                    r = Results(algo, threads, hashbits, tuplepower, [time])
                    results_dict[algo][threads][hashbits] = r
            # ===== Dry =====
            if algo == "dry":
                observations = dry_stdout_matcher.findall(s)
                if len(observations) == 0:
                    continue

                threads = int(observations[0][2])
                hashbits = int(observations[0][3])
                time = int(observations[0][4])/1000 # To get seconds

                if hashbits not in results_dict[algo][threads]:
                    r = Results(algo, threads, hashbits, tuplepower, [time])
                    results_dict[algo][threads][hashbits] = r
                else:
                    results_dict[algo][threads][hashbits].time.append(time)
    
    return results_dict

                

def makePerfGraphForSpeceficThreadUsingBothAlgos(results, thread, name, data_gen_already_removed):
    total_tuples = 2 ** 24
    data_generation_time = results["dry"][1][1].time_average()
    print(data_generation_time)
    hash_bits = range(1,19)
    
    ctm_perf = []
    independent_perf = []

    for h in hash_bits:
        #For Count then move
        ctm_time_average = results["ctm"][thread][h].time_average()
        if not data_gen_already_removed:
            ctm_partition_time  = ctm_time_average - data_generation_time
        else:
            ctm_partition_time = ctm_time_average
        ctm_million_tuples_per_sec = (total_tuples / ctm_partition_time) / 1_000_000
        ctm_perf.append(ctm_million_tuples_per_sec)

        #For Independant
        i_time_average = results["i"][thread][h].time_average()
        if not data_gen_already_removed:
            i_partition_time = i_time_average - data_generation_time
        else:
            i_partition_time = i_time_average
        i_million_tuples_per_sec = (total_tuples / i_partition_time) / 1_000_000
        independent_perf.append(i_million_tuples_per_sec)
    

    plt.figure(figsize=(10, 6))
    plt.plot(hash_bits, ctm_perf, marker='o', label='Count-Then-Move (CTM)', linewidth=2)
    plt.plot(hash_bits, independent_perf, marker='s', label='Independent', linewidth=2)
    
    plt.title('Performance: Millions of Tuples per Second', fontsize=16)
    plt.xticks(range(19))
    plt.xlabel('Hash Bits', fontsize=12)
    plt.ylabel('Millions of Tuples per Second', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig(name + '.png')
    plt.close()



def makePerfGraphForAllThreadsInGivenAlgo(results, algo_name, name):
    total_tuples = 2 ** 24
    data_generation_time = results["dry"][1][1].time_average()
    hash_bits = range(1, 19)
    
    # Get all thread counts available for this algorithm
    thread_counts = sorted(results[algo_name].keys())
    
    plt.figure(figsize=(12, 8))
    
    # Line styles and markers for different threads
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    colors = plt.cm.viridis(np.linspace(0, 1, len(thread_counts)))
    
    # Plot a line for each thread count
   


#TODO:
# - Also go through data_XXX.txt files 
# - Do nessesary calculations for example: tuples / (time - data gen)
# - Create plots
# - Figure out what other plots you need (go to paper)
# - Make plots pretty


def main():
    # ==== PERF GRAPHS =====
    algos = ["ctm", "i", "dry"] #NOTE: Order has to match perf_dirs
    perf_dirs = [ctm_perf_dir, independent_perf_dir, dry_perf_dir]

    results_dict = getPerfResultsFromDirS(algos, perf_dirs)
    makePerfGraphForSpeceficThreadUsingBothAlgos(results_dict, 32, 'performance_graph_from_perf', False) #Should be the same as performance_graph_from_stdout

    # ==== STDOUT GRAPHS =====
    file_paths = [data_ctm, data_independent, data_dry]
    tuplepower = 24
    stdout_results_dict = getDataFileResults(algos, file_paths, tuplepower)

    makePerfGraphForSpeceficThreadUsingBothAlgos(stdout_results_dict, 32, 'performance_graph_from_stdout', True) #Should be the same as performance_graph_from_perf


    
if __name__ == "__main__":
    main()
