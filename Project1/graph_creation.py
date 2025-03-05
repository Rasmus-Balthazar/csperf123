from collections import defaultdict
import os
import re
from dataclasses import dataclass
import matplotlib.pyplot as plt

# === Where to find data ===
data_ctm = "out/data_cmt.txt" #file for stdout for cmt
data_independent = "out/data_independent.txt" #file for stdout for independent
data_dry = "out/data_dry.txt" #file for stdout for data generation

ctm_perf_dir = "out/ctm/"
independent_perf_dir = "out/independent/"
dry_perf_dir = "out/dry/"

# ==== Data Structures ====
iterations = 10 

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
input_perm_matcher = re.compile(r"(Performance counter stats for \'\.\/main\.exe )([\d]+).([\d]+).([\d]+)", re.MULTILINE)
time_matcher = re.compile(r"([0-9]*\.?[0-9]*)( seconds time elapsed)", re.MULTILINE)


def getPerfResultsFromDirS(algos, perf_dirs):
    results_dict: dict[str, dict[int, dict[int, Results]]] = defaultdict(lambda: defaultdict(lambda: dict()))

    for i in range(3):
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

def makePerfGraph(results, thread):
    total_tuples = 2 ** 24
    data_generation_time = results["dry"][1][1].time_average()
    hash_bits = range(1,19)
    
    ctm_perf = []
    independent_perf = []

    for h in hash_bits:
        #FOr Count then move
        ctm_time_average = results["ctm"][thread][h].time_average()
        ctm_partition_time  =ctm_time_average - data_generation_time
        ctm_million_tuples_per_sec = (total_tuples / ctm_partition_time) / 1_000_000
        ctm_perf.append(ctm_million_tuples_per_sec)

        #FOr Independant
        i_time_average = results["i"][thread][h].time_average()
        i_partition_time  =i_time_average - data_generation_time
        i_million_tuples_per_sec = (total_tuples / i_partition_time) / 1_000_000
        independent_perf.append(i_million_tuples_per_sec)
    

    plt.figure(figsize=(10, 6))
    plt.plot(hash_bits, ctm_perf, marker='o', label='Count-Then-Move (CTM)', linewidth=2)
    plt.plot(hash_bits, independent_perf, marker='s', label='Independent', linewidth=2)
    
    plt.title('Performance: Millions of Tuples per Second', fontsize=16)
    plt.xlabel('Hash Bits', fontsize=12)
    plt.ylabel('Millions of Tuples per Second', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    

    plt.savefig('performance_graph.png')
    plt.close()


#TODO:
# - Also go through data_XXX.txt files 
# - Do nessesary calculations for example: tuples / (time - data gen)
# - Create plots
# - Figure out what other plots you need (go to paper)
# - Make plots pretty


def main():
    algos = ["ctm", "i", "dry"] #NOTE: Order has to match perf_dirs
    perf_dirs = [ctm_perf_dir, independent_perf_dir, dry_perf_dir]

    results_dict = getPerfResultsFromDirS(algos, perf_dirs)
    makePerfGraph(results_dict, 32)

if __name__ == "__main__":
    main()
