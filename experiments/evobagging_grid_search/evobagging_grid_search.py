import numpy as np
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import itertools
sys.path.append(str(Path.cwd().parent.parent))
from DataProcessing import DataProcessing
from EvoBagging import EvoBagging

# GLOBAL VARIABLES
PATH_to_dict = Path(__file__).parent.parent / "optimal_bag_number" / "results_optimal_bags.csv"
CPUs = multiprocessing.cpu_count()
output_file_name = "evobagging_grid_search.csv"
dataset_names = ["mnist", "breast_cancer", "abalone",
                 "red_wine", "pima", "car", "tic-tac-toe",
                 "ionosphere", "churn", "flare", "ring", "two-spiral"]
number_of_iteration = {"mnist": 20,
                       "breast_cancer": 20,
                       "abalone": 35,
                       "red_wine": 25,
                       "pima": 15,
                       "car": 30,
                       "tic-tac-toe": 20,
                       "ionosphere": 20,
                       "churn": 35,
                       "flare": 40,
                       "ring": 15,
                       "two-spiral": 40}
G = [0.1, 0.15, 0.20, 0.25, 0.30]
M = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
MS = [0.05, 0.1]
K = [10 ** 3, 2 * 10 ** 3, 3 * 10 ** 3, 4 * 10 ** 3, 5 * 10 ** 3, 6 * 10 ** 3, 7 * 10 ** 3, 8 * 10 ** 3, 9 * 10 ** 3,
     10 ** 4, 11 * 10 ** 3, 12 * 10 ** 3, 13 * 10 ** 3, 14 * 10 ** 3, 15 * 10 ** 3, 16 * 10 ** 3, 17 * 10 ** 3,
     18 * 10 ** 3, 19 * 10 ** 3, 20 * 10 ** 3]
df = pd.read_csv(PATH_to_dict).iloc[:, :-1]
dict_bags = df.set_index('Dataset_name').to_dict()["optimal_number_of_bags"]
total_number_of_tasks = len(G) * len(M) * len(MS) * len(K) * len(dataset_names)
random_states = [1, 10, 20, 50, 70]
# END OF GLOBAL VARIABLES


def run_5cv_evobagging(work_data):
    cv_scores = []
    # Unpacking work data
    dataset_name = work_data[0]
    m = work_data[1]
    ms = work_data[2]
    k = work_data[3]
    g = work_data[4]

    for r in random_states:
        data = DataProcessing()
        data.from_original_paper(dataset_name, 0.2, r)

        number_of_mutated_bags = int(m * dict_bags[dataset_name])
        generation_gap_size = int(g * dict_bags[dataset_name])
        mutation_size = int(data.X_train.shape[0] * ms)

        model = EvoBagging(number_of_initial_bags=dict_bags[dataset_name],
                           maximum_bag_size=data.X_train.shape[0],
                           generation_gap_size=generation_gap_size,
                           k=k,
                           mutation_size=mutation_size,
                           mode='classification',
                           number_of_mutated_bags=number_of_mutated_bags,
                           number_of_iteration=number_of_iteration[dataset_name],
                           selection='naive',
                           logging=False,
                           disable_progress_bar=True,
                           metric_name="accuracy_score")

        model.fit(data.X_train, data.y_train)
        pred_test = model.predict(data.X_test)
        cv_scores.append((np.sum(pred_test == data.y_test) / data.y_test.shape[0]))

    with open(output_file_name, "a") as f:
         f.write(f"{dataset_name},{g},{m},{ms},{k},{np.mean(cv_scores)}" + "\n")


if __name__ == "__main__":
    with open(output_file_name, "a") as f:
        f.write("Dataset_name,G,M,MS,K,CV_score\n")

    for dataset_name in dataset_names:
        print(f"Processing dataset : {dataset_name}")
        list_data = [[dataset_name], M, MS, K, G]
        work_data = [p for p in itertools.product(*list_data)]

        p = multiprocessing.Pool(CPUs)
        for _ in tqdm(p.imap_unordered(run_5cv_evobagging, work_data), total=len(work_data)):
            pass
