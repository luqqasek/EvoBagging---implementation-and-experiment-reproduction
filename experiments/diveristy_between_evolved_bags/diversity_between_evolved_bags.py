from sklearn.ensemble import BaggingClassifier
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import os
import warnings
import json
sys.path.append(str(Path.cwd().parent.parent))
from DataProcessing import DataProcessing
from diversity import *
from EvoBagging import EvoBagging
warnings.filterwarnings("ignore")

PATH_to_dict = Path(__file__).parent.parent / "optimal_bag_number" / "results_optimal_bags.csv"
df = pd.read_csv(PATH_to_dict).iloc[:, :-1]
dict_bags = df.set_index('Dataset_name').to_dict()["optimal_number_of_bags"]

PATH_to_params = Path(__file__).parent.parent / "evobagging_grid_search" / "params.json"
f = open(PATH_to_params)
params = json.load(f)

file_name = "diversity_between_bags.csv"
dataset_names = ["red_wine", "ring", "mnist", "car"]
iterations = {"breast_cancer": 20,
              "red_wine": 25,
              "pima": 15,
              "mnist": 20,
              "abalone": 25,
              "car": 30,
              "tic-tac-toe": 20,
              "ionosphere": 20,
              "churn": 35,
              "flare": 40,
              "ring": 15,
              "two-spiral": 40}
diversity_measures = ["Q statistics",
                      "Disagreement",
                      "Double fault",
                      "KW variance",
                      "Entropy",
                      "Generalized diversity"]


def get_diversity_measures(preds, y):
    q_stats = q_statistics(preds, y)
    disag = disagreement(preds, y)
    dfault = double_fault(preds, y)
    kw_var = kohavi_wolpert_variance(preds, y)
    e = entropy(preds, y)
    gen_div = generalized_diversity(preds, y)
    return q_stats, disag, dfault, kw_var, e, gen_div


def calculate_diversity():
    f = open(file_name, "a")
    f.write("dataset_name,measure,bagging_measure,evobagging_measure\n")

    for dataset_name in dataset_names:
        print(dataset_name)
        data = DataProcessing()
        data.from_original_paper(dataset_name=dataset_name, test_size=0.2)

        # Training Evobagging model
        model = EvoBagging(number_of_initial_bags=dict_bags[dataset_name],
                           maximum_bag_size=int(data.X_train.shape[0]),
                           generation_gap_size=int(params[dataset_name]["G"] * dict_bags[dataset_name]),
                           k=params[dataset_name]["K"],
                           mutation_size=int(data.X_train.shape[0] * params[dataset_name]["MS"]),
                           mode='classification',
                           number_of_mutated_bags=int(params[dataset_name]["M"] * dict_bags[dataset_name]),
                           number_of_iteration=iterations[dataset_name],
                           selection='naive',
                           logging=False,
                           disable_progress_bar=False,
                           metric_name="accuracy_score")
        model.fit(data.X_train, data.y_train)

        # Evaluating measures on EvoBagging
        evobagging_preds = np.zeros((data.y_test.shape[0], len(model.model)))
        for i, bag in enumerate(model.model):
            preds = bag['classifier'].predict(data.X_test)
            evobagging_preds[:, i] = preds
        evo_bagging_diversity_measures = get_diversity_measures(evobagging_preds, data.y_test)

        # Training bagging classifier
        clf = BaggingClassifier(n_estimators=dict_bags[dataset_name])
        clf.fit(data.X_train, data.y_train)

        # Evaluating measures on Bagging
        bagging_preds = np.zeros((data.y_test.shape[0], dict_bags[dataset_name]))
        for i, est in enumerate(clf.estimators_):
            preds = est.predict(data.X_test)
            bagging_preds[:, i] = preds
        bagging_diversity_measures = get_diversity_measures(bagging_preds, data.y_test)

        for i in range(len(diversity_measures)):
            f.write(f"{dataset_name},{diversity_measures[i]},{bagging_diversity_measures[i]},{evo_bagging_diversity_measures[i]}\n")


if __name__ == "__main__":
    if os.path.exists(file_name):
        print(f"File : {file_name} removed")
        os.remove(f"{file_name}")

    # creating table
    calculate_diversity()
