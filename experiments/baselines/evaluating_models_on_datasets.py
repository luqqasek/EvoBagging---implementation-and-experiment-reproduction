from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
import json
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
import os
import warnings
sys.path.append(str(Path.cwd().parent.parent))
from DataProcessing import DataProcessing
from EvoBagging import EvoBagging

warnings.filterwarnings("ignore")

PATH_to_dict = Path(__file__).parent.parent / "optimal_bag_number" / "results_optimal_bags.csv"
df = pd.read_csv(PATH_to_dict).iloc[:, :-1]
dict_bags = df.set_index('Dataset_name').to_dict()["optimal_number_of_bags"]

PATH_to_params = Path(__file__).parent.parent / "evobagging_grid_search" / "params.json"
f = open(PATH_to_params)
params = json.load(f)

file_name = "training_results.csv"

n_experiments = 30
dataset_names = ["red_wine", "abalone", "breast_cancer", "pima", "mnist", "car", "tic-tac-toe",
                 "ionosphere", "churn", "flare", "ring", "two-spiral"]
models = ["BaggingClassifier", "RandomForestClassifier", "ExtraTreesClassifier", "XGBoost", "EvoBagging"]
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


def eval_baseline_sklearn(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    met_train = accuracy_score(y_train, train_preds)
    met_test = accuracy_score(y_test, test_preds)
    return met_train, met_test


if __name__ == "__main__":
    if os.path.exists(file_name):
        print(f"File : {file_name} removed")
        os.remove(f"{file_name}")

    f = open(file_name, "a")
    f.write("dataset,model,mean_train_accuracy,sd_train_accuracy,mean_test_accuracy,sd_test_accuracy\n")

    for dataset_name in dataset_names:

        print(f"Training on {dataset_name} dataset")

        data = DataProcessing()
        data.from_original_paper(dataset_name=dataset_name, test_size=0.2)

        for baseline in models:

            all_met_train = []
            all_met_test = []

            print(f"Training {baseline} on {dataset_name} dataset")
            for _ in tqdm(range(n_experiments)):
                if baseline in ["BaggingClassifier", "ExtraTreesClassifier", "RandomForestClassifier"]:
                    clf = eval(f"{baseline}(n_estimators={dict_bags[dataset_name]},n_jobs=-1)")
                    met_train, met_test = eval_baseline_sklearn(clf,
                                                                data.X_train, data.y_train,
                                                                data.X_test, data.y_test)

                elif baseline == "XGBoost":
                    if len(np.unique(data.y_train)) > 2:
                        clf = xgb.XGBClassifier(n_estimators=dict_bags[dataset_name], objective='multi:softprob')
                    else:
                        clf = xgb.XGBClassifier(n_estimators=dict_bags[dataset_name])
                    clf.fit(data.X_train, data.y_train, eval_metric='mlogloss')
                    train_preds = clf.predict(data.X_train)
                    test_preds = clf.predict(data.X_test)
                    met_train = accuracy_score(data.y_train, train_preds)
                    met_test = accuracy_score(data.y_test, test_preds)

                elif baseline == "EvoBagging":
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
                                       disable_progress_bar=True,
                                       metric_name="accuracy_score")
                    # Training
                    model.fit(data.X_train, data.y_train)

                    test_preds = model.predict(data.X_test)
                    train_preds = model.predict(data.X_train)

                    # Calculating metrics
                    met_train = accuracy_score(data.y_train, train_preds)
                    met_test = accuracy_score(data.y_test, test_preds)

                all_met_train.append(met_train)
                all_met_test.append(met_test)

            f.write(f"{dataset_name},{baseline},{np.mean(all_met_train)},{np.std(all_met_train)},{np.mean(all_met_test)},{np.std(all_met_test)}\n")
    f.close()

