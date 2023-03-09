from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path
from tqdm import tqdm
import os
import warnings
sys.path.append(str(Path.cwd().parent.parent))
from DataProcessing import DataProcessing
from EvoBagging import EvoBagging
warnings.filterwarnings("ignore")

file_name = "nbit_parity.csv"
models = ["BaggingClassifier", "RandomForestClassifier", "ExtraTreesClassifier", "EvoBagging"]
n_bag_range = list(range(5, 100, 5))
params = {"G": 0.2, "M": 0.1, "MS": 1, "K": 500}
nbits = [6, 8]
iter = 50
n_experiments = 30


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
    f.write("nbit,model,n_bags,mean_train_accuracy,sd_train_accuracy,mean_test_accuracy,sd_test_accuracy\n")

    for bits in nbits:
        dataset_name = f"{bits}bit"
        data = DataProcessing()
        data.from_original_paper(dataset_name=dataset_name, test_size=0.2)
        for n_bags in n_bag_range:
            for baseline in models:
                all_met_train = []
                all_met_test = []

                print(f"Training {baseline} on {dataset_name} dataset - number of bags : {n_bags}")
                for _ in tqdm(range(n_experiments)):
                    if baseline in ["BaggingClassifier", "ExtraTreesClassifier", "RandomForestClassifier"]:
                        clf = eval(f"{baseline}(n_estimators={n_bags},n_jobs=-1)")
                        met_train, met_test = eval_baseline_sklearn(clf,
                                                                    data.X_train, data.y_train,
                                                                    data.X_test, data.y_test)

                    elif baseline == "EvoBagging":
                        model = EvoBagging(number_of_initial_bags=n_bags,
                                           maximum_bag_size=int(data.X_train.shape[0]),
                                           generation_gap_size=int(params["G"] * n_bags),
                                           k=params["K"],
                                           mutation_size=params["MS"],
                                           mode='classification',
                                           number_of_mutated_bags=int(params["M"] * n_bags),
                                           number_of_iteration=iter,
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
                f.write(f"{bits},{baseline},{n_bags},{np.mean(all_met_train)},{np.std(all_met_train)},{np.mean(all_met_test)},{np.std(all_met_test)}\n")

