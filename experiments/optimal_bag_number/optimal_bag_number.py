from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
from pathlib import Path
import os
sys.path.append(str(Path.cwd().parent.parent))
from DataProcessing import DataProcessing

file_name = "results_optimal_bags.csv"

if __name__ == '__main__':
    if os.path.exists(file_name):
        print(f"File : {file_name} removed")
        os.remove(f"{file_name}")

    dataset_names = ["mnist", "breast_cancer", "abalone",
                     "red_wine", "pima", "car", "tic-tac-toe",
                     "ionosphere", "churn", "flare", "ring", "two-spiral"]
    n_bag_range = list(range(5, 100, 5))

    cv_score_max = []
    n_bag_maximal = []

    for data_name in dataset_names:
        print(data_name)
        data = DataProcessing()
        data.from_original_paper(data_name, 0)
        cv_scores = []
        for n_bags in tqdm(n_bag_range):
            clf = BaggingClassifier(n_estimators=n_bags)
            score = cross_val_score(clf, data.X_train, data.y_train, cv=5).mean()
            cv_scores.append(score)
        n_bag = n_bag_range[np.argmax(cv_scores)]
        n_bag_maximal.append(n_bag)
        cv_score_max.append(max(cv_scores))

    results = pd.DataFrame(list(zip(dataset_names, n_bag_maximal, cv_score_max)),
                           columns=['Dataset_name', 'optimal_number_of_bags', "maximal_score"])
    results.to_csv("results_optimal_bags.csv", index=False)

