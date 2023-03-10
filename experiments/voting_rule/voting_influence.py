import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score
sys.path.append(str(Path.cwd().parent.parent))
from DataProcessing import DataProcessing
from EvoBagging import EvoBagging

PATH_to_dict = Path(__file__).parent.parent / "optimal_bag_number" / "results_optimal_bags.csv"
df = pd.read_csv(PATH_to_dict).iloc[:, :-1]
dict_bags = df.set_index('Dataset_name').to_dict()["optimal_number_of_bags"]

PATH_to_params = Path(__file__).parent.parent / "evobagging_grid_search" / "params.json"
f = open(PATH_to_params)
params = json.load(f)

data_set_names = ["pima", "two-spiral"]
voting_rules = ["majority", 'weighted']


def run_voting_rule_their_implementation():
    f = open("voting_influence.csv", "a")
    f.write("dataset_name,iteration,voting_rule,test_accuracy\n")

    for dataset_name in data_set_names:
        print(dataset_name)
        data = DataProcessing()
        data.from_original_paper(dataset_name=dataset_name, test_size=0.2)

        number_of_mutated_bags = int(params[dataset_name]["M"] * dict_bags[dataset_name])
        generation_gap_size = int(params[dataset_name]["G"] * dict_bags[dataset_name])
        mutation_size = int(data.X_train.shape[0] * params[dataset_name]["MS"])

        model = EvoBagging(number_of_initial_bags=dict_bags[dataset_name],
                           maximum_bag_size=int(data.X_train.shape[0]),
                           generation_gap_size=generation_gap_size,
                           k=params[dataset_name]["K"],
                           mutation_size=mutation_size,
                           mode='classification',
                           number_of_mutated_bags=number_of_mutated_bags,
                           number_of_iteration=20,
                           selection='naive',
                           logging=False,
                           metric_name="accuracy_score",
                           classifier_restricted=True)

        # Running fit method manually

        # 1. Initialization
        model.X = data.X_train
        model.y = data.y_train
        generation = []  # generation is set of bags with all the information about them

        for i in range(model.number_of_initial_bags):
            # Generating bag
            entity_info = {"ids": model.generate_bag()}

            # Training classifier
            bag_x = model.X[entity_info["ids"], :]
            bag_y = model.y[entity_info["ids"]]

            entity_info["performance"], entity_info["classifier"], entity_info[
                "predictions"] = model.train_classifier_on_one_bag(bag_x, bag_y)
            entity_info["fitness"] = model.fitness_score(entity_info["performance"], entity_info["ids"])

            # Adding bag to generation
            generation.append(entity_info)

        # saving generation
        model.model = generation.copy()

        # saving accuracy
        # for first voting rule
        voting_rule = voting_rules[0]
        model.voting_rule = voting_rule
        test_preds = model.predict(data.X_test)
        met_test = accuracy_score(data.y_test, test_preds)
        f.write(f"{dataset_name},{0},{voting_rule},{met_test}\n")
        # for second voting rule
        voting_rule = voting_rules[1]
        model.voting_rule = voting_rule
        test_preds = model.predict(data.X_test)
        met_test = accuracy_score(data.y_test, test_preds)
        f.write(f"{dataset_name},{0},{voting_rule},{met_test}\n")
        model.voting_rule = voting_rules[0]

        # 2. Evolution process
        for epoch in tqdm(range(model.number_of_iteration)):
            generation = model.evolve_generation(generation)

            # evaluating new generation
            for bag_id in range(len(generation)):
                bag_x = model.X[generation[bag_id]["ids"].astype(int), :]
                bag_y = model.y[generation[bag_id]["ids"].astype(int)]

                generation[bag_id]["performance"], generation[bag_id]["classifier"], generation[bag_id][
                    "predictions"] = model.train_classifier_on_one_bag(bag_x, bag_y)
                generation[bag_id]["fitness"] = model.fitness_score(performance=generation[bag_id]["performance"],
                                                                    bag=generation[bag_id]["ids"])

            # Saving generation
            model.model = generation.copy()

            # saving experiment
            # for first voting rule
            voting_rule = voting_rules[0]
            model.voting_rule = voting_rule
            test_preds = model.predict(data.X_test)
            met_test = accuracy_score(data.y_test, test_preds)
            f.write(f"{dataset_name},{epoch+1},{voting_rule},{met_test}\n")
            # for second voting rule
            voting_rule = voting_rules[1]
            model.voting_rule = voting_rule
            test_preds = model.predict(data.X_test)
            met_test = accuracy_score(data.y_test, test_preds)
            f.write(f"{dataset_name},{epoch+1},{voting_rule},{met_test}\n")
            model.voting_rule = voting_rules[0]
    f.close()


def run_voting_corrected():
    f = open("voting_influence_corrected.csv", "a")
    f.write("dataset_name,iteration,voting_rule,test_accuracy\n")

    for voting_rule in voting_rules:
        for dataset_name in data_set_names:
            print(f"{dataset_name} with voting rule : {voting_rule}")
            data = DataProcessing()
            data.from_original_paper(dataset_name=dataset_name, test_size=0.2)

            number_of_mutated_bags = int(params[dataset_name]["M"] * dict_bags[dataset_name])
            generation_gap_size = int(params[dataset_name]["G"] * dict_bags[dataset_name])
            mutation_size = int(data.X_train.shape[0] * params[dataset_name]["MS"])

            model = EvoBagging(number_of_initial_bags=dict_bags[dataset_name],
                               maximum_bag_size=int(data.X_train.shape[0]),
                               generation_gap_size=generation_gap_size,
                               k=params[dataset_name]["K"],
                               mutation_size=mutation_size,
                               mode='classification',
                               number_of_mutated_bags=number_of_mutated_bags,
                               number_of_iteration=20,
                               selection='naive',
                               logging=False,
                               metric_name="accuracy_score",
                               voting_rule=voting_rule,
                               classifier_restricted=True)

            # Running fit method manually

            # 1. Initialization
            model.X = data.X_train
            model.y = data.y_train
            generation = []  # generation is set of bags with all the information about them

            for i in range(model.number_of_initial_bags):
                # Generating bag
                entity_info = {"ids": model.generate_bag()}

                # Training classifier
                bag_x = model.X[entity_info["ids"], :]
                bag_y = model.y[entity_info["ids"]]

                entity_info["performance"], entity_info["classifier"], entity_info[
                    "predictions"] = model.train_classifier_on_one_bag(bag_x, bag_y)
                entity_info["fitness"] = model.fitness_score(entity_info["performance"], entity_info["ids"])

                # Adding bag to generation
                generation.append(entity_info)

            # saving generation
            model.model = generation.copy()

            # saving accuracy
            model.voting_rule = voting_rule
            test_preds = model.predict(data.X_test)
            met_test = accuracy_score(data.y_test, test_preds)
            f.write(f"{dataset_name},{0},{voting_rule},{met_test}\n")

            # 2. Evolution process
            for epoch in tqdm(range(model.number_of_iteration)):
                generation = model.evolve_generation(generation)

                # evaluating new generation
                for bag_id in range(len(generation)):
                    bag_x = model.X[generation[bag_id]["ids"].astype(int), :]
                    bag_y = model.y[generation[bag_id]["ids"].astype(int)]

                    generation[bag_id]["performance"], generation[bag_id]["classifier"], generation[bag_id][
                        "predictions"] = model.train_classifier_on_one_bag(bag_x, bag_y)
                    generation[bag_id]["fitness"] = model.fitness_score(performance=generation[bag_id]["performance"],
                                                                        bag=generation[bag_id]["ids"])

                # Saving generation
                model.model = generation.copy()

                # saving experiment
                test_preds = model.predict(data.X_test)
                met_test = accuracy_score(data.y_test, test_preds)
                f.write(f"{dataset_name},{epoch + 1},{voting_rule},{met_test}\n")

    f.close()


if __name__ == "__main__":
    for file_name in ["voting_influence.csv", "voting_influence_corrected.csv"]:
        if os.path.exists(file_name):
            print(f"File : {file_name} removed")
            os.remove(f"{file_name}")

    # Running first experiment
    run_voting_rule_their_implementation()

    # Running our experiment
    run_voting_corrected()
