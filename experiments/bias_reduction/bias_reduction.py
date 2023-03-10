import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn import metrics as mm
import json

sys.path.append(str(Path.cwd().parent.parent))
from DataProcessing import DataProcessing
from EvoBagging import EvoBagging

PATH_to_dict = Path(__file__).parent.parent / "optimal_bag_number" / "results_optimal_bags.csv"
df = pd.read_csv(PATH_to_dict).iloc[:, :-1]
dict_bags = df.set_index('Dataset_name').to_dict()["optimal_number_of_bags"]

test_size = 0
datasets_for_graph = ["breast_cancer", "red_wine", "pima", "mnist"]
datasets_for_table = ["red_wine", "abalone", "breast_cancer", "pima", "mnist", "car", "tic-tac-toe",
                      "ionosphere", "churn", "flare", "ring", "two-spiral"]

PATH_to_params = Path(__file__).parent.parent / "evobagging_grid_search" / "params.json"
f = open(PATH_to_params)
params = json.load(f)

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


def create_graph_data():
    f = open("bias_reduction_graph.csv", "a")
    f.write("dataset_name,iteration,bias,restricted\n")

    for dataset_name in datasets_for_graph:
        m = params[dataset_name]["M"]
        optimal_number_of_bags = dict_bags[dataset_name]
        g = params[dataset_name]["G"]
        ms = params[dataset_name]["MS"]
        k = params[dataset_name]["K"]

        data = DataProcessing()
        data.from_original_paper(dataset_name=dataset_name, test_size=test_size)

        for restricted in [True, False]:
            number_of_mutated_bags = int(m * dict_bags[dataset_name])
            generation_gap_size = int(g * dict_bags[dataset_name])
            mutation_size = int(data.X_train.shape[0] * ms)

            model = EvoBagging(number_of_initial_bags=optimal_number_of_bags,
                               maximum_bag_size=int(data.X_train.shape[0]),
                               generation_gap_size=generation_gap_size,
                               k=k,
                               mutation_size=mutation_size,
                               mode='classification',
                               number_of_mutated_bags=number_of_mutated_bags,
                               number_of_iteration=10,
                               selection='naive',
                               logging=False,
                               metric_name="accuracy_score",
                               classifier_restricted=restricted)

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

            biases = []
            for bag in model.model:
                predictions = bag["classifier"].predict(data.X_train)
                biases.append(np.sum(predictions != data.y_train) / len(data.y_train))
            f.write(f"{dataset_name},{0},{np.mean(biases)},{restricted}\n")

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
                biases = []
                for bag in model.model:
                    predictions = bag["classifier"].predict(data.X_train)
                    biases.append(np.sum(predictions != data.y_train) / len(data.y_train))
                f.write(f"{dataset_name},{epoch + 1},{np.mean(biases)},{restricted}\n")
    f.close()


def create_table_data():
    f = open("bias_reduction_table.csv", "a")
    f.write("dataset_name,first_bias,last_bias,reduction,restricted\n")

    for dataset_name in datasets_for_table:
        print(dataset_name)
        m = params[dataset_name]["M"]
        optimal_number_of_bags = dict_bags[dataset_name]
        g = params[dataset_name]["G"]
        ms = params[dataset_name]["MS"]
        k = params[dataset_name]["K"]
        n_iter = iterations[dataset_name]

        data = DataProcessing()
        data.from_original_paper(dataset_name=dataset_name, test_size=test_size)

        for restricted in [True, False]:
            number_of_mutated_bags = int(m * dict_bags[dataset_name])
            generation_gap_size = int(g * dict_bags[dataset_name])
            mutation_size = int(data.X_train.shape[0] * ms)

            model = EvoBagging(number_of_initial_bags=optimal_number_of_bags,
                               maximum_bag_size=int(data.X_train.shape[0]),
                               generation_gap_size=generation_gap_size,
                               k=k,
                               mutation_size=mutation_size,
                               mode='classification',
                               number_of_mutated_bags=number_of_mutated_bags,
                               number_of_iteration=n_iter,
                               selection='naive',
                               logging=False,
                               metric_name="accuracy_score",
                               classifier_restricted=restricted)

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

            # evaluating current generation
            best_generation = generation.copy()
            model.model = generation.copy()
            y_hat = model.predict(data.X_train)
            best_performance = eval(f"mm.{model.metric}(data.y_train, y_hat)")

            biases = []
            for bag in model.model:
                predictions = bag["classifier"].predict(data.X_train)
                biases.append(np.sum(predictions != data.y_train) / len(data.y_train))
            first_bias = np.mean(biases)

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

                # Checking whether this generation is better than the best
                model.model = generation.copy()
                y_hat = model.predict(data.X_train)
                perf = eval(f"mm.{model.metric}(data.y_train, y_hat)")
                if perf > best_performance:
                    best_generation = generation.copy()
                    best_performance = perf

            #model.model = best_generation.copy()
            # saving experiment
            biases = []
            for bag in model.model:
                predictions = bag["classifier"].predict(data.X_train)
                biases.append(np.sum(predictions != data.y_train) / len(data.y_train))
            last_bias = np.mean(biases)
            f.write(
                f"{dataset_name},{first_bias},{last_bias},{(first_bias - last_bias) / first_bias * 100},{restricted}\n")
    f.close()


if __name__ == "__main__":
    for file_name in ["bias_reduction_graph.csv", "bias_reduction_table.csv"]:
        if os.path.exists(file_name):
            print(f"File : {file_name} removed")
            os.remove(f"{file_name}")

    print("CREATING GRAPH DATA")
    create_graph_data()

    print("CREATING TABLE DATA")
    create_table_data()
