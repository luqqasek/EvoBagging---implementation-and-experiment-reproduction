import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
sys.path.append(str(Path.cwd().parent.parent))
from DataProcessing import DataProcessing
from EvoBagging import EvoBagging

PATH_to_dict = Path(__file__).parent.parent / "optimal_bag_number" / "results_optimal_bags.csv"
df = pd.read_csv(PATH_to_dict).iloc[:, :-1]
dict_bags = df.set_index('Dataset_name').to_dict()["optimal_number_of_bags"]

PATH_to_params = Path(__file__).parent.parent / "evobagging_grid_search" / "params.json"
f = open(PATH_to_params)
params = json.load(f)

dataset_name = "pima"
number_of_iteration = 15
optimal_number_of_bags = dict_bags[dataset_name]
S = [0.25, 0.5, 0.75, 1]
G = [.0333, .1333, .1667, .20, .5]
M = [.01, .08, .1, .2]
MS = [.01, .06, .1, .2]
K = [1000, 2000, 10000, 20000]
m = params[dataset_name]["M"]
g = params[dataset_name]["G"]
ms = params[dataset_name]["MS"]
k = params[dataset_name]["K"]


def maximum_bag_size_influence(data):
    f = open("maximum_bag_size_s.csv", "a")
    f.write("ratio,test_accuracy,avg_bag_size,avg_tree_depth,restricted\n")

    for restricted in [True, False]:
        for s in S:
            number_of_mutated_bags = int(m * dict_bags[dataset_name])
            generation_gap_size = int(g * dict_bags[dataset_name])
            mutation_size = int(data.X_train.shape[0] * ms)

            model = EvoBagging(number_of_initial_bags=optimal_number_of_bags,
                               maximum_bag_size=int(data.X_train.shape[0]*s),
                               generation_gap_size=generation_gap_size,
                               k=k,
                               mutation_size=mutation_size,
                               mode='classification',
                               number_of_mutated_bags=number_of_mutated_bags,
                               number_of_iteration=number_of_iteration,
                               selection='naive',
                               logging=False,
                               metric_name="accuracy_score",
                               classifier_restricted=restricted)

            model.fit(data.X_train, data.y_train)
            predictions_test = model.predict(data.X_test)
            accuracy = np.sum(predictions_test == data.y_test)/data.y_test.shape[0]*100
            avg_depth = np.mean([gen["classifier"].tree_.max_depth for gen in model.model])
            avg_bag_size = np.mean([len(gen["ids"]) for gen in model.model])

            f.write(f"{s},{accuracy},{avg_bag_size},{avg_depth},{restricted}\n")
    f.close()


def generation_gap_influence(data):
    f = open("generation_gap.csv", "a")
    f.write("G,iteration,avg_fitness,restricted\n")

    for restricted in [True, False]:
        for g_tmp in G:
            number_of_mutated_bags = int(m * dict_bags[dataset_name])
            generation_gap_size = int(g_tmp * dict_bags[dataset_name])
            mutation_size = int(data.X_train.shape[0] * ms)

            model = EvoBagging(number_of_initial_bags=optimal_number_of_bags,
                               maximum_bag_size=int(data.X_train.shape[0]),
                               generation_gap_size=generation_gap_size,
                               k=k,
                               mutation_size=mutation_size,
                               mode='classification',
                               number_of_mutated_bags=number_of_mutated_bags,
                               number_of_iteration=number_of_iteration,
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

            avg_fitness = np.mean([gen["fitness"] for gen in model.model])
            f.write(f"{g_tmp},{0},{avg_fitness},{restricted}\n")

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
                avg_fitness = np.mean([gen["fitness"] for gen in model.model])
                f.write(f"{g_tmp},{epoch + 1},{avg_fitness},{restricted}\n")
    f.close()


def mutated_bags_influence(data):
    f = open("mutated_bags.csv", "a")
    f.write("M,iteration,avg_fitness,restricted\n")

    for restricted in [True, False]:
        for m_tmp in M:
            number_of_mutated_bags = int(m_tmp * dict_bags[dataset_name])
            generation_gap_size = int(g * dict_bags[dataset_name])
            mutation_size = int(data.X_train.shape[0] * ms)

            model = EvoBagging(number_of_initial_bags=optimal_number_of_bags,
                               maximum_bag_size=int(data.X_train.shape[0]),
                               generation_gap_size=generation_gap_size,
                               k=k,
                               mutation_size=mutation_size,
                               mode='classification',
                               number_of_mutated_bags=number_of_mutated_bags,
                               number_of_iteration=number_of_iteration,
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

            avg_fitness = np.mean([gen["fitness"] for gen in model.model])
            f.write(f"{m_tmp},{0},{avg_fitness},{restricted}\n")

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
                avg_fitness = np.mean([gen["fitness"] for gen in model.model])
                f.write(f"{m_tmp},{epoch + 1},{avg_fitness},{restricted}\n")
    f.close()


def mutation_size_influence(data):
    f = open("mutation_size.csv", "a")
    f.write("MS,iteration,avg_fitness,restricted\n")

    for restricted in [True, False]:
        for ms_tmp in MS:
            number_of_mutated_bags = int(m * dict_bags[dataset_name])
            generation_gap_size = int(g * dict_bags[dataset_name])
            mutation_size = int(data.X_train.shape[0] * ms_tmp)

            model = EvoBagging(number_of_initial_bags=optimal_number_of_bags,
                               maximum_bag_size=int(data.X_train.shape[0]),
                               generation_gap_size=generation_gap_size,
                               k=k,
                               mutation_size=mutation_size,
                               mode='classification',
                               number_of_mutated_bags=number_of_mutated_bags,
                               number_of_iteration=number_of_iteration,
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

            avg_fitness = np.mean([gen["fitness"] for gen in model.model])
            f.write(f"{ms_tmp},{0},{avg_fitness},{restricted}\n")

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
                avg_fitness = np.mean([gen["fitness"] for gen in model.model])
                f.write(f"{ms_tmp},{epoch + 1},{avg_fitness},{restricted}\n")
    f.close()


def bag_size_influence(data):
    f = open("bag_size.csv", "a")
    f.write("K,test_accuracy,avg_bag_size,restricted\n")

    for restricted in [True, False]:
        for k_tmp in K:
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
                               number_of_iteration=number_of_iteration,
                               selection='naive',
                               logging=False,
                               metric_name="accuracy_score",
                               classifier_restricted=restricted)

            model.fit(data.X_train, data.y_train)
            predictions_test = model.predict(data.X_test)
            accuracy = np.sum(predictions_test == data.y_test)/data.y_test.shape[0]*100
            avg_bag_size = np.mean([len(gen["ids"]) for gen in model.model])

            f.write(f"{k_tmp},{accuracy},{avg_bag_size},{restricted}\n")
    f.close()


if __name__ == "__main__":

    for file_name in ["bag_size.csv", "generation_gap.csv", "maximum_bag_size_s.csv", "mutated_bags.csv",
                      "mutation_size.csv"]:
        if os.path.exists(file_name):
            print(f"File : {file_name} removed")
            os.remove(f"{file_name}")

    data_pima = DataProcessing()
    data_pima.from_original_paper(dataset_name=dataset_name, test_size=0.2)

    print("MAXIMUM BAG SIZE INFLUENCE")
    maximum_bag_size_influence(data_pima)

    print("GENERATION GAP INFLUENCE")
    generation_gap_influence(data_pima)

    print("NUMBER OF MUTATED BAGS INFLUENCE")
    mutated_bags_influence(data_pima)

    print("MUTATION SIZE INFLUENCE")
    mutation_size_influence(data_pima)

    print("BAG SIZE INFLUENCE")
    bag_size_influence(data_pima)
