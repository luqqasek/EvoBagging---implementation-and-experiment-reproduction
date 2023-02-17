import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics as mm
import random
from inspect import getmembers, isfunction
from tqdm import tqdm


class EvoBagging:
    def __init__(self,
                 number_of_initial_bags: int,
                 maximum_bag_size: int,
                 generation_gap_size: int,
                 k: float,
                 mutation_size: int,
                 number_of_mutated_bags: int,
                 fitness_threshold: float = None,
                 number_of_iteration: int = None,
                 performance_threshold: float = None,
                 stopping_condition_performance_threshold: float = None,
                 minimize: bool = False,
                 mode: str = None,
                 metric_name: str = None,
                 logging: bool = False,
                 disable_progress_bar: bool = False,
                 classifier_restricted: bool = True,
                 selection: str = 'naive'):
        """Evolutionary Bagging model.

        Parameters
        ----------
        number_of_initial_bags:  Number of bags (in paper denoted as N)
        maximum_bag_size: Sample size (in paper denoted as S)
        generation_gap_size: number of generation gap (in paper denoted as G)
        k: User-defined hyper-parameter for encouraging larger bags (in paper denoted as K)
        mutation_size: number of data instances mutated in each bag in mutation stage (in paper denoted as MS)
        number_of_mutated_bags: number of individuals that are mutated (in paper denoted as M)
        fitness_threshold: stopping condition, if average fitness is higher, training is stopped
        number_of_iteration: Maximum number of iterations while fitting model to data (this is stopping condition)
        stopping_condition_performance_threshold: for regression problems specify threshold of metric until data
            instances are moved to other child
        minimize: required when stopping_condition_performance_threshold provided. True if performance should be
            maximized , False otherwise
        mode: variable specifying whether the problem is classification or regression
        metric_name: valid metric name from sklearn.metrics package. List of possible metrics
            Regression
            https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
            Classification
            https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        logging: boolean value specifying whether training metric should be printed every iteration
        disable_progress_bar: boolean value specifying whether training progress bar should be disabled
        classifier_restricted: boolean value specifying whether max depth of tree should be limited to
            number of features
        selection: selection scheme for crossover, naive or rank
        """
        assert fitness_threshold or number_of_iteration or \
               stopping_condition_performance_threshold, "Specify stopping conditions"
        assert mode in ["classification",
                        "regression"], "Specify whether you want to perform classification or regression"
        assert metric_name in [function[0] for function in getmembers(mm, isfunction)], "Wrong metric"

        # Model hyper-parameters
        self.number_of_initial_bags = number_of_initial_bags
        self.generation_gap_size = generation_gap_size
        self.maximum_bag_size = maximum_bag_size
        self.number_of_mutated_bags = number_of_mutated_bags
        self.mutation_size = mutation_size
        self.K = k
        self.performance_threshold = performance_threshold

        # Stopping criteria
        self.fitness_threshold = fitness_threshold
        self.number_of_iteration = number_of_iteration
        self.stopping_condition_performance_threshold = stopping_condition_performance_threshold

        # Additional attributes
        self.mode = mode
        self.logging = logging
        self.disable_progress_bar = disable_progress_bar
        self.X = None
        self.y = None
        self.model = None
        self.metric = metric_name
        self.minimize = minimize
        self.selection = selection
        self.classifier_restricted = classifier_restricted

    def generate_bag(self):
        """
        Returns ids that belong to randomly generated bag out of all data instances
        """
        # minimum size of bag
        min_size = int(self.maximum_bag_size / 2)

        # size of generated bag
        size = random.randrange(start=min_size, stop=self.maximum_bag_size)

        # output ids
        ids = random.choices(np.arange(self.X.shape[0]), k=size)

        return np.array(ids)

    # noinspection PyUnboundLocalVariable
    def train_classifier_on_one_bag(self, X, y):
        """
        Trains single model on given data and returns performance of this model, model itself and predictions
        """
        if self.mode == "classification":
            if self.classifier_restricted:
                clf = DecisionTreeClassifier(max_depth=X.shape[1])
            else:
                clf = DecisionTreeClassifier()
        elif self.mode == "regression":
            if self.classifier_restricted:
                clf = DecisionTreeRegressor(max_depth=X.shape[1])
            else:
                clf = DecisionTreeRegressor()
        clf.fit(X, y)
        preds = clf.predict(X)
        perf = eval(f"mm.{self.metric}(y, preds)")
        return perf, clf, preds

    def fitness_score(self, performance, bag):
        """Calculate fitness score for given bag"""
        return performance * (self.K + len(bag)) / self.K

    def fit(self, X, y):
        """
        Trains model on given data following evolution strategy:
        1. Initialization of generation 0
        2. Evolving generation until one of stopping conditions is fulfilled
        """
        # 1. Initialization
        self.X = X
        self.y = y
        generation = []  # generation is set of bags with all the information about them

        for i in range(self.number_of_initial_bags):
            # entity_info will have following keys:
            # ids - ids of examples from training set which form a bag
            # performance - performance on this bag
            # classifier - classifier or regression tree trained on bag
            # predictions - predictions for given bag

            # Generating bag
            entity_info = {"ids": self.generate_bag()}

            # Training classifier
            bag_x = self.X[entity_info["ids"], :]
            bag_y = self.y[entity_info["ids"]]

            entity_info["performance"], entity_info["classifier"], entity_info[
                "predictions"] = self.train_classifier_on_one_bag(bag_x, bag_y)
            entity_info["fitness"] = self.fitness_score(entity_info["performance"], entity_info["ids"])

            # Adding bag to generation
            generation.append(entity_info)

        # evaluating current generation
        best_generation = generation.copy()
        self.model = generation.copy()
        y_hat = self.predict(X)
        best_performance = eval(f"mm.{self.metric}(y, y_hat)")

        # 2. Evolution process
        for epoch in tqdm(range(self.number_of_iteration), disable=self.disable_progress_bar):
            generation = self.evolve_generation(generation)

            # evaluating new generation
            for bag_id in range(len(generation)):
                bag_x = self.X[generation[bag_id]["ids"].astype(int), :]
                bag_y = self.y[generation[bag_id]["ids"].astype(int)]

                generation[bag_id]["performance"], generation[bag_id]["classifier"], generation[bag_id][
                    "predictions"] = self.train_classifier_on_one_bag(bag_x, bag_y)
                generation[bag_id]["fitness"] = self.fitness_score(performance=generation[bag_id]["performance"],
                                                                   bag=generation[bag_id]["ids"])

            # Checking whether this generation is better than the best
            self.model = generation.copy()
            y_hat = self.predict(X)
            perf = eval(f"mm.{self.metric}(y, y_hat)")
            if perf > best_performance:
                best_generation = generation.copy()
                best_performance = perf

            # Prints logs if specified
            if self.logging:
                self.print_logs(epoch)

            # Stop iteration if conditions are fulfilled
            if self.check_stopping_conditions():
                self.model = best_generation
                break

        self.model = best_generation.copy()

    def evolve_generation(self, old_generation):
        """
        It does one evolution of generation:
        1. Creates generation gap which is G new bags
        2. Does crossover on ( self.number_of_initial_bags - self.generation_gap_size ) bags from old generation
        3. Merges generation gap with crossover population = new population
        4. Do mutation on M randomly chosen bags from new population
        """
        # Performing generation gap
        new_generation = self.generation_gap()

        # Performing crossover
        crossover = self.crossover(old_generation)
        # noinspection PyTypeChecker
        new_generation = new_generation + crossover

        # Performing mutation
        new_generation = self.mutation(new_generation)

        return new_generation

    def generation_gap(self):
        """
        Generation gap process. Returns generation_gap_size new bags.
        """
        generation = []
        for i in range(self.generation_gap_size):
            entity_info = {"ids": self.generate_bag()}

            # Training classifier
            bag_x = self.X[entity_info["ids"], :]
            bag_y = self.y[entity_info["ids"]]

            entity_info["performance"], entity_info["classifier"], entity_info[
                "predictions"] = self.train_classifier_on_one_bag(bag_x, bag_y)

            # Calculating fitness
            entity_info["fitness"] = self.fitness_score(entity_info["performance"], entity_info["ids"])
            generation.append(entity_info)

        return generation

    def crossover(self, old_generation):
        """
        Conducts crossover process.
        Creates C/2 pairs from old_generation without repetition, where
        C = ( self.number_of_initial_bags - self.generation_gap_size )
        For each pair creates two offspring based on following logic
               take element from parent A, check if prediction is accurate, if yes, move to offspring A,
               if not move to offspring B. Do the the same for parent B but if yes move to offspring B,
               if not move to offspring A.
        """
        c = self.number_of_initial_bags - self.generation_gap_size
        # if c is not even make it even
        if c % 2 != 0:
            c = c-1

        # Selection process
        # Naive selection
        if self.selection == "naive":
            pairs, old_generation = self.naive_selection(old_generation, c)
        # Rank selection
        elif self.selection == "rank":
            pairs, old_generation = self.rank_selection(old_generation, c)

        children = []

        for pair in pairs:
            parent_a = old_generation[pair[0]]
            parent_b = old_generation[pair[1]]
            offspring_a = {"ids": np.array([])}
            offspring_b = {"ids": np.array([])}

            # Moving elements from parent A to offsprings
            parent_a_y_hat = parent_a["predictions"].astype(int)
            parent_a_y = self.y[parent_a["ids"]].astype(int)
            accurate_predictions = self.evaluate_prediction(parent_a_y, parent_a_y_hat)
            offspring_a["ids"] = parent_a["ids"][accurate_predictions]
            offspring_b["ids"] = parent_a["ids"][~accurate_predictions]

            # Moving elements from parent B to offsprings
            parent_b_y_hat = parent_b["predictions"].astype(int)
            parent_b_y = self.y[parent_b["ids"]].astype(int)
            accurate_predictions = self.evaluate_prediction(parent_b_y, parent_b_y_hat)
            offspring_b["ids"] = np.append(offspring_b["ids"], parent_b["ids"][accurate_predictions])
            offspring_a["ids"] = np.append(offspring_a["ids"], parent_b["ids"][~accurate_predictions])

            children.append(offspring_a)
            children.append(offspring_b)

        return children

    @staticmethod
    def rank_selection(generation, c):
        """
        Performs rank selection and returns c pairs of parents without repetition
        based on https://www.obitko.com/tutorials/genetic-algorithms/selection.php
        """
        # in place sorting based on fitness
        generation.sort(key=lambda e: e['fitness'], reverse=True)

        # normalization coefficient
        norm_coefficient = len(generation) * (len(generation) + 1) / 2

        # probabilities of selecting parents
        probabilities = [(i + 1)/norm_coefficient for i in range(len(generation))]

        # sample c parents
        generation = np.random.choice(a=generation, p=probabilities, size=c, replace=False)
        permuted_generation_idxs = np.random.permutation(len(generation))

        # creates c/2 pairs
        pairs = np.split(permuted_generation_idxs, c / 2)
        return pairs, generation

    @staticmethod
    def naive_selection(generation, c):
        """
        Performs naive selection where c individuals with highest fitness are returned as well as pairs for crossover
        """
        # in place sorting based on fitness
        generation.sort(key=lambda e: e['fitness'], reverse=True)
        generation = generation[:c]

        # sample c parents
        permuted_generation_idxs = np.random.permutation(len(generation))

        # creates c/2 pairs
        pairs = np.split(permuted_generation_idxs, c / 2)
        return pairs, generation

    def evaluate_prediction(self, y, y_hat):
        """
        Evaluate method for crossover computations
        returns true if prediction is accurate, false otherwise
        """
        if self.mode == "classification":
            # for classification model performs exact comparison therefore if prediction
            # doesn't match the data instance is moved to another child
            return np.array(y == y_hat)
        elif self.mode == "regression":
            return np.array(np.abs(y-y_hat) < self.performance_threshold)

    def mutation(self, new_population):
        """
        Performs mutation stage. Chooses self.number_of_mutated_bags bags from given population. For each of these bags
        replaces self.mutation_size data instances in this bag with elements randomly chosen from training set but not
        belonging to this bag.
        """
        # Sample individuals to mutate
        elements_to_mutate = random.sample(range(len(new_population)), self.number_of_mutated_bags)
        for i in elements_to_mutate:
            element = new_population[i]

            # Data instances from individual
            x = element["ids"]

            # All data instances belonging to training set
            all_xs = np.arange(self.X.shape[0])

            # Complement of set of data instances from individual
            x_complement = np.setdiff1d(all_xs, x)

            # Choosing data instances to replace
            x_to_replace = np.random.choice(x, self.mutation_size, replace=False)

            # Choosing replacement
            replacement = np.random.choice(x_complement, self.mutation_size, replace=True)

            # Changing xs with random instances
            x = np.delete(x, np.where(np.in1d(x, x_to_replace)))
            x = np.append(x, replacement)

            # Saving mutated individual
            new_population[i]["ids"] = x

        return new_population

    def check_stopping_conditions(self):
        """
        Returns True if one of stopping condition is fulfilled , False otherwise
        """
        conditions_fulfilled = 0
        if self.fitness_threshold:
            finesses = [individual["fitness"] for individual in self.model]
            if np.mean(finesses) > self.fitness_threshold:
                conditions_fulfilled += 1

        if self.stopping_condition_performance_threshold:
            # noinspection PyUnusedLocal
            preds = self.predict(self.X)
            performance = eval(f"mm.{self.metric}(self.y, preds)")
            if self.minimize and performance < self.stopping_condition_performance_threshold:
                conditions_fulfilled += 1
            elif (not self.minimize) and performance > self.stopping_condition_performance_threshold:
                conditions_fulfilled += 1

        return conditions_fulfilled > 0

    def predict(self, X):
        """
        Make predictions on current model using majority voting scheme
        """

        all_predictions = np.zeros([X.shape[0], len(self.model)])
        final_predictions = np.zeros([X.shape[0]])
        for i, individual in enumerate(self.model):
            all_predictions[:, i] = individual["classifier"].predict(X)

        if self.mode == "classification":
            all_predictions = all_predictions.astype(int)
            final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, all_predictions)
        elif self.mode == "regression":
            final_predictions = np.apply_along_axis(lambda x: np.mean(x), 1, all_predictions)

        return final_predictions

    def print_logs(self, epoch):
        """Prints specified logs"""
        pred = self.predict(self.X)
        print(f"\nEpoch : {epoch}")
        print(f"Accuracy : {round(np.sum(pred == self.y) / self.y.shape[0] * 100, 2)} %")
