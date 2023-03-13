# EvoBagging-implementation

Implementation of Evolutionary Bagging algorithm as part of assignment for Ensamble Learning Course at Centrale-Supelec

## Description

This repository consists of implementation of Evolutionary bagging algorithm proposed in "Evolutionary bagging for 
ensemble learning" by Giang Ngoa , Rodney Beardb, Rohitash Chandraa 
[Link to article](https://arxiv.org/pdf/2208.02400.pdf). Original implementation given by authors of paper can be found under following link 
[repository](https://github.com/sydney-machine-learning/evolutionary-bagging/tree/ff45402bff174f8ac5a4710c0b8425456f34e93b).
Our implementation decreases computation speed 7 times (on red wine dataset original implementation took 50s while ours 7s to train 
for 25 iterations) [Notebook with results](implementation_comparison/implementation_comparison.ipynb). However results reported in paper are different than one computed her. Despite classification model our implementation allows user to create regression model.

## Repository organization
```
.
├── data 
│   *folder containing data files from original repostiory*
├── experiments 
│   *folder containing subdirectories with scripts to recreate experiments from article as well as .csv with results*
│    detail description of conducted experiments and python notebook file to plot graphs and create tables.    
│    ├── comparison_with_paper
│    ├── baselines
│    ├── bias_reduction
│    ├── diveristy_between_evolved_bags
│    ├── evobagging_grid_search
│    ├── influence_of_hyper_parameters
│    ├── nbit_parity
│    ├── optimal_bag_number     
│    └── voting_rule
├── implementation_comparison
│   *folder containing results of comparison of training Evolutionary Bagging algorithm on red wine dataset using 
│   original implementation as well and implementation from this repository*
├── DataProcessing.py
│   *class implementation of data container for easier data preparation for EvoBagging class*
└── Evobagging.py
    *class implementation of Evolutionary Bagging algorithm*
```

## Example usage

This repository provides access to transformed data used in original experiment by DataProcessing class. After loading data
is ready to training. After loading DataProcessing from DataProcessing.py and EvoBagging from EvoBagging.py

```python
data = DataProcessing()
data.from_original_paper(dataset_name="red_wine", test_size=0.2)
```
To train model we need to firstly initiate it and then use fit method. 

```python
model = EvoBagging(number_of_initial_bags=50,
                   maximum_bag_size=data.X_train.shape[0],
                   generation_gap_size=10,
                   k=5000,
                   mutation_size=65,
                   mode='classification',
                   number_of_mutated_bags=5,
                   number_of_iteration=25,
                   selection='naive',
                   logging=False,
                   metric_name="accuracy_score")
model.fit(data.X_train, data.y_train)
```
In order to predict on test set one need to run following
```python
predictions_test = model.predict(data.X_test)
```

It is not obligatory to use DataProcessing() class. Model accept numpy matrix of size (num_data_instance, num_features)
as X_train and numpy array of size (num_data_instance, ) as y_train

Thera are:
- 2 possible selections schemes: naive and rank selection based on fitness
- 2 modes : "regression" and "classification"
- multiple metrics possible to use, the requirement is that they need only two arguments prediction vector and true vector: 
  - Regression - https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
  - Classification - https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

Further documentation is provided in EvoBagging class

## Implementation of experiments

In subdirectory experiments one can find implementation and results of experiments from cited paper. Description of
experiments can be found [here](experiments/README.md). Comparison of graphs and tables reported in paper can be found
[here](experiments/comparison_with_paper/README.md). Graphs were made by following [python notebook](experiments/Graph_creation.ipynb)

## Inaccuracies and differences between original implementation and paper

What is written in paper | What is implemented by authors in their repository | What we did | 
--- | --- | --- | 
"Bag B will be mutated by replacing selected random samples given by the mutation size (MS) from B with the same amount of random samples from complement of B." | All occurrences of sampled replacements are deleted from bag and replaced with new observations, however as mutation size is fixed it can happen that more samples will be deleted than introduced to bag | the same as in authors repository |
No information about data preprocessing | Data is significantly changed e.g in red wine there is initially 6 classes from 3 to 8, authors assign all values <6,5 as 0 and >6,5 as 1 | we follow what was done in authors repository |
"Note that we use a binary encoded evolutionary algorithm where each gene in the chromosome (individual) represents whether the data instance is in the bag as shown in Fig 1." | In fact binary encoding of data instances in each bag is not possible in this scenario since data instances in ach bag are sampled with replacement thus it can be one or more data instances with the same index | We store indexes of data that are in each bag |
"We simplify the step of selecting C crossover parents by using a rank selection scheme where the offspring with the highest fitness are selected for crossover" | Rank selection scheme includes randomness, however in their implementation always c bags with highest fitness are chosen for corssover | In our implementation user can chose which selection scheme he wants to use "naive" or "rank" |
"we terminate the EvoBagging training after a certain number of iterations which have been determined from trial experiments." | There is no detailed information how number of iterations is established | We chose the same number of iteration for dataset as indicated in paper |
No information about parameters of classifier (e.g max depth) | While building classifier for specific bag authors limit max depth of classifier to number of features | we allow user to choose whether max depth of classifier should be fixed to number of features by introducing parameter classifier_restricted |
Mean test accuracy on Red wine dataset is reported to be 92.76% with standard deviation (0.013) (30 experiments) | By running provided implementation we obtained lower accuracy | Left box stands in line with accuracy obtained by our implementation [Notebook with results](implementation_comparison/implementation_comparison.ipynb) |
Models are trained for given number of iterations | Reported test accuracy comes from iteration where model obtained highest training accuracy - not always last iteration | Depends on the experiment |
Table 5 on page 8 | maximal tree depth is limited by number of features which is 8 for Pima dataset therefore it is not possible to obtain higher average depth using their script | we run both variants, one with restricted depth of classifier and one without |
"In each bag of the ensemble, we obtain the average bias of the respective individual learner for all the data samples in the test set." | Bias reduction is reported on train set | we report bias on trian set |
"To find the optimal number of bags for bagging, we run a search with an interval of 10 and select the one with the highest test classification metric" | In implementation interval of 5 i used between 5 and 100 | We use the same search space as in implementation |
Bag B will be mutated by replacing selected random samples given by the mutation size (MS ) from B with the same amount of random samples from Bc | It is not clear whether data instances used to replace samples are sampled with replacement or not. In implementation random sampling without replacement is done | We do the same as in original implementation thus random sampling without replacement |
Number of iterations for nbit parity problem is not specified | Iteration number is set to 20 | We fix number of iterations to 20 |
Voting rule experiment | Voting rule has impact on crossover part however while comparing two schemes of prediction model is trained using majority voting and accuracy on test set is reported based on majority voting and weighted voting | We implement experiment as it was implemented in repository as well as our experiment where training and evaluation is done using each of voting schemes |
