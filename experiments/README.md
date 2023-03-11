# Experiments description

This directory contains folder with scripts to conduct experiments as well as results in *.csv* format

## *Optimal_bag_number*

As stated in paper:

*"To facilitate fair comparisons with other ensemble models, the number of bags in EvoBagging and all baselines is 
equal to the optimal number of bags for bagging"*

To obtain optimal number of bags we train BaggingClassifier for each dataset (mnist, breast_cancer, abalone,
red_wine, pima, car, tic-tac-toe, ionosphere, churn, flare, ring, two-spiral) with number of bags equal to 5,10,15,
...,95. We use 5-fold cross validation with accuracy metric and for each dataset we return number of bags that resulted
in highest cross validation score. Results are saved in *results_optimal_bags.csv*

Note : in implementation by paper authors they used 3-fold cross valdation

## *Evo_bagging_grid_search*

As stated in paper:

*"We determine other hyper-parameters (G, M, MS , and K) by evaluating accuracy (5-fold cross-validation) 
on the training set as follows:*
- *G ∈ {10%, 15%, 20%, 25%, 30%} of N* 
- *M ∈ {5%, 6%, 7%, 8%, 9%, 10%} of N*
- *MS ∈ {5%, 10%} of S* 
- *K ∈ {1000, 2000, 3000, ..., 20000}"*

We perform described hyper-parameter grid search optimization. Random states for train/test split in each step of 
cross validation ares set to [1, 10, 20, 50, 70]. Since way of choosing number of iterations for Evolutionary Bagging
algorithm is not described in paper we set it to values given in paper. Namely:


|    Dataset    | number of iterations |
|:-------------:|:--------------------:|
| mnist         | 20                   |
| breast_cancer | 20                   |
| abalone       | 35                   |
| red_wine      | 25                   |
| pima          | 15                   |
| car           | 30                   |
| tic-tac-toe   | 20                   |
| ionosphere    | 20                   |
| churn         | 35                   |
| flare         | 40                   |
| two-spiral    | 40                   |

Results for each set of parameters are saved in *evo_bagging_grid_search.csv*. Python script
*create_json_with_best_params.py* creates dictionary *params.json* with best set of parameters. In case of there are
multiple best sets of parameters, random one is chosen.

## *Baseline*

Script in this folder trains 5 models (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
XGBoost, EvoBagging) on each dataset with 20% train/test split. Hyper-parameters for EvoBagging algorithm is taken from 
*params.json* in Evo_bagging_grid_search directory. Number of estimators for each classifier is taken from 
*results_optimal_bags.csv* file in optimal_bag_number directory. Training for each model is repeated 30 times 
and mean train and test accuracy is saved along with standard deviation. Results are saved in *training_results.csv*

## *nbit_parity*

Classification accuracy on the 6-bit and 8-bit parity problems with varying number of bags is computed. We consider 
20% train/test split and report mean train and test accuracy along with its standard deviation.
For each number of bags we rerun each model training 30 times. Considered models are:
BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, EvoBagging. Each EvoBagging training is done 
with following parameters as stated in paper:
- G: 0.2 of N
- M: 0.1 of N
- MS: 1 
- K: 500
- number of iterations: 50 

## *influence_of_hyper_parameters*

Script conduct experiment on the sensitivity of EvoBagging on changes in the hyper-parameters including maximum bag size S , 
generation gap G, number of mutated bags M, mutation size MS , and the bag size control K. These experiments
are run on the Pima dataset with its best configuration resulted in grid search. Number of bags comes from optimal_bag_number
experiment and number of iterations is set to 15. This experiment follows experiments from section 4.4 from paper

## *bias_reduction*

Script runs experiments from section 4.5.1 from paper. It measures the average bias in classification task 
across all bags by averaging percentage of wrongly classified data instances. Script creates two files:
- bias_reduction_graph.csv - for Red wine, Breast cancer, Pima and Mnist dataset it returns average bias in each
iteration for iteration number up to 10 
- bias_reduction_table.csv - for all datasets it returns average bias in first iteration and last iteration as well as
percentage reduction of bias.

## *diveristy_between_evolved_bags*

Script runs experiments from section 4.5.3 from paper. It returns six diversity measures between 
individual learners for EvoBagging and bagging for 4 datasets (red_wine, ring, mnist, car) for test sets obtained with
20% train/test split. Considered diversity measures are:
- Q statistics
- Disagreement
- Double fault
- Kohavi-Wolpert variance
- Entropy
- Generalized diversity

Results are saved in *diversity_between_bags.csv*.

## *voting_rule*

Script runs experiments from section 4.5.6 from paper. This experiment checks impact of voting rule on test accuracy at
each iteration. It is conducted on Pima dataset. It creates two files:
- voting_influence.csv - which stores results of experiment conducted in the same manner as it was implemented by authors
of paper
- voting_influence_corrected.csv - stores results of experiment conducted in a way incorporating fact that voting rule has
impact on training. In authors implementation training is done using majority voting and in each iteration accuracy on test
set using majority voting and using weighted voting is reported. However as voting scheme has impact on prediction therefore
it has impact on bag fitness value which is used in crossoover operation we run experiments to see whether change of voting rule
used in training influences metric.

Formula for weighted voting can be found in paper.