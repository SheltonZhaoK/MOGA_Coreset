import os, warnings, sys, traceback, scipy, time, multiprocessing, argparse

import pandas as pd
import numpy as np
import lazygrid as lg

# from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
# warnings.filterwarnings(action='ignore', category=FutureWarning)
from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score
from math import dist


from prototypes import bayesian_core_set as bc
from prototypes import coreset_pipeline as cp
from visualizer import plot_coreset, plot_fitness_scatter

from collections import Counter


from data import *
from Data import Data
from utilities import set_seed, euclidean_dist, minority_class_ratio

from coreset_methods.sampling import ImportanceSampling, UniformSampling, BalancedUniformSampling
from coreset_methods.evoCore import EvoCore
from coreset_methods.full import Full
from coreset_methods.frankWolfe import FrankWolfe
from coreset_methods.giga import GIGA
from coreset_methods.hilbert import Hilbert
from coreset_methods.bpsvi import SparseVICoreset
from coreset_methods.orthoPursuit import OrthoPursuit
from coreset_methods.kmeans import KMeansSampling, KCenterGreedySampling
from coreset_methods.moga import MOGA

from configs.benchmark_config import Configs
['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 
'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 
'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 
'jaccard_weighted', 'matthews_corrcoef', 'max_error', 'mutual_info_score', 'neg_brier_score', 'neg_log_loss', 
'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 
'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_negative_likelihood_ratio', 
'neg_root_mean_squared_error', 'normalized_mutual_info_score', 'positive_likelihood_ratio', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']
def main(report, configs, args, solutionSize):
    
    for data in os.listdir(configs["inputDir"]):
        output_dir = os.path.join(configs["outputDir"])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_data, train_labels, test_data, test_labels = Data().get_data(data, configs)
        original_indices = list(train_data.index)

        #====================== Benchmarking ======================
        train_data = train_data.reset_index(drop=True).values
        train_labels = train_labels.reset_index(drop=True).values
        
        method = MOGA(train_data, train_labels, RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"]),
                      "f1_weighted", sizePop=100, solutionSize=solutionSize, numGen=100, name="MOGA_naive")
        start = time.time()
        core_xs, core_ys, indices = method.fit_transform(train_data, train_labels)
        fittingTime = round(time.time() - start, 4)

        
        classifiers = [LogisticRegression(random_state=configs["seed"], max_iter=configs["max_iter"]) 
               for _ in range(len(core_xs))]

        # Fit each classifier on its corresponding core set
        for i, clf in enumerate(classifiers):
            clf.fit(core_xs[i], core_ys[i])

        # Predict with each classifier and store predictions
        predictions = []
        for clf in classifiers:
            predictions.append(clf.predict(train_data))  # Use test_data for test predictions
        predictions = np.array(predictions)
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(np.unique(train_labels))).argmax(), axis=0, arr=predictions)
        accuracy_train = round(accuracy_score(train_labels, majority_votes),4)

        test_predictions = []
        for clf in classifiers:
            test_predictions.append(clf.predict(test_data))
        test_predictions = np.array(test_predictions)
        test_majority_votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(np.unique(test_labels))).argmax(), axis=0, arr=test_predictions)
        accuracy_test = round(accuracy_score(test_labels, test_majority_votes),4)
    
        core_x = pd.DataFrame(core_xs[0])
        core_y = pd.DataFrame(core_ys[0])
        train_class_ratio = ':'.join(f"{size}" for clas, size in core_y.value_counts().items())
        test_class_ratio =  ':'.join(f"{size}" for clas, size in test_labels.value_counts().items())

        train_minority_ratio, train_minority_difference = minority_class_ratio(core_y)
        test_minority_ratio, test_minority_difference = minority_class_ratio(core_y)

        report.loc[len(report)] = [data, configs["seed"], "MOGA_naive_ensemble", core_x.shape[0], train_class_ratio, train_minority_ratio, test_data.shape[0], test_class_ratio, test_minority_ratio, accuracy_train, accuracy_test, fittingTime]
        report.to_csv(os.path.join("../output/results", f"{args.d}_ensemble_training_fastmnn_{configs['max_iter']}_iterations_{solutionSize}_solution.csv"))
        print(report)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmarking coreset methods')
    parser.add_argument('-d', choices = ["reference", "drug"], type=str, help='specify which dataset to use', required=True)
    args = parser.parse_args()

    max_iter = 100
    solutionSize = 11
    if args.d == "drug":
        inputDir = "../data/ligand_discovery"
    else:
        inputDir = "../data/cellLine_integration"

    report = pd.DataFrame(columns=["Data","Seed","Coreset Methods", "Train Size", "Train Class Ratio", "Minority Ratio", "Test Size", "Test Class Ratio", "Minority Ratio",
                                    "Train Accuracy", "Test Accuracy", "Coreset Selection Time"])
    
    for seed in range(30):
        configs = Configs(seed+1, max_iter, inputDir).configs
        set_seed(configs["seed"])
        main(report, configs, args, solutionSize)