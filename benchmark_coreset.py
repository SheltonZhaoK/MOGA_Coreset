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
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score
from math import dist

from prototypes import bayesian_core_set as bc
from prototypes import coreset_pipeline as cp
from visualizer import plot_coreset

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
from coreset_methods.moga_no_f1 import MOGA as MOGA_nf1

from configs.benchmark_config import Configs
def main(report, configs, args):  
    for data in os.listdir(configs["inputDir"]):
        # data = "C1_LLU_rsem" # fitness plot python3 benchmark_coreset.py -d reference -i cca -s 10 "C1_LLU_rsem"
        # data = "10X_NCI_cellranger2.0" # scatter plot python3 benchmark_coreset.py -d reference -i fastmnn -s 5 "10X_NCI_cellranger2.0"
        # data = "ICELL8_SE_kallisto"
        # data = "mate1"
        # data = "C1_LLU_featureCounts"
        output_dir = os.path.join(configs["outputDir"])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_data_og, train_labels_og, test_data_og, test_labels_og = Data().get_data(data, configs, args)
        original_indices = list(train_data_og.index)

        #====================== Benchmarking ======================
        train_data = train_data_og.reset_index(drop=True).values
        train_labels = train_labels_og.reset_index(drop=True).values

        coreset_methods = {
                        "Full": Full(),
                        "EvoCore": EvoCore(estimator=clone(RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"])), pop_size=configs["Coresets"]["ec"]["pop_size"], max_generations=configs["max_iter"],
                                    max_points_in_core_set=configs["Coresets"]["ec"]["max_points_in_core_set"], n_splits=configs["Coresets"]["ec"]["n_splits"], random_state=configs["seed"]),
                        
                        "MOGA_naive": MOGA(train_data, train_labels, RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"]), "f1_weighted", numGen=configs["max_iter"], name = "MOGA_naive"),
                        "MOGA_distance": MOGA(train_data, train_labels, RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"]), "f1_weighted",numGen=configs["max_iter"], distance = True, name = "MOGA_distance"),
                        "MOGA_balance": MOGA(train_data, train_labels, RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"]),  "f1_weighted", numGen=configs["max_iter"],classDist = "balance", name = "MOGA_balance"),
                        "MOGA_preserve": MOGA(train_data, train_labels, RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"]),  "f1_weighted", numGen=configs["max_iter"], classDist = "preserve", name = "MOGA_preserve"),
                        
                        "MOGA_distance_nf1": MOGA_nf1(train_data, train_labels, RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"]), "f1_weighted",numGen=configs["max_iter"], distance = True, name = "MOGA_distance"),
                        "MOGA_balance_nf1": MOGA_nf1(train_data, train_labels, RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"]),  "f1_weighted", numGen=configs["max_iter"],classDist = "balance", name = "MOGA_balance"),
                        "MOGA_preserve_nf1": MOGA_nf1(train_data, train_labels, RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"]),  "f1_weighted", numGen=configs["max_iter"], classDist = "preserve", name = "MOGA_preserve"),
                        "MOGA_d+b": MOGA_nf1(train_data, train_labels, RidgeClassifier(random_state=configs["seed"], max_iter=configs["max_iter"]),  "f1_weighted", numGen=configs["max_iter"], distance = True, classDist = "balance", name = "MOGA_d+b"),
                        }

        train_indices = {}
        coreset_methods_name = list(coreset_methods.keys())
        for name in coreset_methods_name:
            method = coreset_methods[name]
            estimator = LogisticRegression(random_state=configs["seed"], max_iter=configs["seed"])
            start = time.time()
            core_x, core_y, indices = method.fit_transform(train_data, train_labels)

            if "MOGA" in name:
                core_x = core_x[0]
                core_y = core_y[0]
                indices = indices[0]

            train_indices[name] = indices

            fittingTime = round(time.time() - start, 4)
            if name == "MOGA_naive":
                coreSize = len(indices)
                coreset_methods.update({ 
                        "KCenterGreedy": KCenterGreedySampling(coreSize, euclidean_dist),
                        "OrthoPursuit": OrthoPursuit(train_data.T, train_data.sum(axis=0), configs["max_iter"]),
                        "Giga": GIGA(train_data.T, train_data.sum(axis=0), configs["max_iter"]),
                        "FrankWolfe": FrankWolfe(train_data.T, train_data.sum(axis=0), configs["max_iter"]),
                        "ImportanceSampling": ImportanceSampling(train_data, train_labels, configs["max_iter"]),
                        "UniformSampling": UniformSampling(coreSize),
                        "Kmeans": KMeansSampling(coreSize, configs["seed"])})
                # coreset_methods_name.extend(["KCenterGreedy", "OrthoPursuit", "Giga", "GrankWolfe","ImportanceSampling","UniformSampling","Kmeans"])
                coreset_methods_name.extend(["UniformSampling","Kmeans"])

            # pd.DataFrame([original_indices[i] for i in indices], columns=['Index']).to_csv(os.path.join(output_dir,"indices", f"{name}_indices_{configs['max_iter']}.csv"), index=False) #extract coreset

            core_train = train_data_og.loc[[original_indices[i] for i in indices]]
            core_train['Labels'] = core_y  # Add the labels as a column named 'Labels'
            core_train.to_csv(os.path.join(output_dir, "selected_data" ,f"{data}_train_{args.i}_{name}_seed{configs['seed']}.csv")) #fixme 
            
            test_df = test_data_og.copy()
            test_df["Labels"] = test_labels_og["label"]
            test_df.to_csv(os.path.join(output_dir, "selected_data" ,f"{data}_test_{args.i}_{name}_seed{configs['seed']}.csv"))

            if len(np.unique(core_y)) == len(np.unique(train_labels)):
                model = estimator.fit(core_x, core_y)
                pred_labels = model.predict(core_x)
                accuracy_train = round(accuracy_score(pred_labels, core_y), 4)

                pred_labels = model.predict(test_data_og)
                accuracy_test = round(accuracy_score(pred_labels, test_labels_og), 4)
            else:
                accuracy_train = 0
                accuracy_test = 0

            core_x = pd.DataFrame(core_x)
            core_y = pd.DataFrame(core_y)
            train_class_ratio = ':'.join(f"{size}" for clas, size in core_y.value_counts().items())
            test_class_ratio =  ':'.join(f"{size}" for clas, size in test_labels_og.value_counts().items())

            train_minority_ratio, train_minority_difference = minority_class_ratio(core_y)
            test_minority_ratio, test_minority_difference = minority_class_ratio(core_y)

            report.loc[len(report)] = [data, configs["seed"], name, core_x.shape[0], train_class_ratio, train_minority_ratio, test_data_og.shape[0], test_class_ratio, test_minority_ratio, accuracy_train, accuracy_test, fittingTime]
            # report.to_csv(os.path.join("../output/results", f"down_size10X_MOGAs_{args.d}_{args.i}_benchmark_seed{configs['seed']}_{configs['max_iter']}iterations.csv")) 
            print(report)
            
        # plot_coreset(train_data, train_labels, train_indices, configs, args, data)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmarking coreset methods')
    parser.add_argument('-d', choices = ["reference", "drug", "custom"], type=str, help='specify which dataset to use', required=True)
    parser.add_argument('-s', type=int, help='specify which seed to use', required=True)
    parser.add_argument('-i', type=str, default = "None", help='specify which integration method to use if -d == reference')
    args = parser.parse_args()

    max_iter = 100
    if args.d == "drug":
        inputDir = "../data/ligand_discovery"

    elif args.d == "reference":
        inputDir = "../data/cellLine_integration"

    elif args.d == "custom":
        inputDir = "/deac/csc/khuriGrp/wangy22/edcs/data"

    report = pd.DataFrame(columns=["Data","Seed","Coreset Methods", "Train Size", "Train Class Ratio", "Minority Ratio", "Test Size", "Test Class Ratio", "Minority Ratio",
                                    "Train Accuracy", "Test Accuracy", "Coreset Selection Time"])

    configs = Configs(args.s, max_iter, inputDir).configs
    set_seed(configs["seed"])
    main(report, configs, args)