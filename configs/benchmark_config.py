# -----------------------------------------------------------
# This script provides hyperparameter used in the experiments
#
# Author: Konghao Zhao
# Created: 2023-10-27
# Modified: 2023-10-30
# 
# -----------------------------------------------------------
class Configs:
    def __init__(self, seed, max_iter = 100, inputDir = None):
        self.configs = {
            "seed": seed,
            "test_size": 0.3,
            "inputDir": inputDir,
            # "outputDir": "../output",
            "outputDir": "/deac/csc/khuriGrp/wangy22/edcs/output", #fixme
            "cv": 5,
            "n_jobs": -1,
            "max_iter": max_iter,

            "DataProcessing":
            {
                "audit":
                {
                    "duplicateRow": True,
                    "duplicateCol": True,
                    "NaN": True,
                    "column_NaN": True,
                    "maxNA": 0.4 #0.25 -> 0-1265, 1-588, features-27 | 0.30 -> 0-1241, 1-563, features-62
                },

                "cleaning":
                {
                    "stdScale": 3,
                    "evaluation": False
                },

                "feature_selection":
                {
                    "correlationUpperBound": 0.98,
                    "scaleRange": [-1, 1],
                    "numVariableFeatures": 100,
                    # "pcaCriteria":0.98
                    "pcaCriteria":10
                },
            },

            "Coresets":
            {
                "ec":
                {
                    "n_splits": 10,
                    "pop_size": 100,
                    "max_points_in_core_set": 500
                },

                "logisticRegression":
                {
                    "penalty": ["l1", "l2", "elasticnet"],
                    "C": [0.1, 1, 10],
                    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                    "max_iter": [max_iter],
                    "random_state": [seed]
                },

                "decisionTree":
                {
                    "criterion": ["gini", "entropy", "log_loss"],
                    # "max_depth": [2, 3, 5, 10, 20],
                    "min_samples_leaf": [5, 10, 20, 50, 100],
                    "random_state": [seed]

                },
                
                "randomForest":
                {
                    "criterion": ["gini", "entropy", "log_loss"],
                    # "max_depth": [2, 3, 5, 10, 20],
                    'max_features': ["sqrt", "log2"],
                    "min_samples_leaf": [5, 10, 20, 50, 100],
                    "random_state": [seed]
                },

                "xgb":
                {
                    # 'max_depth': [2, 3, 5, 10, 20],
                    'learning_rate': [0.1, 0.01, 0.001],
                    'subsample': [0.5, 0.7, 1]
                },

                "naive":
                {
                    "random_state": [seed]
                },

            }
        }