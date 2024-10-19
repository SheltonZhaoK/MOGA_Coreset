import os, sys
import pandas as pd

from configs.baseline_config import Arguments
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from data import make_data
from collections import Counter
from sklearn.linear_model import LogisticRegression

def format_gridSearch_results(searcher, outputFile):
    cv_results = searcher.cv_results_
    mean_test_scores = cv_results['mean_test_score']
    params = cv_results['params']
    std_test_scores = cv_results['std_test_score']
    fit_times = cv_results['mean_fit_time']
    std_fit_times = cv_results['std_fit_time']

    sorted_results = sorted(zip(mean_test_scores, params, std_test_scores, fit_times, std_fit_times), key=lambda x: x[0], reverse=True)

    if not os.path.exists("../output"):
        os.makedirs("../output")
    with open(outputFile, 'w') as f:
        for score, param, std, fit_time, std_fit_time in sorted_results:
            f.write(f"Score: {score:.3f} +- {std:.3f}, Fit Time: {fit_time:.3f} +- {std_fit_time:.3f}, Parameters: {param}\n")
        print(f"{searcher.best_estimator_} search results saved to {outputFile}")

def read_data(dataFile, labelfile):
    data = pd.read_csv(dataFile, index_col=None, low_memory=False)
    data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    label = pd.read_csv(labelfile, index_col=None, low_memory=False)
    label.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    return data, label

def main(inputDir, outputDir, label2balance, label2predict, config):
    report = pd.DataFrame(columns = ["Datasets", "Training Size", "Accuracy"])
    for dataset in os.listdir(inputDir):
        dataFile = os.path.join(inputDir, dataset, "fastmnn_umap.csv")
        labelFile = os.path.join(inputDir, dataset, "fastmnn_labels.csv")
        data, annotations = read_data(dataFile, labelFile)
        
        train_data, test_data, train_annotations, test_annotations = train_test_split(data, annotations, 
                                                                            test_size=config["test_size"],
                                                                            random_state=config["seed"])

        train_x, train_y, train_ix = make_data(train_data, train_annotations, label2balance, label2predict, balance = True)
        # print(train_annotations.loc[train_ix][label2balance].value_counts())
        test_x, test_y, test_ix = make_data(test_data, test_annotations, label2balance, label2predict, balance = False)
        # print(test_annotations.loc[test_ix][label2balance].value_counts())

        lr = GridSearchCV(LogisticRegression(), param_grid=config["Classifiers"]["LR"], cv=5, scoring="accuracy", n_jobs = -1, refit=True)
        lr.fit(train_x, train_y[:,0])
        format_gridSearch_results(lr, os.path.join(outputDir, f"{dataset}_LR_gridSearch.txt"))

        lr_prediction = lr.predict(test_x)
        accuracy = accuracy_score(lr_prediction, test_y[:,0])

        report.loc[len(report)] = [dataset, len(train_data) , round(accuracy,4)]
    report.to_csv(os.path.join(outputDir, "baseline_performance.csv"))
        
if __name__ == "__main__":
    inputDir = "../data/cellLine_integration"
    outputDir = "../output/baseline_performance"
    label2balance, label2predict = "CELL_LINE", "CELL_LINE"
    config = Arguments().arguments
    main(inputDir, outputDir, label2balance, label2predict, config)