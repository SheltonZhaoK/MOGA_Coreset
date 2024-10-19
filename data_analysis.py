import os
import pandas as pd
import numpy as np

# process benchmarking with cca reference data | max 10 seeds ####
# seed = 4
# methods = ["sctransform", "harmony", "fastmnn", "cca"]
# result_dir = "/deac/csc/khuriGrp/zhaok220/thesis/output/results"
# output_dir = "/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results"

# all_methods_data = []
# for method in methods:
#     data_list = [pd.read_csv(os.path.join(result_dir, f"down_size10X_MOGAs_reference_{method}_benchmark_seed{seed}_100iterations.csv"))]
#     for result in os.listdir(result_dir):
#         if f"down_size10X_MOGAs_reference_{method}" in result:  # Filter files for the current method
#             data = pd.read_csv(os.path.join(result_dir, result))
#             if len(data) == len(data_list[0]):
#                 data_list.append(data)

#     concatenated_data = pd.concat(data_list[:10], ignore_index=True)
#     idx = concatenated_data.groupby(['Coreset Methods', 'Data'])['Test Accuracy'].idxmax()
#     data = concatenated_data.loc[idx]

# #     # ensemble_data = pd.read_csv(os.path.join(result_dir, f"reference_ensemble_training_{method}_100_iterations_11_solution.csv"))
# #     # idx = ensemble_data.groupby(['Coreset Methods', 'Data'])['Test Accuracy'].idxmax()
# #     # ensemble_data = ensemble_data.loc[idx]
# #     # data = pd.concat([data, ensemble_data], ignore_index=True)

#     map = data[data['Coreset Methods'] == 'Full'].set_index('Data')['Test Accuracy'].to_dict()
#     def calculate_performance_improvement(row):
#         full_accuracy = map.get(row['Data'], 0)
#         return row['Test Accuracy'] - full_accuracy

#     data['Performance Improvement'] = data.apply(calculate_performance_improvement, axis=1)
#     data['Integration Methods'] = method 
#     all_methods_data.append(data)  # Append the processed data for the current method

# final_data = pd.concat(all_methods_data, ignore_index=True)

# final_data.replace({"kCenterGreedy": "KCenterGreedy", "orthoPursuit":"OrthoPursuit", "giga": "Giga", "frankWolfe": "FrankWolfe",
#                     "importanceSampling": "ImportanceSampling", "uniformSampling": "UniformSampling", "kmeans": "Kmeans", "sctransform": "SCTransform",
#                     "cca": "CCA", "harmony": "Harmony", "fastmnn":"FastMNN"}, inplace=True)

# final_data.to_csv(os.path.join(output_dir, f"down_size10X_MOGAs_reference_all_benchmark_10seed_max.csv"), index=False)

# # process benchmarking with cca reference data | average 10 seeds ####
# seed = 4
# name = "sctransform"
# result_dir = "/deac/csc/khuriGrp/zhaok220/thesis/output/results"
# data_List = [pd.read_csv(f"/deac/csc/khuriGrp/zhaok220/thesis/output/results/reference_{name}_benchmark_seed{seed}_100iterations.csv")]
# for result in os.listdir(result_dir):
#     data = pd.read_csv(os.path.join(result_dir, result))
#     if len(data) == len(data_List[0]) and "cca" in result:
#         data_List.append(data)
# concatenated_data = pd.concat(data_List[:10], ignore_index=True)
# data = concatenated_data.groupby(['Coreset Methods', 'Data'])['Test Accuracy'].mean().reset_index()
# data.to_csv(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results", f"reference_{name}_benchmark_10seed_average.csv"))

#### process benchmarking with drug data | seed #7 ###

# result_dir = "/deac/csc/khuriGrp/zhaok220/thesis/output/results"
# output_dir = "/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results"
# data_list = [pd.read_csv(os.path.join(result_dir, f"drug_None_benchmark_seed0_100iterations.csv"))]
# for result in os.listdir(result_dir):
#     if "drug_None_benchmark_" in result:  # Filter files for the current method
#         data = pd.read_csv(os.path.join(result_dir, result))
#         if len(data) == len(data_list[0]):
#             data_list.append(data)

# concatenated_data = pd.concat(data_list[:10], ignore_index=True)
# idx = concatenated_data.groupby(['Coreset Methods', 'Data'])['Test Accuracy'].idxmax()
# data = concatenated_data.loc[idx]
# map = data[data['Coreset Methods'] == 'Full'].set_index('Data')['Test Accuracy'].to_dict()
# def calculate_performance_improvement(row):
#     full_accuracy = map.get(row['Data'], 0)
#     return row['Test Accuracy'] - full_accuracy
# data['Performance Improvement'] = data.apply(calculate_performance_improvement, axis=1)
# data.replace({"kCenterGreedy": "KCenterGreedy", "orthoPursuit":"OrthoPursuit", "giga": "Giga", "frankWolfe": "FrankWolfe",
#                     "importanceSampling": "ImportanceSampling", "uniformSampling": "UniformSampling", "kmeans": "Kmeans", "sctransform": "SCTransform"}, inplace=True)
# data.to_csv(os.path.join(output_dir, f"drug_all_benchmark_10seed_max.csv"), index=False)

### process drug data ####
# result_dir = "/deac/csc/khuriGrp/zhaok220/thesis/output/results"
# output_dir = "/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results"
# data_list = [pd.read_csv(os.path.join(result_dir, f"no_f1_MOGAs_drug_None_benchmark_seed0_100iterations.csv"))]
# for result in os.listdir(result_dir):
#     if "no_f1_MOGAs_drug_None_benchmark_seed" in result:  # Filter files for the current method
#         data = pd.read_csv(os.path.join(result_dir, result))
#         if len(data) == len(data_list[0]):
#             data_list.append(data)

# concatenated_data = pd.concat(data_list[:10], ignore_index=True)
# idx = concatenated_data.groupby(['Coreset Methods', 'Data'])['Test Accuracy'].idxmax()
# data = concatenated_data.loc[idx]
# map = data[data['Coreset Methods'] == 'Full'].set_index('Data')['Test Accuracy'].to_dict()
# def calculate_performance_improvement(row):
#     full_accuracy = map.get(row['Data'], 0)
#     return row['Test Accuracy'] - full_accuracy
# data['Performance Improvement'] = data.apply(calculate_performance_improvement, axis=1)
# # data.replace({"kCenterGreedy": "KCenterGreedy", "orthoPursuit":"OrthoPursuit", "giga": "Giga", "frankWolfe": "FrankWolfe",
# #                     "importanceSampling": "ImportanceSampling", "uniformSampling": "UniformSampling", "kmeans": "Kmeans", "sctransform": "SCTransform"}, inplace=True)
# data.to_csv(os.path.join(output_dir, f"nof1_drug_MOGA_10max.csv"), index=False)

### combine f1 moga and nf1 moga ###
# for name in ["drug", "reference"]:
    # no_f1_data = pd.read_csv(f"/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/no_f1_{name}_all_benchmark_10seed_max.csv")
    # no_f1_data = no_f1_data[no_f1_data['Coreset Methods'].isin(["MOGA_distance_nf1", "MOGA_balance_nf1", "MOGA_preserve_nf1", "MOGA_d+b"])]
    # no_f1_data["F1"] = "No_F1"

    # f1_data = pd.read_csv(f"/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/{name}_all_benchmark_10seed_max.csv")
    # f1_data = f1_data[f1_data['Coreset Methods'].isin(["MOGA_distance", "MOGA_balance", "MOGA_preserve", "MOGA_naive"])]
    # f1_data["F1"] = "Yes_F1"

    # data = pd.concat([no_f1_data, f1_data])
    # data.to_csv(os.path.join(output_dir, f"combine_f1_nf1_{name}_10max_MOGAs.csv"), index=False)
    # print(data["Coreset Methods"].value_counts())

seed = 4
methods = ["sctransform", "harmony", "fastmnn", "cca"]
result_dir = "/deac/csc/khuriGrp/zhaok220/thesis/output/results"
output_dir = "/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results"

all_methods_data = []
for method in methods:
    data_list = [pd.read_csv(os.path.join(result_dir, f"down_size10X_MOGAs_reference_{method}_benchmark_seed{seed}_100iterations.csv"))]
    for result in os.listdir(result_dir):
        if f"down_size10X_MOGAs_reference_{method}" in result:  # Filter files for the current method
            data = pd.read_csv(os.path.join(result_dir, result))
            if len(data) == len(data_list[0]):
                data_list.append(data)

    concatenated_data = pd.concat(data_list[:], ignore_index=True)
    idx = concatenated_data.groupby(['Coreset Methods', 'Data'])['Test Accuracy'].idxmax()
    data = concatenated_data.loc[idx]

    map = data[data['Coreset Methods'] == 'Full'].set_index('Data')['Test Accuracy'].to_dict()
    def calculate_performance_improvement(row):
        full_accuracy = map.get(row['Data'], 0)
        return row['Test Accuracy'] - full_accuracy

    data['Performance Improvement'] = data.apply(calculate_performance_improvement, axis=1)
    data['Integration Methods'] = method 
    all_methods_data.append(data)  # Append the processed data for the current method

final_data = pd.concat(all_methods_data, ignore_index=True)

final_data.replace({"kCenterGreedy": "KCenterGreedy", "orthoPursuit":"OrthoPursuit", "giga": "Giga", "frankWolfe": "FrankWolfe",
                    "importanceSampling": "ImportanceSampling", "uniformSampling": "UniformSampling", "kmeans": "Kmeans", "sctransform": "SCTransform",
                    "cca": "CCA", "harmony": "Harmony", "fastmnn":"FastMNN"}, inplace=True)

final_data.to_csv(os.path.join(output_dir, f"down_size10X_MOGAs_reference_all_benchmark_10seed_max.csv"), index=False)