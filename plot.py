import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd

# default 6.4 width & 4.8 height
matplotlib.rcParams['svg.fonttype'] = 'none'
newColor = [
    "#D6594C", "#4D7787", "#AAB083", "#E88C1F", "#7D8995", "#E9BD27", 
    "#7262ac", "#046586", "#28A9A1", "#C9A77C", "#F4A016",'#F6BBC6','#E71F19' #backup colors
]
# plt.rcParams["font.family"] = "Times"

# reference benechmarking
# names = ["cca", "sctransform", "fastmnn"]
# names = ["sctransform"]
# values = ["Train Accuracy", "Test Accuracy", "Minority Ratio", "Coreset Selection Time", "Train Size"]
# for name in names:
#     data = pd.read_csv(f"/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/reference_{name}_benchmark_10seed_max.csv")
#     fig, axs = plt.subplots(3, 2, figsize=(6.4*2, 4.8*3))  # Adjusted figsize to accommodate 4 subplots
#     axs = axs.flatten()  # Flatten the 2x2 grid to easily iterate over it
    
#     for i, value in enumerate(values):
#         # Plot each value in its respective subplot
#         sns.boxplot(data=data, x="Coreset Methods", y=value, ax=axs[i], 
#                     order =  ["Full", "uniformSampling", "EvoCore", "MOGA_naive", "MOGA_distance", "MOGA_balance", "MOGA_preserve", "kCenterGreedy", "orthoPursuit", "giga", "frankWolfe","importanceSampling", "kmeans"],
#                     showmeans=True,
#                     meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red","markersize":"2"})
#         fig.autofmt_xdate(rotation=45)

#         if value in ["Train Accuracy", "Test Accuracy"]:
#             axs[i].set_ylim((0.2, 1.05)) #sctransform
#             # axs[i].set_ylim((0, 1.05)) #fastmnn

#         if i == 0:
#             axs[i].axhline(y=data[data['Coreset Methods'] == 'Full']['Train Accuracy'].mean(), color='r', linestyle='--')
#         elif i == 1:
#             axs[i].axhline(y=data[data['Coreset Methods'] == 'Full']['Test Accuracy'].mean(), color='r', linestyle='--')

#         axs[i].spines['top'].set_visible(False)
#         axs[i].spines['right'].set_visible(False)

#     plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"reference_{name}_benchmark_10seed_max.svg"))
#     plt.close()

# drug benechmarking
# data = pd.read_csv("/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/drug_benchmark_seed7.csv")
# values = ["Train Accuracy","Minority Ratio", "Test Accuracy", "Coreset Selection Time"]
# for value in values:
#     fig = plt.figure(figsize = (6.4*2, 9.6))
#     sns.boxplot(data=data, x="Coreset Methods", y=value)
#     fig.autofmt_xdate(rotation=45)
#     # if value == "Train Accuracy" or "Test Accuracy":
#     #     plt.ylim((0.45,1.05))
#     plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"drug_benchmark_seed7_{value}.svg"))

# names = ["sctransform"]
# values = ["Test Accuracy"]
# for name in names:
#     for value in values:
#         fig = plt.figure(figsize = (6.4, 9.6))
#         data = pd.read_csv(f"/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/reference_{name}_benchmark_10seed_average.csv")
#         sns.boxplot(data=data, x="Coreset Methods", y=value)
#         fig.autofmt_xdate(rotation=45)
#         plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"reference_{name}_benchmark_10seed_average.svg"))

# # figure 2
# data = pd.read_csv("/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/reference_all_benchmark_10seed_max.csv")
# data = data[data['Coreset Methods'].isin(['MOGA_naive', 'MOGA_naive_ensemble'])]
# plt.figure(figsize=(6.4, 4.8))
# # ax = sns.barplot(data=data, x="Coreset Methods", y="Test Accuracy", order =  ["MOGA_naive", "MOGA_naive_ensemble"], 
# #                  hue = "Integration Methods",errorbar = "sd", palette = colors2)
# ax = sns.barplot(data=data, x="Integration Methods", y="Test Accuracy", order =  ["CCA", "FastMNN", "Harmony", "SCTransform"], 
#                  hue ="Coreset Methods",errorbar = "sd", palette = newColor)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure2.svg"))
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure2.png"))

# # figure 4
# data = pd.read_csv("/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/reference_all_benchmark_10seed_max.csv")
# data = data[~data['Coreset Methods'].isin(['MOGA_naive_ensemble'])]
# fig, axs = plt.subplots(2, 1, figsize=(6.4*2, 4.8*2))
# axs = axs.flatten() 
# for i, y in enumerate(["Test Accuracy", "Minority Ratio"]):
#     axs[i] = sns.boxplot(data=data, x="Integration Methods", y=y, order =  ["Harmony", "FastMNN", "CCA", "SCTransform"], ax=axs[i],
#                 hue ="Coreset Methods", palette = newColor, showmeans=True, 
#                 hue_order = ["Full", "EvoCore", "MOGA_naive", "MOGA_distance", "MOGA_balance", "MOGA_preserve", "UniformSampling","ImportanceSampling", "Kmeans", "KCenterGreedy", "OrthoPursuit", "Giga", "FrankWolfe", ],
#                 meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red","markersize":"2"}, whiskerprops={'color':'black'}, capprops={'color':'black'})
#     axs[i].spines['top'].set_visible(False)
#     axs[i].spines['right'].set_visible(False)
#     if i == 1:
#         axs[i].set_ylim((0, 0.6)) 

# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure4.svg"))
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure4.png"))

# # figure 5
# data = pd.read_csv("/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/drug_all_benchmark_10seed_max.csv")
# data = data[data['Coreset Methods'].isin(['Full', "MOGA_naive", "MOGA_distance", "MOGA_balance", "MOGA_preserve"])]
# methods = ["MOGA_naive", "MOGA_distance", "MOGA_balance", "MOGA_preserve"]
# fig, axs = plt.subplots(2, 2, figsize=(6.4*2, 4.8*2))
# axs = axs.flatten() 
# for i, method in enumerate(methods):
#     axs[i] = sns.barplot(data=data, x="Coreset Methods", y="Test Accuracy", order =  ["Full", method], ax=axs[i], palette = newColor, hue = "Data")
#     axs[i].spines['top'].set_visible(False)
#     axs[i].spines['right'].set_visible(False)
#     axs[i].set_ylim((0, 0.7)) 

# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure5.svg"))
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure5.png"))

# #figure6
# data = pd.read_csv("/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/reference_all_benchmark_10seed_max.csv")
# reference_data = data[data['Coreset Methods'].isin(["MOGA_naive", "MOGA_distance", "MOGA_balance", "MOGA_preserve"])]
# fig = plt.figure(figsize=(6.4*2, 4.8))
# ax = sns.boxplot(data=reference_data, x="Data", y="Performance Improvement", hue = "Integration Methods", palette = newColor, showmeans=True, 
#             meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red","markersize":"2"}, whiskerprops={'color':'black'}, capprops={'color':'black'})
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# fig.autofmt_xdate(rotation=45)
# plt.tight_layout()
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure6.svg"))
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure6.png"))

# #figure7
# data = pd.read_csv("/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/drug_all_benchmark_10seed_max.csv")
# reference_data = data[data['Coreset Methods'].isin(["MOGA_naive", "MOGA_distance", "MOGA_balance", "MOGA_preserve"])]
# fig = plt.figure(figsize=(6.4*2, 4.8))
# ax = sns.boxplot(data=reference_data, x="Data", y="Performance Improvement", palette = newColor, showmeans=True, 
#             meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red","markersize":"2"}, whiskerprops={'color':'black'}, capprops={'color':'black'})
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# fig.autofmt_xdate(rotation=45)
# plt.tight_layout()
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure7.svg"))
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", f"Figure7.png"))


# combine reference and drug performance improvement
# data1 = pd.read_csv("/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/reference_all_benchmark_10seed_max.csv")
# reference_data1 = data1[data1['Coreset Methods'].isin(["MOGA_naive", "MOGA_distance", "MOGA_balance", "MOGA_preserve"])]
# data2 = pd.read_csv("/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/drug_all_benchmark_10seed_max.csv")
# drug_data = data2[data2['Coreset Methods'].isin(["MOGA_naive", "MOGA_distance", "MOGA_balance", "MOGA_preserve"])]
# fig, axs = plt.subplots(2, 1, figsize=(6.4*2, 4.8*2))
# sns.boxplot(data=reference_data1, x="Data", y="Performance Improvement", hue="Integration Methods", palette=newColor, showmeans=True,
#             meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red","markersize":"2"}, whiskerprops={'color':'black'}, capprops={'color':'black'}, ax=axs[0])
# axs[1].spines['top'].set_visible(False)
# axs[1].spines['right'].set_visible(False)
# axs[1].tick_params(axis='x', rotation=45)
# sns.boxplot(data=drug_data, x="Data", y="Performance Improvement", palette=newColor, showmeans=True, 
#             meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red","markersize":"2"}, whiskerprops={'color':'black'}, capprops={'color':'black'}, ax=axs[1])
# axs[0].spines['top'].set_visible(False)
# axs[0].spines['right'].set_visible(False)
# axs[0].tick_params(axis='x', rotation=45)
# plt.tight_layout()
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", "Combined_Figure6_Figure7.svg"))
# plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", "Combined_Figure6_Figure7.png"))

# combine reference and drug performance improvement
fig, axs = plt.subplots(2, 2, figsize=(6.4*4, 4.8*2))
axs = axs.flatten()
for i, name in enumerate(["drug", "reference"]):
    data = pd.read_csv(f"/deac/csc/khuriGrp/zhaok220/thesis/output/integrated_results/combine_f1_nf1_{name}_10max_MOGAs.csv")
    if name == "drug":
        ax = sns.barplot(data=data, x="Data", y="Performance Improvement", palette = newColor, hue = "Coreset Methods", ax=axs[0],
        hue_order = ["MOGA_naive","MOGA_distance", "MOGA_balance", "MOGA_preserve", "MOGA_distance_nf1", "MOGA_balance_nf1", "MOGA_preserve_nf1", "MOGA_d+b"])
        ax = sns.boxplot(data=data, x="Data", y="Performance Improvement", hue="F1", palette=newColor, showmeans=True, meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red","markersize":"2"}, whiskerprops={'color':'black'}, capprops={'color':'black'}, ax=axs[1])
    else:
        ax = sns.boxplot(data=data, x="Data", y="Performance Improvement", hue="Coreset Methods", palette=newColor, showmeans=True, meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red","markersize":"2"}, whiskerprops={'color':'black'}, capprops={'color':'black'},
        hue_order = ["MOGA_naive","MOGA_distance", "MOGA_balance", "MOGA_preserve", "MOGA_distance_nf1", "MOGA_balance_nf1", "MOGA_preserve_nf1", "MOGA_d+b"], ax=axs[2])

        ax = sns.boxplot(data=data, x="Data", y="Performance Improvement", hue="F1", palette=newColor, showmeans=True, meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red","markersize":"2"}, whiskerprops={'color':'black'}, capprops={'color':'black'}, ax=axs[3])
    
for i in range(4):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", "f1_comparison_moga.svg"))
plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/figures", "f1_comparison_moga.png"))