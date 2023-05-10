import captum.attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
import seaborn as sns


def calculate_integrated_gradient(ig, input, baseline, target, n_variables, n):
    attributions, delta = ig.attribute(input, baseline, target=target, return_convergence_delta=True)
    attributions = attributions.numpy().reshape(n, 13, 237)
    variable_names = pd.read_csv("../LSTM/Additional_data/variables_names.csv")
    variable_names.columns = ["var_num", "var_name"]
    df_attributions = pd.DataFrame(attributions.mean(axis=0), columns=variable_names.var_name)
    features_max_attribution = df_attributions.max().abs().sort_values(ascending=False).iloc[:n_variables]

    # plot max attribution
    plt.subplots(figsize=(15, 6))
    plt.bar(features_max_attribution.index, features_max_attribution.values)
    plt.show()

    # plot attribution change over time
    plt.subplots(figsize=(15, 8))
    for column in features_max_attribution.index:
        plt.plot(df_attributions[column], label=column)
    plt.legend()
    plt.show()

    plt.subplots(figsize=(15, 8))
    ax = sns.heatmap(df_attributions.loc[:, features_max_attribution.index], linewidth=0.5)
    plt.show()

    return attributions, delta


def calculate_feature_permutation(fp, input, target, n_variables, n):
    attributions = fp.attribute(input, target=target)
    attributions = attributions.numpy().reshape(n, 13, 237)
    variable_names = pd.read_csv("../LSTM/Additional_data/variables_names.csv")
    variable_names.columns = ["var_num", "var_name"]
    df_attributions = pd.DataFrame(attributions.mean(axis=0), columns=variable_names.var_name)
    features_max_attribution = df_attributions.max().abs().sort_values(ascending=False).iloc[:n_variables]

    # plot max attribution
    plt.subplots(figsize=(15, 6))
    plt.bar(features_max_attribution.index, features_max_attribution.values)
    plt.show()

    # plot attribution change over time
    plt.subplots(figsize=(15, 8))
    for column in features_max_attribution.index:
        plt.plot(df_attributions[column], label=column)
    plt.legend()
    plt.show()

    plt.subplots(figsize=(15, 8))
    ax = sns.heatmap(df_attributions.loc[:, features_max_attribution.index], linewidth=0.5)
    plt.show()

    return attributions

def calculate_feature_ablation(fa, input, target, n_variables, n):
    attributions = fa.attribute(input, target=target, show_progress=True)
    attributions = attributions.numpy().reshape(n, 13, 237)
    variable_names = pd.read_csv("../LSTM/Additional_data/variables_names.csv")
    variable_names.columns = ["var_num", "var_name"]
    df_attributions = pd.DataFrame(attributions.mean(axis=0), columns=variable_names.var_name)
    features_max_attribution = df_attributions.max().abs().sort_values(ascending=False).iloc[:n_variables]

    # plot max attribution
    plt.subplots(figsize=(15, 6))
    plt.bar(features_max_attribution.index, features_max_attribution.values)
    plt.show()

    # plot attribution change over time
    plt.subplots(figsize=(15, 8))
    for column in features_max_attribution.index:
        plt.plot(df_attributions[column], label=column)
    plt.legend()
    plt.show()

    plt.subplots(figsize=(15, 8))
    ax = sns.heatmap(df_attributions.loc[:, features_max_attribution.index], linewidth=0.5)
    plt.show()

    return attributions
