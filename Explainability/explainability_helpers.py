import captum.attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
import seaborn as sns
import torch
import scipy



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


def run_lstm_for_lime(input, network=torch.load("../LSTM/Final_models/lstm.pickle"), batch_n=12800, device="cpu"):
    """
    Function to evaluate performance of the model
    :param batch_n: size of learning/validating batch
    :param network: lstm network to
    :param input: dataset on which we should calculate accuracy
    :param device: gpu or cpu
    :return: list of outputs
    """
    outputs_list = []
    loader = torch.utils.data.DataLoader(input, batch_size=batch_n,
                                         shuffle=False, num_workers=0)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data.float()
            inputs = inputs.to(device)

            # calculate outputs by running images through the network
            outputs = network(inputs)

            # outputs_list.append(list(outputs.data[:, 1].cpu().detach().numpy().reshape(-1, )))

        return scipy.special.softmax(outputs.numpy(), axis=1)
