import captum.attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
import seaborn as sns
import torch
import scipy
from sklearn.metrics import PrecisionRecallDisplay

from LSTM.LSTM_helpers import calculate_accuracy


def calculate_integrated_gradient(ig, input, baseline, target, n_variables, n):
    attributions, delta = ig.attribute(input, baseline, target=target, return_convergence_delta=True)
    attributions = attributions.numpy().reshape(n, 13, 237)
    variable_names = pd.read_csv("../LSTM/Additional_data/variables_names.csv")
    variable_names.columns = ["var_num", "var_name"]
    df_attributions = pd.DataFrame(attributions.mean(axis=0), columns=variable_names.var_name)
    features_max_attribution = df_attributions.abs().max().sort_values(ascending=False).iloc[:n_variables]

    # plot max attribution
    plt.subplots(figsize=(15, 6))
    plt.bar(features_max_attribution.index, features_max_attribution.values)
    plt.xticks(rotation=-45)
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
    plt.xticks(rotation=-45)
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


def partial_dependence(data, variable, n, approach="static"):
    # Network preps
    batch_n = 128

    # Load model
    network_trained = torch.load("../LSTM/Final_models/lstm.pickle")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    col_index = np.where(data.columns == variable)[0][0]

    if (data[variable].dtype == "float64") | (data[variable].dtype == "float32"):

        data = data.replace(to_replace=-9999, value=np.nan)
        variable_max = data[variable].max()
        variable_min = data[variable].min()

        if approach == "static":
            xs_list = np.linspace(variable_min, variable_max, n)
            average_f_result = []
            average_f_predict = []

            with open('../LSTM/Additional_data/data_test_LSTM.pickle', 'rb') as file:
                data = pickle.load(file)

            for xs in xs_list:
                data["x"][:, :, col_index] = xs
                test_y = data["y"][:10000].reshape(-1).tolist()

                testloader = torch.utils.data.DataLoader(data["x"][:10000], batch_size=batch_n, shuffle=False,
                                                         num_workers=0)

                acc, outputs_list, avg_precision, predicted_list = calculate_accuracy_PDP(network=network_trained, loader=testloader,
                                                                           targets=test_y,
                                                                           device=device)

                avg_predicted = np.mean(predicted_list)
                average_f_predict.append(avg_predicted)

                avg_response = np.mean(outputs_list)
                average_f_result.append(avg_response)

            return xs_list, average_f_result, average_f_predict

        else:
            # dorobić opcję z tworzeniem wartości od t=0 do ustalonego t=13
            pass

    else:

        if approach == "static":

            xs_list = data[variable].value_counts().head(100).index.tolist()
            xs_list.sort()
            average_f_result = []
            average_f_predict = []

            with open('../LSTM/Additional_data/data_test_LSTM.pickle', 'rb') as file:
                data = pickle.load(file)

            for xs in xs_list:
                data["x"][:, :, col_index] = xs
                test_y = data["y"][:10000].reshape(-1).tolist()

                testloader = torch.utils.data.DataLoader(data["x"][:10000], batch_size=batch_n, shuffle=False,
                                                         num_workers=0)

                acc, outputs_list, avg_precision, predicted_list = calculate_accuracy_PDP(network=network_trained, loader=testloader,
                                                                           targets=test_y,
                                                                           device=device)

                avg_predicted = np.mean(predicted_list)
                average_f_predict.append(avg_predicted)

                avg_response = np.mean(outputs_list)
                average_f_result.append(avg_response)

            return xs_list, average_f_result, average_f_predict
        else:
            # dorobić opcję z tworzeniem wartości od t=0 do ustalonego t=13
            pass

def calculate_accuracy_PDP(network, loader, targets, device, batch_n=128):
    """
    Function to evaluate performance of the model for partial dependence plots

    :param batch_n: size of learning/validating batch
    :param network: lstm network to
    :param loader: data loader for the dataset on which we should calculate accuracy
    :param targets: true y label
    :param device: gpu or cpu
    :return: list of outputs and accuracy values
    """

    correct = 0
    total = 0
    outputs_list = []
    predicted_list = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data.float()
            labels = targets[i * batch_n:(i + 1) * batch_n]
            labels = torch.IntTensor(labels)
            #             print(labels.shape)
            labels = torch.reshape(labels, (-1,))
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)

            # calculate outputs by running images through the network
            outputs = network(inputs)

            outputs_list.append(list(outputs.data[:, 1].cpu().detach().numpy().reshape(-1, )))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_list.append(list(predicted.cpu().detach().numpy().reshape(-1, )))

    # Calculate AUC
    outputs_list = [item for sublist in outputs_list for item in sublist]
    predicted_list = [item for sublist in predicted_list for item in sublist]

    display = PrecisionRecallDisplay.from_predictions(targets, outputs_list, name="LSTM")
    plt.show()
    average_precision = display.average_precision
    print("AP: ", average_precision)
    accuracy = 100 * correct / total

    return [accuracy, outputs_list, average_precision, predicted_list]
