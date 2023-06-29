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


def partial_dependence(data, variable, n, approach="static", ceteris_paribus=False, client=None):
    # Network preps
    batch_n = 128
    training_default_rate = 0.04285

    # Load model
    network_trained = torch.load("../LSTM/Final_models/lstm.pickle")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    col_index = np.where(data.columns[3:] == variable)[0][0]
    variables_names = pd.read_csv("../LSTM/Additional_data/variables_names.csv")
    print("Variable index cross check:", variables_names.iloc[col_index, 1])

    if (data[variable].dtype == "float64") | (data[variable].dtype == "float32"):

        data = data.replace(to_replace=-9999, value=np.nan)
        variable_max = data[variable].max()
        variable_min = data[variable].min()

        xs_list = np.linspace(variable_min, variable_max, n)

        average_f_predict_bootstrap = []
        average_f_result_bootstrap = []

        if approach == "static":

            with open('../LSTM/Additional_data/data_test_LSTM.pickle', 'rb') as file:
                data = pickle.load(file)

            for i in range(2):

                average_f_result = []
                average_f_predict = []

                data_x_down, data_y_down = downsample_explainability_sample(data, i=i, training_default_rate= training_default_rate)

                for xs in xs_list:
                    data_x, data_y = data_x_down.copy(), data_y_down.copy()

                    if ceteris_paribus:
                        x_orig = data_x[client, 12, col_index]
                        if xs == xs_list[0]:
                            print("Oiriginal X value ", x_orig)
                        data_x[client, :, col_index] = xs
                        data_y = data_y[client].reshape(-1).tolist()
                        if xs == xs_list[0]:
                            print("Target value: ", data_y[0])

                        testloader = torch.utils.data.DataLoader(data_x[client], batch_size=batch_n, shuffle=False,
                                                                 num_workers=0)
                    else:
                        data_x[:, :, col_index] = xs
                        # test_y = data["y"][:10000].reshape(-1).tolist()
                        data_y = data_y.reshape(-1).tolist()[:]
                        testloader = torch.utils.data.DataLoader(data_x[:], batch_size=batch_n, shuffle=False,
                                                                 num_workers=0)


                    acc, outputs_list, predicted_list = calculate_accuracy_PDP(network=network_trained,
                                                                                              loader=testloader,
                                                                                              targets=data_y,
                                                                                              device=device)

                    avg_predicted = np.mean(predicted_list)
                    average_f_predict.append(avg_predicted)

                    avg_response = np.mean(outputs_list)
                    average_f_result.append(avg_response)

                average_f_predict_bootstrap.append(average_f_predict)
                average_f_result_bootstrap.append(average_f_result)

            return xs_list, average_f_predict_bootstrap, average_f_result_bootstrap

        else:

            with open('../LSTM/Additional_data/data_test_LSTM.pickle', 'rb') as file:
                data = pickle.load(file)

            for i in range(2):

                average_f_result = []
                average_f_predict = []

                data_x_down, data_y_down = downsample_explainability_sample(data, i=i, training_default_rate= training_default_rate)

                for xs in xs_list:

                    data_x, data_y = data_x_down.copy(), data_y_down.copy()

                    if ceteris_paribus:
                        x_orig = data_x[client, 12, col_index]
                        if xs == xs_list[0]:
                            print("Oiriginal X value ", x_orig)
                        first_obs = data_x[client, 0, col_index]
                        data_x[client, :, col_index] = np.transpose(np.linspace(first_obs, xs, 13))
                        data_y = data_y[client].reshape(-1).tolist()
                        if xs == xs_list[0]:
                            print("Target value: ", data_y[0])

                        testloader = torch.utils.data.DataLoader(data_x[client], batch_size=batch_n, shuffle=False,
                                                                 num_workers=0)

                    else:
                        first_obs = data_x[:, 0, col_index]
                        data_x[:, :, col_index] = np.transpose(np.linspace(first_obs, xs, 13))
                        # test_y = data["y"][:10000].reshape(-1).tolist()
                        data_y = data_y.reshape(-1).tolist()[:]
                        testloader = torch.utils.data.DataLoader(data_x[:], batch_size=batch_n, shuffle=False,
                                                                 num_workers=0)

                    acc, outputs_list, predicted_list = calculate_accuracy_PDP(network=network_trained,
                                                                                              loader=testloader,
                                                                                              targets=data_y,
                                                                                              device=device)

                    avg_predicted = np.mean(predicted_list)
                    average_f_predict.append(avg_predicted)

                    avg_response = np.mean(outputs_list)
                    average_f_result.append(avg_response)

                average_f_predict_bootstrap.append(average_f_predict)
                average_f_result_bootstrap.append(average_f_result)

            return xs_list, average_f_predict_bootstrap, average_f_result_bootstrap

    else:

        data = data.replace(to_replace=-9999, value=np.nan)
        xs_list = data[variable].value_counts().head(n).index.tolist()
        xs_list.sort()

        average_f_predict_bootstrap = []
        average_f_result_bootstrap = []


        if approach == "static":

            with open('../LSTM/Additional_data/data_test_LSTM.pickle', 'rb') as file:
                data = pickle.load(file)

            for i in range(2):

                average_f_result = []
                average_f_predict = []

                data_x_down, data_y_down = downsample_explainability_sample(data, i=i, training_default_rate= training_default_rate)

                for xs in xs_list:
                    data_x, data_y = data_x_down.copy(), data_y_down.copy()

                    if ceteris_paribus:
                        x_orig = data_x[client, 12, col_index]
                        if xs == xs_list[0]:
                            print("Oiriginal X value ", x_orig)
                        data_x[client, :, col_index] = xs
                        data_y = data_y[client].reshape(-1).tolist()
                        if xs == xs_list[0]:
                            print("Target value: ", data_y[0])

                        testloader = torch.utils.data.DataLoader(data_x[client], batch_size=batch_n, shuffle=False,
                                                                 num_workers=0)
                    else:
                        data_x[:, :, col_index] = xs
                        # test_y = data["y"][:10000].reshape(-1).tolist()
                        data_y = data_y.reshape(-1).tolist()[:]
                        testloader = torch.utils.data.DataLoader(data_x[:], batch_size=batch_n, shuffle=False,
                                                                 num_workers=0)


                    acc, outputs_list, predicted_list = calculate_accuracy_PDP(network=network_trained,
                                                                                              loader=testloader,
                                                                                              targets=data_y,
                                                                                              device=device)

                    avg_predicted = np.mean(predicted_list)
                    average_f_predict.append(avg_predicted)
                    avg_response = np.mean(outputs_list)
                    average_f_result.append(avg_response)

                average_f_predict_bootstrap.append(average_f_predict)
                average_f_result_bootstrap.append(average_f_result)

            return xs_list, average_f_predict_bootstrap, average_f_result_bootstrap

        else:

            with open('../LSTM/Additional_data/data_test_LSTM.pickle', 'rb') as file:
                data = pickle.load(file)

            for i in range(2):
                average_f_result = []
                average_f_predict = []

                data_x_down, data_y_down = downsample_explainability_sample(data, i=i, training_default_rate= training_default_rate)

                for xs in xs_list:
                    data_x, data_y = data_x_down.copy(), data_y_down.copy()

                    if ceteris_paribus:
                        x_orig = data_x[client, 12, col_index]
                        if xs == xs_list[0]:
                            print("Oiriginal X value", x_orig)
                        first_obs = data_x[client, 0, col_index]
                        data_x[client, :, col_index] = np.transpose(np.linspace(first_obs, xs, 13))
                        data_y = data_y[client].reshape(-1).tolist()
                        if xs == xs_list[0]:
                            print("Target value: ", data_y[0])

                        testloader = torch.utils.data.DataLoader(data_x[client], batch_size=batch_n, shuffle=False,
                                                                 num_workers=0)

                    else:
                        first_obs = data_x[:, 0, col_index]
                        data_x[:, :, col_index] = np.transpose(np.linspace(first_obs, xs, 13))
                        data_y = data_y.reshape(-1).tolist()[:]


                        testloader = torch.utils.data.DataLoader(data_x[:], batch_size=batch_n, shuffle=False,
                                                                 num_workers=0)

                    acc, outputs_list, predicted_list = calculate_accuracy_PDP(network=network_trained,
                                                                                              loader=testloader,
                                                                                              targets=data_y,
                                                                                              device=device)

                    avg_predicted = np.mean(predicted_list)
                    average_f_predict.append(avg_predicted)

                    avg_response = np.mean(outputs_list)
                    average_f_result.append(avg_response)

                average_f_predict_bootstrap.append(average_f_predict)
                average_f_result_bootstrap.append(average_f_result)

            return xs_list, average_f_predict_bootstrap, average_f_result_bootstrap

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
            inputs = data.float().reshape(-1,13,237)
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
    accuracy = 100 * correct / total
    print("Accuracy: ", accuracy)

    return [accuracy, outputs_list, predicted_list]

def downsample_explainability_sample(data, i, training_default_rate):

    test_y = pd.DataFrame(data["y"], columns=["target"])
    if i == 0:
        print("Shape of the test data before adjustments: ", data["y"].shape)
        print("Default rate before adjustment set: ", data["y"].mean())

    # Calculate how many defaults we should keep to have the same DR as in the training set
    target_bad_number = round(
        training_default_rate * test_y.loc[test_y.target == 0, :].shape[0] / (
                1 - training_default_rate))
    to_remove_bad_number = test_y.loc[test_y.target == 1, :].shape[0] - target_bad_number

    # to_remove_bad = test_y.loc[test_y.target == 1, :].sample(76283, random_state=123)
    to_remove_bad = test_y.loc[test_y.target == 1, :].sample(to_remove_bad_number, random_state=i)

    data_test_single_run_y = np.delete(data["y"], to_remove_bad.index, axis=0)
    if i == 0:
        print("Shape of the test data after adjustments: ", data_test_single_run_y.shape)
        print("Default rate after adjustment set: ", data_test_single_run_y.mean())

    # adjust X
    data_test_single_run_x = np.delete(data["x"], to_remove_bad.index, axis=0)

    return data_test_single_run_x, data_test_single_run_y
