import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay, roc_curve


def prepare_ews_data_3(input_path, output_path, time_windows):
    """
    prepare dictionary with x and y for pytorch dataloader

    :param time_windows: possible lengths of sequences
    :param input_path: path to input
    :param output_path: path to output
    :returns None - it saves dictionary with X and Y elements that can be fed into pytorch dataloader
    """
    df = pd.read_parquet(input_path)
    df = df.drop(columns=["S_2"], inplace=False)

    # Randomly remove the last k observations for a customer (replace values with 0)
    customers = df.customer_ID.unique()
    n_to_keep = np.random.choice(time_windows, size=len(customers))

    indices_to_keep_list = []
    for i in range(len(customers)):
        indices_to_keep = np.arange(i * 13, (i * 13) + n_to_keep[i]).tolist()
        indices_to_keep_list.append(indices_to_keep)

    indices_to_keep_list = [item for sublist in indices_to_keep_list for item in sublist]
    indices_to_remove_list = list(set(np.arange(0, df.shape[0])).difference(indices_to_keep_list))

    df.iloc[indices_to_remove_list, 2:] = 0

    y_df = df.loc[:, ["target", "customer_ID"]].groupby("customer_ID").mean().to_numpy("int")
    x_df = df.drop(columns=["target", "customer_ID"]).to_numpy("float32").reshape((-1, 13, 237), order="C")

    data = {"y": y_df, "x": x_df}

    with open(output_path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def calculate_accuracy(network, loader, targets, device, data_type, batch_n=128):
    """
    Function to evaluate performance of the model

    :param batch_n: size of learning/validating batch
    :param network: lstm network to
    :param loader: data loader for the dataset on which we should calculate accuracy
    :param targets: true y label
    :param device: gpu or cpu
    :param data_type: validation or training
    :return: list of outputs and accuracy values
    """
    correct = 0
    total = 0
    outputs_list = []
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

    # Calculate AUC
    outputs_list = [item for sublist in outputs_list for item in sublist]
    auc = roc_auc_score(targets, outputs_list)

    if data_type == 'valid':
        print(f'Accuracy of the network on the validation: {100 * correct // total} %')
        print(f'AUC of the network on the validation: {100 * round(auc, 3)} %')

        display = PrecisionRecallDisplay.from_predictions(targets, outputs_list, name="LSTM")
        _ = display.ax_.set_title("2-class Precision-Recall curve")
        plt.show()

    if data_type == 'train':
        print(f'Accuracy of the network on the train: {100 * correct // total} %')
        print(f'AUC of the network on the train: {100 * round(auc, 3)} %')

    accuracy = 100 * correct / total

    return [accuracy, outputs_list]


