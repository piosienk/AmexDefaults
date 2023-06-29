import numpy as np
import pickle
import pandas as pd
import torch
from LSTM.LSTM_helpers import calculate_accuracy
from sklearn.metrics import PrecisionRecallDisplay
from LSTM.LSTM_model import LSTM

# Settings
# downsampling defaults to 4.3%?
down_defaults = True
training_default_rate = 0.04285
batch_n = 128
m = 20

# Load model
network_trained = torch.load("./Final_models/ews_medium.pickle")

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

avg_precision_list = []
auc_list = []

if down_defaults:

    # Prepare array for sequence length analysis
    seq_length_stats = np.zeros(shape=[6, m])

    # Load data and downsample positive cases randomly
    for i in range(m):
        print("test run {}:".format(i))
        with open('./Additional_data/data_test_EWS_medium.pickle', 'rb') as file:
            data_test = pickle.load(file)

        test_y = pd.DataFrame(data_test["y"], columns=["target"])
        if i == 0:
            print("Shape of the test data before adjustments: ", data_test["y"].shape)
            print("Default rate before adjustment set: ", data_test["y"].mean())

        # Calculate how many defaults we should keep to have the same DR as in the training set
        target_bad_number = round(
                training_default_rate * test_y.loc[test_y.target == 0, :].shape[0] / (1 - training_default_rate))
        to_remove_bad_number = test_y.loc[test_y.target == 1, :].shape[0] - target_bad_number

        # to_remove_bad = test_y.loc[test_y.target == 1, :].sample(76283, random_state=123)
        to_remove_bad = test_y.loc[test_y.target == 1, :].sample(to_remove_bad_number, random_state=i)

        data_test_single_run_y = np.delete(data_test["y"], to_remove_bad.index, axis=0)
        if i == 0:
            print("Shape of the test data after adjustments: ", data_test_single_run_y.shape)
            print("Default rate after adjustment set: ", data_test_single_run_y.mean())

        # adjust X
        data_test_single_run_x= np.delete(data_test["x"], to_remove_bad.index, axis=0)

        lengths_list = []
        for j in range(data_test_single_run_x.shape[0]):
            lengths_list.append(np.argmax(data_test_single_run_x[j].max(axis=1) == 0))

        testloader = torch.utils.data.DataLoader(data_test_single_run_x[:], batch_size=batch_n, shuffle=False,
                                                     num_workers=0)

        targets_test = data_test_single_run_y[:].reshape(-1).tolist()

        acc, outputs_list, avg_precision, auc = calculate_accuracy(network=network_trained, loader=testloader,
                                                                  targets=targets_test,
                                                                  device=device, data_type="valid")

        df_seq_analysis = pd.DataFrame([lengths_list, outputs_list, targets_test]).transpose()
        df_seq_analysis.columns = ["length", "output", "target"]
        seq_length_stats[:,i] = df_seq_analysis.groupby(by=["length"]).apply(
            lambda x: PrecisionRecallDisplay.from_predictions(x["target"], x["output"], name="LSTM").average_precision)

        avg_precision_list.append(avg_precision)
        auc_list.append(auc)


    print("AP list: ", avg_precision_list)
    print("AP mean: ", np.mean(avg_precision_list))
    print("AP std: ", np.std(avg_precision_list))
    print("AUC mean: ", np.mean(auc_list))
    print("AUC std: ", np.std(auc_list))
    print("####################")
    print(seq_length_stats.mean(axis=1))

else:

    with open('./Additional_data/data_test_EWS_medium.pickle', 'rb') as file:
        data_test = pickle.load(file)

    test_y = pd.DataFrame(data_test["y"], columns=["target"])
    print("Shape of the test data before adjustments: ", data_test["y"].shape)
    print("Default rate before adjustment set: ", data_test["y"].mean())

    # Calculate how many defaults we should keep to have the same DR as in the training set
    target_bad_number = round(
        training_default_rate * test_y.loc[test_y.target == 0, :].shape[0] / (1 - training_default_rate))
    to_remove_bad_number = test_y.loc[test_y.target == 1, :].shape[0] - target_bad_number

    # to_remove_bad = test_y.loc[test_y.target == 1, :].sample(76283, random_state=123)
    to_remove_bad = test_y.loc[test_y.target == 1, :].sample(to_remove_bad_number, random_state=123)

    data_test["y"] = np.delete(data_test["y"], to_remove_bad.index, axis=0)
    print("Shape of the test data after adjustments: ", data_test["y"].shape)
    print("Default rate after adjustment set: ", data_test["y"].mean())

    # adjust X
    data_test["x"] = np.delete(data_test["x"], to_remove_bad.index, axis=0)

    testloader = torch.utils.data.DataLoader(data_test["x"][:], batch_size=batch_n, shuffle=False,
                                             num_workers=0)

    targets_test = data_test["y"][:].reshape(-1).tolist()

    acc, outputs_list, avg_precision, auc = calculate_accuracy(network=network_trained, loader=testloader,
                                                          targets=targets_test,
                                                          device=device, data_type="valid")

    print("AP: ", avg_precision)
    print("AUC: ", auc)