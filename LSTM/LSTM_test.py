import numpy as np
import pickle
import pandas as pd
import torch
from LSTM.LSTM_helpers import calculate_accuracy
from LSTM.LSTM_model import LSTM

# Settings
# downsampling defaults to 5%?
down_defaults = True
batch_n = 128

# Load data and downsample positive cases randomly
# TO DO - add pseudo cross validation only on selection of these downsampled cases
with open('./Additional_data/data_test_LSTM.pickle', 'rb') as file:
    data_test = pickle.load(file)

if down_defaults:
    test_y = pd.DataFrame(data_test["y"], columns=["target"])
    # Remove 75,750 defaults in the test sample (to have 4.9% of defaults, as in training sample)
    print("Shape of the test data before adjustments: ", data_test["y"].shape)
    print("Default rate before adjustment set: ", data_test["y"].mean())

    # to_remove_bad = test_y.loc[test_y.target == 1, :].sample(75825, random_state=123)
    to_remove_bad = test_y.loc[test_y.target == 1, :].sample(76283, random_state=123)

    data_test["y"] = np.delete(data_test["y"], to_remove_bad.index, axis=0)
    print("Shape of the test data after adjustments: ", data_test["y"].shape)
    print("Default rate after adjustment set: ", data_test["y"].mean())

# adjust X
data_test["x"] = np.delete(data_test["x"], to_remove_bad.index, axis=0)

testloader = torch.utils.data.DataLoader(data_test["x"][:], batch_size=batch_n, shuffle=False, num_workers=0)

targets_test = data_test["y"][:].reshape(-1).tolist()

# Load model
network_trained = torch.load("./Final_models/lstm.pickle")

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

acc, outputs_list= calculate_accuracy(network=network_trained, loader=testloader, targets=targets_test,
                   device=device, data_type="valid")