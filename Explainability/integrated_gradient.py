import captum.attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn as nn

# Load model
from Explainability.explainability_helpers import calculate_integrated_gradient

network_trained = torch.load("../LSTM/Final_models/lstm.pickle")
ig = captum.attr.IntegratedGradients(network_trained)

# Settings
# downsampling defaults to 5%?
down_defaults = True
batch_n = 128
element = 0
n_variables = 15

# Load data and downsample positive cases randomly
# TO DO - add pseudo cross validation only on selection of these downsampled cases
with open('../LSTM/Additional_data/data_test_LSTM.pickle', 'rb') as file:
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

# Prepare input and baseline for integrated gradient
bad_index = np.where(data_test["y"].reshape(-1) == 1)
good_index = np.where(data_test["y"].reshape(-1) == 0)

input_bad = torch.from_numpy(data_test["x"][bad_index[0][:1500], :, :]).reshape(-1, 13, 237)
input_good = torch.from_numpy(data_test["x"][good_index[0][:1500], :, :]).reshape(-1, 13, 237)
input_test = torch.from_numpy(data_test["x"][:1500, :, :]).reshape(-1, 13, 237)

baseline_bad = torch.zeros(input_bad.shape)
baseline_good = torch.zeros(input_good.shape)
baseline = torch.zeros(input_test.shape)

y_bad = np.repeat(1, 1500).tolist()
y_good = np.repeat(0, 1500).tolist()
y_test = data_test["y"][:1500].reshape(-1).tolist()

del data_test

attributions, delta = calculate_integrated_gradient(ig, input_test, baseline, y_test, n_variables, 1500)
attributions_bad, delta_bad = calculate_integrated_gradient(ig, input_bad, baseline_bad, y_bad, n_variables, 1500)
attributions_good, delta_good = calculate_integrated_gradient(ig, input_good, baseline_good, y_good, n_variables, 1500)

print('IG Attributions:', attributions_bad)
print('Convergence Delta:', delta_bad)

