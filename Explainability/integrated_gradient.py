import captum.attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn as nn

# Load model
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
input_test = torch.from_numpy(data_test["x"][:1000, :, :]).reshape(1000,13,237)
baseline = torch.zeros(input_test.shape)
y_target = data_test["y"][:1000].reshape(1000).tolist()

attributions, delta = ig.attribute(input_test, baseline, target=y_target, return_convergence_delta=True)
attributions = attributions.numpy().reshape(1000, 13,237)
variable_names = pd.read_csv("../LSTM/Additional_data/variables_names.csv")
variable_names.columns = ["var_num", "var_name"]
df_attributions = pd.DataFrame(attributions.mean(axis=0), columns=variable_names.var_name)
features_max_attribution = df_attributions.max().abs().sort_values(ascending=False).iloc[:n_variables]

# plot max attribution
plt.subplots(figsize=(15,6))
plt.bar(features_max_attribution.index, features_max_attribution.values)
plt.show()

# plot attribution change over time
plt.subplots(figsize=(15,8))
for column in features_max_attribution.index:
    plt.plot(df_attributions[column], label=column)
plt.legend()
plt.show()


print('IG Attributions:', attributions)
print('Convergence Delta:', delta)

