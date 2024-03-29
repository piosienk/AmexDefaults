import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Explainability.explainability_helpers import partial_dependence

data = pd.read_parquet(data_path + "/test_data_for_LSTM.parquet")
variable = "B_19"
n = 50
approach = "dynamic"
# down_defaults = False
#
# test_y = data.groupby(by=["customer_ID"])["target"].max()
# bad_clients = test_y[test_y == 1].index
# # Calculate how many defaults we should keep to have the same DR as in the training set
# target_bad_number = round(
#     training_default_rate * test_y[test_y == 0].shape[0] / (1 - training_default_rate))
# to_remove_bad_number = test_y[test_y == 1].shape[0] - target_bad_number
#
# # to_remove_bad = test_y.loc[test_y.target == 1, :].sample(76283, random_state=123)
# to_remove_bad = pd.DataFrame(bad_clients).sample(to_remove_bad_number, random_state=0)
# data_adj = data[~data["customer_ID"].isin(to_remove_bad["customer_ID"])]


xs_list, average_f_result_list, average_f_predict_list = partial_dependence(data, variable, n, approach=approach)

average_f_result = np.array(average_f_result_list).mean(axis=0)
average_f_predict = np.array(average_f_predict_list).mean(axis=0)

fig, ax1 = plt.subplots(figsize=(10,5))

a, b, c = plt.hist(data[variable], bins=xs_list)
ax1.stairs(a, b, color="orange", fill=True)
ax1.set_ylabel("Values Histogram")
ax1.set_xlabel("Variable {} value".format(variable))
ax2 = ax1.twinx()

ax2.plot(xs_list, average_f_predict, "o-")
ax2.set_ylabel("Average Prediction 0-1")
ax2.set_xlabel("Variable {} value".format(variable))
plt.show()

fig, ax1 = plt.subplots(figsize=(10,5))
a, b, c = plt.hist(data[variable], bins=xs_list)
ax1.stairs(a, b, color="orange", fill=True)
ax1.set_ylabel("Values Histogram")
ax1.set_xlabel("Variable {} value".format(variable))

ax2 = ax1.twinx()

ax2.plot(xs_list, average_f_result, "o-")
ax2.set_ylabel("Average Prediction (continuous)")
ax2.set_xlabel("Variable {} value".format(variable))
plt.show()
print("Explanations Finished")
