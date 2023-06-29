import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Explainability.explainability_helpers import partial_dependence

data = pd.read_parquet(data_path + "/test_data_for_LSTM.parquet")
variable = "B_4"
n = 50
approach = "static"
client = 50


xs_list, average_f_result_list, average_f_predict_list = partial_dependence(data, variable, n, approach=approach,
                                                                  ceteris_paribus=True, client=client)

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
