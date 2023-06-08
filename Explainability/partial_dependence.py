import pandas as pd
import matplotlib.pyplot as plt
from Explainability.explainability_helpers import partial_dependence

data = pd.read_parquet(data_path + "/test_data_for_LSTM.parquet")
variable = "D_39"
n = 50
approach = "dynamic"


xs_list, average_f_result, average_f_predict = partial_dependence(data, variable, n, approach=approach)

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
