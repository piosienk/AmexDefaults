import pandas as pd
import matplotlib.pyplot as plt
from Explainability.explainability_helpers import partial_dependence

data = pd.read_parquet(data_path + "/test_data_split.parquet")
variable = "B_4"
n = 20


xs_list, average_f_result, average_f_predict = partial_dependence(data, variable, n, approach="static")

plt.subplots(figsize=(10,5))
plt.plot(xs_list, average_f_predict)
plt.show()

# plt.plot(xs_list, average_f_result)
# plt.show()

print(9)
