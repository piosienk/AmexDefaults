import pickle
import lime.lime_tabular as lt
import pandas as pd
import numpy as np

from Explainability.explainability_helpers import run_lstm_for_lime

lstm_data_path = "../LSTM/Additional_data"

with open(lstm_data_path+"/data_train_LSTM.pickle", 'rb') as file:
    data_train = pickle.load(file)

variable_names = pd.read_csv(lstm_data_path+"/variables_names.csv", sep=",").iloc[:,1].tolist()

# int columns lists (actually categorical) taken from LSTM data processing 1
# nanflags = ["flag" in i for i in variable_names]
# nanflags_vars = np.array(variable_names)[np.array(nanflags)]
# float_to_nanflag = ["D_73", "D_76", "R_9", "D_82", "B_29", "D_106", "R_26", "D_108", "D_110", "D_111", "B_39", "B_42",
#                     "D_132", "D_134", "D_135", "D_136", "D_137", "D_138", "D_142"]
# add_cat_var = ['B_8_0', 'B_8_1', 'D_54_-1', 'D_54_0', 'D_54_1', 'D_103_0', 'D_103_1', 'D_128_0', 'D_128_1',
#                'D_129_0', 'D_129_1', 'D_130_0', 'D_130_1', 'D_139_0', 'D_139_1', 'D_140_0', 'D_140_1', 'D_143_0',
#                'D_143_1']

# cat_variables_final = nanflags_vars.tolist() + float_to_nanflag + add_cat_var
# cat_variables_final_indexes = np.array([np.where(np.array(variable_names)==i) for i in cat_variables_final]).reshape(-1)
y_train = data_train["y"].reshape(-1)

kernel_width = np.sqrt(237*13)*0.75

lime_explainer = lt.RecurrentTabularExplainer(training_data=data_train["x"][:100000], mode="classification",
                                              training_labels=y_train[:100000], feature_names=variable_names,
                                              # categorical_features=cat_variables_final_indexes,
                                         # categorical_names=cat_variables_final,
                                              kernel_width=kernel_width,
                                         discretize_continuous=False, random_state=123)

del data_train

with open(lstm_data_path+"/data_test_LSTM.pickle", 'rb') as file:
    data_test = pickle.load(file)

y_test = data_test["y"].reshape(-1)

for i in range(1):
    example = i

    explanation = lime_explainer.explain_instance(data_test["x"][example].reshape(1,13,237), run_lstm_for_lime,
                                                  labels=[y_test[example]], num_samples=1000)
    explanation.save_to_file("./Tests/lime_explanation_ex_{}_y={}_kern={}.html".format(example, y_test[example], kernel_width))
    print(y_test[example])
