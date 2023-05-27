import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from Logistic.logistic_helpers import calculate_diff_in_clients

# Settings
np.random.seed(123)
training_default_rate = 0.04285

# Check performance on test sample

# load lr model
with open('./Final_models/lm_model_13_obs_20_vars.pickle', 'rb') as file:
    lm = pickle.load(file)

# load WoE values
with open('./Additional_data/WoE_binning_merged_13snaps.pickle', 'rb') as file:
    woe_dict = pickle.load(file)

# load selected features
with open('./Additional_data/selected_features.pickle', 'rb') as file:
    selected_features = pickle.load(file)

# Inlcude observations with less than 13 snapshots?
include_non_13 = False
# Downsample defaults to 5%
down_defaults = True

df_test = pd.read_parquet(data_path + "/test_data_split.parquet").fillna(-9999)

# count number of observations per customer
customers_counter = df_test.iloc[:, [0, 1]].groupby("customer_ID").count()
sel_customers = customers_counter.loc[(customers_counter.target == 13)].reset_index().drop(columns="target")
if not include_non_13:
    df_test = df_test.loc[df_test.customer_ID.isin(sel_customers.customer_ID), :]

df_x = df_test.groupby("customer_ID").last().reset_index()
df_x = df_x.drop(columns=["S_2", "customer_obs"])
print(df_x.shape)

# Do the same modifications as in case of training sample
first_snapshot = df_test.groupby("customer_ID").first().reset_index()
first_snapshot = first_snapshot.loc[first_snapshot.customer_ID.isin(df_x.customer_ID),:]
first_snapshot = first_snapshot.drop(columns=["S_2", "customer_obs"])
x_variables = df_x.columns.drop(["customer_ID", "target"])

df_diff = calculate_diff_in_clients(df_x, first_snapshot, x_variables)
df_diff = df_diff.fillna(0)

df_x_test = pd.concat([df_x, df_diff], axis=1).fillna(0).reset_index(drop=True)
df_x_test = df_x_test.drop(columns=["customer_ID"])

# Additional discrete variables in which additional NaNs indicator column should be added
# with median or best fitting group imputuation:
# used in LR only to change feature type
add_int_var = ["D_44", "D_64", "D_68", "D_70", "D_72", "D_78", "D_79", "D_81", "D_83", "D_84", "D_89", "D_91",
               "D_107", "D_117", "D_122", "D_124", "D_125", "D_145"]

# Additional categorical variables where NaN should be coded as another group + one-hot encoded
#  used in LR only to change feature type
add_cat_var = ["B_8", "D_54", "D_66", "D_103", "D_114", "D_116", "D_120", "D_126", "D_128", "D_129", "D_130", "D_139",
               "D_140", "D_143"]

var_to_remove = ["D_87", "D_88", "D_87_diff", "D_88_diff"]

for column in add_int_var:
    df_x_test[column] = df_x_test[column].astype('int16')
for column in add_cat_var:
    df_x_test[column] = df_x_test[column].astype('int16')
#
df_x_test = df_x_test.drop(columns=var_to_remove)

# Replace original values with WoE values (calculate based only on training)
x_variables = df_x_test.columns[1:]
for column in x_variables:
    woe_bin = woe_dict[column]

    if isinstance(df_x_test[column][0], np.int16):
        print("Int: ", column)

        for index, row in woe_bin.iterrows():
            df_x_test.loc[df_x_test[column].isin(row["bin"]), column] = row["woe"]
    else:
        print("Float: ", column)

        for index, row in woe_bin.iterrows():

            if row["Bin"] == "nan":

                df_x_test[column] = df_x_test[column].replace(-9999, np.nan)
                df_x_test.loc[df_x_test[column].isna(), column] = row["woe"]
            else:
                left = float(row["Bin"].split(",")[0].replace("(", ""))
                right = float(row["Bin"].split(",")[1].replace("]", ""))
                df_x_test.loc[(df_x_test[column] > left) & (df_x_test[column] <= (right + 1e-4)), column] = row["woe"]

avg_precision_list = []
auc_list = []

if down_defaults:
    # Remove 75,750 defaults in the test sample
    print("Shape of the test data before adjustments: ", df_x_test.shape)
    print("Default rate before adjustment set: ", df_x_test["target"].mean())

    for i in range(20):
        print("test run {}:".format(i))

        test_y = df_x_test["target"]

        # Calculate how many defaults we should keep to have the same DR as in the training set
        target_bad_number = round(
                training_default_rate * len(test_y.loc[test_y == 0]) / (1 - training_default_rate))
        to_remove_bad_number = len(test_y.loc[test_y == 1])- target_bad_number

        # to_remove_bad = test_y.loc[test_y.target == 1, :].sample(76283, random_state=123)
        to_remove_bad = test_y.loc[test_y == 1].sample(to_remove_bad_number, random_state=i)

        df_x_test_single_run = df_x_test.drop(index=to_remove_bad.index)

        if i == 0:
            print("Shape of the test data after adjustments: ", df_x_test_single_run["target"].shape)
            print("Default rate after adjustment set: ", df_x_test_single_run["target"].mean())

        # Make prediction on test sample
        y_predict = lm.predict(df_x_test_single_run.loc[:,selected_features])
        y_predict_prob = lm.predict_proba(df_x_test_single_run.loc[:,selected_features])
        y_test = df_x_test_single_run.target


        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_prob[:,1])

        # Precision-recall curve for LR
        display = metrics.PrecisionRecallDisplay.from_predictions(y_test, y_predict_prob[:,1], name="LR+WoE")
        _ = display.ax_.set_title("Test 2-class Precision-Recall curve")
        plt.show()
        avg_precision = display.average_precision
        auc = metrics.roc_auc_score(y_test.to_numpy(), y_predict_prob[:,1])

        avg_precision_list.append(avg_precision)
        auc_list.append(auc)

    print("AP list: ", avg_precision_list)
    print("AP mean: ", np.mean(avg_precision_list))
    print("AP std: ", np.std(avg_precision_list))
    print("AUC mean: ", np.mean(auc_list))
    print("AUC std: ", np.std(auc_list))


else:
    # Make prediction on test sample
    y_predict = lm.predict(df_x_test.loc[:, selected_features])
    y_predict_prob = lm.predict_proba(df_x_test.loc[:, selected_features])
    y_test = df_x_test.target

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_prob[:, 1])

    # Precision-recall curve for LR
    display = metrics.PrecisionRecallDisplay.from_predictions(y_test, y_predict_prob[:, 1], name="LR+WoE")
    _ = display.ax_.set_title("Test 2-class Precision-Recall curve")
    plt.show()
    avg_precision = display.average_precision
    auc = metrics.roc_auc_score(y_test.to_numpy(), y_predict_prob[:, 1])

    print("AP: ", avg_precision)
    print("AUC: ", auc)

coefs = pd.DataFrame(selected_features)
coefs["coef"] = lm.coef_.reshape(-1, 1)
print("LR coefficients: ", coefs.loc[(coefs.coef != 0) & (coefs.coef.abs() > 1e-5), :])