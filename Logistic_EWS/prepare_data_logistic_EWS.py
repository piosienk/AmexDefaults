import numpy as np
import pandas as pd
from Logistic.logistic_helpers import perform_merging_int, perform_merging_float, calculate_diff_in_clients
pd.options.mode.chained_assignment = None
import pickle

# Settings
alpha = 0.05
k = 1000  # for int values
k2 = 15000  # for float values
time_windows = [2, 3, 4, 5, 6]


# Load and preprocess
df_train = pd.read_parquet(data_path + "/train_data_split.parquet").fillna(-9999)
df_train = df_train.sort_values(by=["customer_ID", "customer_obs"])
customers_counter = df_train.iloc[:,[0,1]].groupby("customer_ID").count()
sel_customers = customers_counter.loc[customers_counter.target == 13].reset_index().drop(columns="target")
df_train = df_train.loc[df_train.customer_ID.isin(sel_customers.customer_ID),:]

# Randomly cut customers observations and create shorter sequences
customers = sel_customers.customer_ID.unique()
n_to_keep = np.random.choice(time_windows, size=len(customers))
indices_to_keep_list = []
for i in range(len(customers)):
    indices_to_keep = np.arange(i * 13, (i * 13) + n_to_keep[i]).tolist()
    indices_to_keep_list.append(indices_to_keep)

indices_to_keep_list = [item for sublist in indices_to_keep_list for item in sublist]
df_train = df_train.iloc[indices_to_keep_list, :].reset_index(drop=True)

## Additional discrete variables in which additional NaNs indicator column should be added
## with median or best fitting group imputuation:
## used in LR only to change feature type
add_int_var = ["D_44", "D_64", "D_68", "D_70", "D_72", "D_78", "D_79", "D_81", "D_83", "D_84", "D_89", "D_91",
               "D_107", "D_117", "D_122", "D_124", "D_125", "D_145"]

## Additional categorical variables where NaN should be coded as another group + one-hot encoded
##  used in LR only to change feature type
add_cat_var = ["B_8", "D_54", "D_66", "D_103", "D_114", "D_116", "D_120", "D_126", "D_128", "D_129", "D_130", "D_139",
               "D_140", "D_143"]

var_to_remove = ["D_87", "D_88", "D_108", "D_111"]

for column in add_int_var:
    df_train[column] = df_train[column].astype('int16')
for column in add_cat_var:
    df_train[column] = df_train[column].astype('int16')

df_train = df_train.drop(columns=var_to_remove)

x_variables = df_train.columns.to_list()
to_remove = ["target", "S_2", "customer_obs"]
for column in to_remove:
    x_variables.remove(column)

df_x = df_train.groupby("customer_ID").last().reset_index()
df_y = df_x.loc[:, ["customer_ID", "target"]]

first_snapshot = df_train.loc[:,x_variables].groupby("customer_ID").first().reset_index()
x_variables.remove("customer_ID")

# Calculate the difference between last and first observation per variable
df_diff = calculate_diff_in_clients(df_x, first_snapshot, x_variables)

df_x2 = pd.concat([df_x, df_diff], axis=1).fillna(0)
df_y.to_parquet("./Additional_data/df_train_y_reduced_merged.parquet")

# Calculate WoE for every column
df_new = df_x2.iloc[:,3:].copy(deep=True)
df_new = df_new.drop(columns="customer_obs")

itr = 0
binning_dict = {}
for column in df_new.iloc[:, :].columns:

    print("Column nr: ", itr)
    itr += 1

    if isinstance(df_x2[column][0], np.int16):
        print(column)

        binning, df_woe = perform_merging_int(df_x2, column, k, alpha)
        df_woe = pd.concat([binning, df_woe], axis=1)

        for index, row in df_woe.iterrows():
            df_new.loc[df_new[column].isin(row["bin"]), column] = row["woe"]
    else:
        print(column)

        binning, df_woe, interval_df = perform_merging_float(df_x2, column, k2, alpha)
        df_woe = pd.concat([binning, df_woe], axis=1)
        df_new[column] = df_new[column].replace(-9999, np.nan)

        for index, row in df_woe.iterrows():

            if row["Bin"] == "nan":
                value = np.nan
                df_new.loc[df_new[column].isna(), column] = row["woe"]
            else:
                left = float(row["Bin"].split(",")[0].replace("(", ""))
                right = float(row["Bin"].split(",")[1].replace("]", ""))
                df_new.loc[(df_new[column] > left) & (df_new[column] <= (right + 1e-4)), column] = row["woe"]

    binning_dict[column] = df_woe
    print(df_woe.drop(columns="Bin"))

df_new.to_parquet(data_path + "/train_data_woe_merged_short.parquet")
with open('./Additional_data/WoE_binning_merged_short.pickle', 'wb') as file:
    pickle.dump(binning_dict, file)