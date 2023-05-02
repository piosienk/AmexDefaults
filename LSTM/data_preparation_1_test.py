import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pickle

np.random.seed(123)

# Load data
df = pd.read_parquet(data_path + "/test_data_split.parquet")
df.sort_values(by=["customer_ID", "customer_obs"], inplace=True)
# Remove static columns
static_columns = ['D_63', 'D_64', 'D_66', 'D_68', 'B_30', 'B_31', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126',
                  "customer_obs"]
df = df.drop(columns=static_columns)

# Missing handling
# 1. Remove columns with excessive number of missings
var_to_remove = ["D_87", "D_88"]
df = df.drop(columns=var_to_remove)
# 2. Change columns to int, create missing indicator flag and impute with median
add_int_var = ["D_44", "D_70", "D_72", "D_78", "D_79", "D_81", "D_83", "D_84", "D_89", "D_91",
               "D_107", "D_122", "D_124", "D_125", "D_145"]

df[add_int_var] = df[add_int_var].astype("int16")
# load pickle with imputer
with open('./Imputers/int_imputer1.pickle', 'rb') as file:
    int_imputer = pickle.load(file)
new_cols = []
for column in add_int_var:
    new_col = column + "_miss_flag"
    new_cols.append(new_col)
    df[new_col] = None

df.loc[:, add_int_var + new_cols] = int_imputer.transform(df[add_int_var])

# 3. One-hot encode categorical variables
add_cat_var = ["B_8", "D_54", "D_103", "D_128", "D_129", "D_130", "D_139", "D_140", "D_143"]
# load pickle
with open('./Imputers/onehot_imputer.pickle', 'rb') as file:
    onehot_imputer = pickle.load(file)
df[onehot_imputer.get_feature_names_out()] = onehot_imputer.transform(df[add_cat_var]).toarray()
df = df.drop(columns=add_cat_var)

# 4. Impute with median
var_nan_impute = ["P_2", "B_2", "D_41", "B_3", "D_45", "D_52", "B_15", "B_16", "B_19", "B_20", "B_22", "D_74",
                  "B_25", "B_26", "D_80", "B_27", "B_33", "S_26", "D_109", "D_112", "B_41"]

# load pickle
with open('./Imputers/int_imputer2.pickle', 'rb') as file:
    int_imputer = pickle.load(file)

df.loc[:, var_nan_impute] = int_imputer.transform(df[var_nan_impute])
################################
# clear variables to free RAM memory
del int_imputer, onehot_imputer

# 5. Create missing indicator flag and impute with median

float_nan_grouping = ["S_3", "D_42", "D_43", "D_46", "D_48", "D_49", "D_50", "P_3", "D_53", "S_7", "D_55", "D_56",
                      "B_13", "S_9", "D_59", "D_61", "D_62", "B_17", "D_69", "D_77", "S_22", "S_24", "S_25",
                      "D_102", "D_104", "D_105", "R_27", "S_27", "D_113", "D_115", "D_118", "D_119", "D_121", "D_123",
                      "D_131", "D_133", "D_141", "D_144"]
float_imputer = SimpleImputer(missing_values=-9999, strategy="median", add_indicator=True)
for column in float_nan_grouping:
    new_col = column + "_miss_flag"

    # load pickle with imputer for this column
    with open('./Imputers/float_imputed_{}.pickle'.format(column), 'rb') as file:
        float_imputer_fit = pickle.load(file)
    float_imputed = float_imputer_fit.transform(df[column].to_numpy().reshape(-1, 1))
    df[[column, new_col]] = float_imputed

df.loc[:, (df.columns.str.find("miss_flag") != -1)] = df.loc[:, (df.columns.str.find("miss_flag") != -1)].astype(int)

# 6. Change mostly NaN columns into missing flags
float_to_nanflag = ["D_73", "D_76", "R_9", "D_82", "B_29", "D_106", "R_26", "D_108", "D_110", "D_111", "B_39", "B_42",
                    "D_132", "D_134", "D_135", "D_136", "D_137", "D_138", "D_142"]
df.loc[:, float_to_nanflag] = (df.loc[:, float_to_nanflag] == -9999).astype(int)

# 7. If there are less than 13 snapshots per customer - remove such values
customers_counter = df.iloc[:, [0, 1]].groupby("customer_ID").count()
sel_customers = customers_counter.loc[(customers_counter.target == 13)].reset_index().drop(columns="target")
df = df.loc[df.customer_ID.isin(sel_customers.customer_ID), :]
df.to_parquet(data_path + "/test_data_for_LSTM.parquet")
