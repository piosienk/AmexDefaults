import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_parquet(data_path + "/train_data_denoised.parquet")

# Perform split - 15% goes to test, rest to the training
X_train, X_test= train_test_split(df.customer_ID.unique(), test_size=0.15, random_state=123)
df_train = df.loc[df.customer_ID.isin(X_train)]
df_test = df.loc[df.customer_ID.isin(X_test)]

print("Default rate in training set: ", df_train.groupby(["customer_ID"])["target"].mean().mean())
print("Default rate in test set: ",df_test.groupby(["customer_ID"])["target"].mean().mean())

del df
gc.collect()

# Select 14,000 defaults in the training sample
print("Shape of the train data before adjustments: ", df_train.shape)
df_y = pd.DataFrame(df_train.groupby(["customer_ID"])["target"].mean()).reset_index()

to_keep_bad = df_y.loc[df_y.target ==1,:].sample(14000, random_state=123).customer_ID
to_remove_bad = df_train.loc[(~df_train.customer_ID.isin(to_keep_bad)) & (df_train.target == 1),:].reset_index(
                                                                                                            drop=True)

df_train_adjusted = df_train.loc[(df_train.customer_ID.isin(to_keep_bad)) | (df_train.target == 0),:].reset_index(
                                                                                                            drop=True)

del df_y, df_train
gc.collect()

df_train_adjusted.drop_duplicates().reset_index()
print("Shape of the train data after adjustments: ", df_train_adjusted.shape)

# Other defaults will be moved to testing sample
print("Shape of the test data before adjustments: ", df_test.shape)
df_test_adjusted = pd.concat([df_test, to_remove_bad], axis=0).reset_index(drop=True)
print("Shape of the test data before adjustments: ", df_test_adjusted.shape)

## Default rate in training set is reduced to 4.90%, in test set it increases to 60.6%
print("Default rate in training set after adjustment: ",
      df_train_adjusted.groupby(["customer_ID"])["target"].mean().mean())
print("Default rate in test set after adjustment: ",
      df_test_adjusted.groupby(["customer_ID"])["target"].mean().mean())

df_train_adjusted.to_parquet(data_path + "/train_data_split.parquet")
df_test_adjusted.to_parquet(data_path + "/test_data_split.parquet")