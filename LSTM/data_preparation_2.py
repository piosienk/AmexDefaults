from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Read data prepared in step 1
df = pd.read_parquet(data_path + "/train_data_for_LSTM.parquet")

# Split data into training and validation (10%)
customers = df.customer_ID.unique()
cus_train, cus_valid = train_test_split(customers, test_size=0.1, random_state=0)

np.save("./Additional_data/customers_train.npy", cus_train)
np.save("./Additional_data/customers_valid.npy", cus_valid)

train = df.loc[df.customer_ID.isin(cus_train),:]
valid = df.loc[df.customer_ID.isin(cus_valid),:]

# Save the data after split
train.to_parquet("./Additional_data/intermediate_train_data_split_reshaping.parquet")
valid.to_parquet("./Additional_data/intermediate_validation_data_split_reshaping.parquet")
