import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Import the first X table
df = pd.read_parquet(data_path + "/Raw/train_data_part1.parquet")
file_list = sorted(os.listdir(data_path + "/Raw/"))

# Import Y table and right join with first X table
train_labels = pd.read_csv(labels_path + "/train_labels.csv")
df = train_labels.merge(df, on="customer_ID", how="right")

# Load other X tables, join with Y and concat with main table
for file in file_list[1:]:
    path_file = os.path.join(data_path + "/Raw", file)
    df_new = pd.read_parquet(path_file)
    df_new = train_labels.merge(df_new, on="customer_ID", how="right")
    df = pd.concat([df,df_new],axis=0)

    # to reduce RAM usage
    del df_new
    gc.collect()


# Create observations per customer counter
# Should be run in Jupyter Notebook in case of memory issues (Jupyter consumes less RAM than PyCharm)
df["customer_obs"] = df.groupby(["customer_ID"]).cumcount() + 1

# To save
df.to_parquet("train_data_merged.parquet")

# Check split by the length of the series
plt.subplots(figsize=(10, 5))
series_length = pd.DataFrame(df.groupby("customer_ID")["customer_obs"].max()).reset_index()
series_length_perc = series_length.groupby("customer_obs").count() / len(series_length) * 100
plot = plt.bar(series_length_perc.index, series_length_perc.customer_ID)
plt.title('Percentage of all observations split by the length of series')
plt.xticks(np.arange(1, 15))

i = 0
for p in plot:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x + width / 2,
             y + height + 1,
             str(round(series_length_perc.customer_ID.iloc[i], 1)) + '%',
             ha='center',
             weight=2)
    i += 1

plt.show()

# Find time dependent and static variables
unique_values = pd.DataFrame(pd.DataFrame(df.iloc[:1001].groupby(["customer_ID"]).nunique()).mean()).reset_index()
unique_values.columns = ["variable", "unique"]

missing_values = pd.DataFrame(df.iloc[:1001].groupby(["customer_ID"]).count().mean()).reset_index()
missing_values.columns = ["variable", "non_missing"]

df_unique = unique_values.iloc[1:,:].merge(missing_values)
df_unique["static_values"] = df_unique.unique != df_unique["non_missing"]
df_static_variables = df_unique.loc[df_unique.static_values == True,:]

# Analyze distributions for static variables
static_columns = df_static_variables.variable.iloc[:-1] # the last one D_128 is not really static, look at its values
print("Static columns: ")
print(static_columns.values)

for i, column in enumerate(static_columns):

    values_counts = pd.DataFrame(df[column].value_counts(dropna=False, normalize=True)).reset_index()

    if i % 2 == 0:
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        axs[0].bar(values_counts.index, values_counts[column] * 100)
        axs[0].set_title(column)
        axs[0].yaxis.set_major_formatter(mtick.PercentFormatter())
        axs[0].set_xticks(values_counts.index)
        axs[0].set_xticklabels(values_counts["index"])
    else:
        axs[1].bar(values_counts.index, values_counts[column] * 100)
        axs[1].set_title(column)
        axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())
        axs[1].set_xticks(values_counts.index)
        axs[1].set_xticklabels(values_counts["index"])

        plt.show()

# Analyze distributions for continous variables
non_continous = ["customer_ID", "target", "S_2", "customer_obs", "D_87"]
non_continous = static_columns.to_list() + non_continous

continous_columns = df.drop(non_continous, axis=1).columns

for i, column in enumerate(continous_columns):

    columns_values = df[column]
    columns_values = columns_values[columns_values < columns_values.quantile(.99)]

    if i % 2 == 0:
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        axs[0].set_yscale('log')
        axs[0].hist(columns_values, bins=100)
        axs[0].set_title(column)

    else:
        axs[1].set_yscale('symlog')
        axs[1].hist(columns_values, bins=100)
        axs[1].set_title(column)

        plt.show()