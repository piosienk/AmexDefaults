import numpy as np
import pandas as pd
import pickle

# Load merged data
df = pd.read_parquet("./train_data_merged.parquet")
df.info(verbose=True)

with open('Additional_data/adjustment_dict.pickle', 'rb') as file:
    adjustment_dict = pickle.load(file)

with open('Additional_data/fractions_dict.pickle', 'rb') as file:
    fractions_dict = pickle.load(file)

# Perform noise removal based on dictionaries obtained during additional analysis
for key in fractions_dict.keys():
    interval = fractions_dict[key]
    try:
        adj = adjustment_dict[key]
    except:
        adj = 0

    i = 100
    while interval > 1 / i:
        i -= 1

    interval_final = 1 / i

    # Code missing value as -9999 (this value is out of scale of all variables)
    df[key] = (np.floor((df[key] + adj) / interval_final)).fillna(-9999).astype(np.int16)

df = df.fillna(-9999)
static_columns = ['D_66', 'D_68', 'B_30', 'B_31', 'B_38', 'D_114',
       'D_116', 'D_117', 'D_120', 'D_126', 'D_87']
df["D_63"] = df["D_63"].apply(lambda x: {"CR":0, "XZ":1, "XM":2, "CO":3, "CL":4, "XL":5}[x]).astype(np.int16)
df["D_64"] = df["D_64"].apply(lambda x: {None:-9999, "O":0, "-1":1, "R":2, "U":3, -9999:-9999}[x]).astype(np.int16)

for column in static_columns:
    df[column] = df[column].fillna(-9999).astype(np.int16)
df.to_parquet(data_path + "/train_data_denoised.parquet")