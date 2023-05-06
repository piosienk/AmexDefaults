import pandas as pd
import numpy as np
import scipy

def group_bins_int(df, column, n):
    """
    Create bins for int variables

    :param df: dataframe with training set
    :param column: column to perform binning on
    :param n: number of bins
    :return: created bins tresholds
    """
    dist_bins = df.loc[df[column] != -9999, [column, "target"]].groupby(column).count().reset_index()
    dist_bins = dist_bins.sort_values(by=column).reset_index(drop=True)
    dist_bins[column] = dist_bins[column].apply(lambda x: [x])
    dist_bins.columns = ["bin", "n"]

    while (dist_bins.n.min() < n) and (dist_bins.n.min() != dist_bins.n.max()):
        i = 0
        while i < len(dist_bins):
            if dist_bins["n"][i] >= n:
                i += 1;
            else:
                if (i == len(dist_bins) - 1):
                    for item in dist_bins.bin.iloc[i]:
                        dist_bins.bin.iloc[i - 1].append(item)
                    dist_bins.n.iloc[i - 1] += dist_bins.n.iloc[i]
                    dist_bins = dist_bins.drop(i).reset_index(drop=True)
                elif (i == 0):
                    for item in dist_bins.bin.iloc[i]:
                        dist_bins.bin.iloc[i - 1].append(item)
                    dist_bins.n.iloc[i + 1] += dist_bins.n.iloc[i]
                    dist_bins = dist_bins.drop(i).reset_index(drop=True)
                else:
                    j = i - 1 if dist_bins.n.iloc[i - 1] < dist_bins.n.iloc[i + 1] else i + 1

                    for item in dist_bins.bin.iloc[i]:
                        dist_bins.bin.iloc[j].append(item)
                    dist_bins.n.iloc[j] += dist_bins.n.iloc[i]
                    dist_bins = dist_bins.drop(i).reset_index(drop=True)

                i = 0

    return dist_bins


def t_value(vector, std_vector):
    """
    calculate t-value of a vector

    :param vector: vector
    :param std_vector: its standard deviation
    :return: list of t-values
    """
    vector = np.array(vector)

    n = len(vector)
    t_list = []
    for i in range(1, n):
        t = (vector[i] - vector[i - 1]) * np.sqrt(n) / std_vector
        t_list.append(t)

    return t_list


def t_pvalue(t, n):
    return scipy.stats.t.sf(abs(t), df=n) * 2


def merge_similar_bins(d, pvalue, df):
    """
    merge bins that have the same/very similar WoE value

    :param d: obtained t-value
    :param pvalue: threshold value
    :param df: dataset
    :return: dataset with merged bins
    """
    n = len(df)
    merge_flag = d < pvalue

    for i in range(1, n):
        if not merge_flag[i - 1]:
            for val in df.loc[i, :]["bin"]:
                df.loc[i - 1, :]["bin"].append(val)
            df.loc[i - 1, "n"] = df.loc[i - 1, :]["n"] + df.loc[i, :]["n"]
            df = df.drop(index=i).reset_index(drop=True)
            return df, 0

    return df, 1


def merge_similar_bins_cont(d, pvalue, df):
    """
    Variant of merging function dedicated to continuous variables
    :param d: obtained t-value
    :param pvalue: threshold value
    :param df: dataset
    :return: dataset with merged bins
    """
    n = len(df)
    merge_flag = d < pvalue

    for i in range(1, n):
        if not merge_flag[i - 1]:
            if (df.loc[i - 1, :]["bin"] == -9999) or (df.loc[i, :]["bin"] == -9999):
                continue
            left_val = df.loc[i - 1, :]["bin"].left
            right_val = df.loc[i, :]["bin"].right
            interval_new = pd.Interval(left_val, right_val, closed="right")

            df["bin"] = df.bin.cat.rename_categories({df.loc[i - 1, "bin"]: interval_new})
            df["bin"] = df.bin.cat.remove_unused_categories()
            df.loc[i - 1, "n"] = df.loc[i - 1, :]["n"] + df.loc[i, :]["n"]
            df = df.drop(index=i).reset_index(drop=True)

            return df, 0

    return df, 1


def perform_merging_int(df, column, k, alpha):
    """
    Performs a whole process of calculating WoE and merging of similar bins iteratively

    :param df: dataset with training examples
    :param column: variable to modify
    :param k: number of bins
    :param alpha: p-value for merging t-test
    :return: modified column variable
    """
    # Initial binning
    binning = group_bins_int(df, column, k)
    if sum(df[column].isna()) != 0:
        binning = binning.append({"bin": [np.nan], "n": sum(df[column].isna())}, ignore_index=True)

    # Assign bins to observations and add target variable
    df_temp = pd.DataFrame(df[column])
    df_temp["Bin"] = None
    for index, row in binning.iterrows():
        df_temp.loc[df_temp[column].isin(row[0]), "Bin"] = index
    df_temp["target"] = df["target"]

    # Calcualte WoE for each bucket
    if isinstance(df[column][0], np.int16):
        distribution_bins = df_temp.loc[:, ["Bin", "target"]].groupby("target").count()
        bad_customers = df_temp.loc[
                            df_temp.target == 1, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                            1, "Bin"]
        good_customers = df_temp.loc[
                             df_temp.target == 0, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                             0, "Bin"]

        df_woe = np.log(good_customers / bad_customers).reset_index()
        df_woe.columns = ["Bin", "woe"]

    #     print(df_woe.head(10))

    std_vector = np.std(df_woe.woe) / np.sqrt(len(df_woe))

    d = t_pvalue(np.array(t_value(df_woe.woe, std_vector)), len(df_woe))
    binning, flag = merge_similar_bins(d, alpha, binning)

    # Stop flag
    flag = 0

    # Merging bins in a loop
    while flag != 1:

        # Assign bins to observations and add target variable
        df_temp = pd.DataFrame(df[column])
        df_temp["Bin"] = None
        for index, row in binning.iterrows():
            df_temp.loc[df_temp[column].isin(row[0]), "Bin"] = index
        df_temp["target"] = df["target"]

        # Calcualte WoE for each bucket
        if isinstance(df[column][0], np.int16):
            distribution_bins = df_temp.loc[:, ["Bin", "target"]].groupby("target").count()
            bad_customers = df_temp.loc[
                                df_temp.target == 1, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                                1, "Bin"]
            good_customers = df_temp.loc[
                                 df_temp.target == 0, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                                 0, "Bin"]

            df_woe = np.log(good_customers / bad_customers).reset_index()
            df_woe.columns = ["Bin", "woe"]

        # Once again calcualte std_vector
        std_vector = np.std(df_woe.woe) / np.sqrt(len(df_woe))

        d = t_pvalue(np.array(t_value(df_woe.woe, std_vector)), len(df_woe))
        binning, flag = merge_similar_bins(d, alpha, binning)

    # Last loop - add missing data bin
    if (df[column] == -9999).any():
        binning = binning.append({"bin": [-9999], "n": sum(df[column] == -9999)}, ignore_index=True)

        # Assign bins to observations and add target variable
        df_temp = pd.DataFrame(df[column])
        df_temp["Bin"] = None
        for index, row in binning.iterrows():
            df_temp.loc[df_temp[column].isin(row[0]), "Bin"] = index
        df_temp["target"] = df["target"]

        if isinstance(df[column][0], np.int16):
            distribution_bins = df_temp.loc[:, ["Bin", "target"]].groupby("target").count()
            bad_customers = df_temp.loc[
                                df_temp.target == 1, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                                1, "Bin"]
            good_customers = df_temp.loc[
                                 df_temp.target == 0, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                                 0, "Bin"]

            df_woe = np.log(good_customers / bad_customers).reset_index()
            df_woe.columns = ["Bin", "woe"]

    return binning, df_woe


## group continous data
def create_bins_continous(df, column, n):
    """
    Create bins for continuous variables

    :param df: dataframe with training set
    :param column: column to perform binning on
    :param n: number of bins
    :return: created bins thresholds
    """
    data = df.loc[:, column]
    data = data.replace(-9999, np.nan)
    k = round(len(data) / n)
    binned = pd.DataFrame(pd.qcut(data, q=k, duplicates="drop"))
    binned.columns = ["bin"]

    if binned["bin"].isna().any():
        binned["bin"] = binned.bin.cat.add_categories(-9999)
        binned.loc[binned["bin"].isna(), :] = -9999

    return binned


def perform_merging_float(df, column, k, alpha):
    """
    Performs a whole process of calculating WoE and merging of similar bins iteratively (continuous variable)

    :param df: dataset with training examples
    :param column: variable to modify
    :param k: number of bins
    :param alpha: p-value for merging t-test
    :return: modified column variable
    """
    if isinstance(df[column][0], np.int16):
        raise Exception("int column not applicable in perform_merging_float function")

    # Initial creation of bins
    df_temp = create_bins_continous(df, column, k)
    binning = df_temp.reset_index().groupby("bin").count().reset_index()
    binning.columns = ["bin", "n"]
    df_temp = pd.concat([df_temp, df["target"]], axis=1)
    df_temp.columns = ["Bin", "target"]

    # Calcualte WoE for each bucket
    distribution_bins = df_temp.loc[:, ["Bin", "target"]].groupby("target").count()
    bad_customers = df_temp.loc[
                        df_temp.target == 1, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[1, "Bin"]
    good_customers = df_temp.loc[
                         df_temp.target == 0, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                         0, "Bin"]

    df_woe = np.log(good_customers / bad_customers).reset_index()
    df_woe.columns = ["Bin", "woe"]

    std_vector = np.std(df_woe.woe) / np.sqrt(len(df_woe))

    d = t_pvalue(np.array(t_value(df_woe.woe, std_vector)), len(df_woe))
    binning, flag = merge_similar_bins_cont(d, alpha, binning)

    # Stop flag
    flag = 0

    # Merging bins in a loop
    while flag != 1:

        # Assign bins to observations and add target variable
        interval_df = pd.DataFrame(binning.bin)

        left_list = []
        right_list = []
        for index, row in interval_df.iterrows():
            try:
                left_list.append(row.bin.left)
                right_list.append(row.bin.right)
            except:
                left_list.append(-9999)
                right_list.append(-9999)

        interval_df["from"] = left_list
        interval_df["to"] = right_list

        interval_mapping = pd.IntervalIndex.from_arrays(left=interval_df["from"],
                                                        right=interval_df["to"], closed="right")
        interval_df = interval_df.set_index(interval_mapping)["bin"]

        df_temp["Bin"] = df[column].map(interval_df).astype(str)
        df_temp["Bin"] = df_temp["Bin"].fillna("-999")

        # Calcualte WoE for each bucket
        distribution_bins = df_temp.loc[:, ["Bin", "target"]].groupby("target").count()
        bad_customers = df_temp.loc[
                            df_temp.target == 1, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                            1, "Bin"]
        good_customers = df_temp.loc[
                             df_temp.target == 0, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                             0, "Bin"]

        df_woe = np.log(good_customers / bad_customers).reset_index()
        df_woe.columns = ["Bin", "woe"]

        # Once again calcualte std_vector
        std_vector = np.std(df_woe.woe) / np.sqrt(len(df_woe))

        d = t_pvalue(np.array(t_value(df_woe.woe, std_vector)), len(df_woe))
        binning, flag = merge_similar_bins_cont(d, alpha, binning)

    # Last loop - add missing data bin
    if (df[column] == -9999).any():

        # Assign bins to observations and add target variable
        interval_df = pd.DataFrame(binning.bin)

        left_list = []
        right_list = []
        for index, row in interval_df.iterrows():
            try:
                left_list.append(row.bin.left)
                right_list.append(row.bin.right)
            except:
                left_list.append(-9999)
                right_list.append(-9999)

        interval_df["from"] = left_list
        interval_df["to"] = right_list

        interval_mapping = pd.IntervalIndex.from_arrays(left=interval_df["from"],
                                                        right=interval_df["to"], closed="right")
        interval_df = interval_df.set_index(interval_mapping)["bin"]
        df_temp["Bin"] = df[column].map(interval_df).astype(str)
        df_temp["Bin"] = df_temp["Bin"].fillna("-999")

        # Calcualte WoE for each bucket
        distribution_bins = df_temp.loc[:, ["Bin", "target"]].groupby("target").count()
        bad_customers = df_temp.loc[
                            df_temp.target == 1, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                            1, "Bin"]
        good_customers = df_temp.loc[
                             df_temp.target == 0, ["Bin", "target"]].groupby("Bin").count() / distribution_bins.loc[
                             0, "Bin"]

        df_woe = np.log(good_customers / bad_customers).reset_index()
        df_woe.columns = ["Bin", "woe"]

    return binning, df_woe, interval_df


def calculate_diff_in_clients(df_x, first_snapshot, x_variables):
    """

    :param df_x: main dataset
    :param first_snapshot: first observations of customer
    :param x_variables: variables that should be included in this calculation
    :return:
    """
    df_diff = pd.DataFrame(index=df_x.index)

    for var in x_variables:
        diff = np.where(df_x[var] == -9999, np.nan, df_x[var]) - np.where(first_snapshot[var] == -9999, np.nan,
                                                                          first_snapshot[var])
        var_diff = var + "_diff"
        df_diff[var_diff] = diff

    return df_diff
