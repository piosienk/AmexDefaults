import gc
import os

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 200)

if __name__ == "__main__":

    # Settings
    np.random.seed(123)
    data_path = "/home/piotr/Jupyter/MasterThesis/Data"
    labels_path = "/home/piotr/Jupyter/MasterThesis/Labels"

    data_processing = True
    logistic = False
    lstm = False
    ews = False


    if data_processing:

        # 1. Main data processing
        os.chdir("./Data_extraction")
        # Execute data processing step 1 - Merging and initial exploration
        with open("./data_exploration_1.py", "r") as file:
            code = compile(file.read(), "./data_exploration_1.py", "exec")
            exec(code)
        gc.collect()

        # Execute data processing step 2 - remove noise based on additional analysis
        with open("./noise_removal_2.py", "r") as file:
            code = compile(file.read(), "./noise_removal_2.py", "exec")
            exec(code)
        gc.collect()
        # Execute data processing step 3 - Split data into training and test samples
        with open("./data_split_3.py", "r") as file:
            code = compile(file.read(), "./data_split_3.py", "exec")
            exec(code)
        gc.collect()

        # 2. Logistic Regression data processing
        os.chdir("./Logistic")
        # Execute Logistic Regression step 1 - Prepare data using WoE approach
        with open("./prepare_data_logistic.py", "r") as file:
            code = compile(file.read(), "./prepare_data_logistic.py", "exec")
            exec(code)

        # 3. LSTM data processing
        os.chdir("./LSTM")
        # Execute Logistic Regression step 1 - Prepare data using WoE approach
        with open("./data_preparation_1.py", "r") as file:
            code = compile(file.read(), "./data_preparation_1.py", "exec")
            exec(code)


    if logistic:

        os.chdir("./Logistic")

        # Execute Logistic Regression step 2 - Fit, transform and test Logistic Model
        with open("./logistic_regression_model.py", "r") as file:
            code = compile(file.read(), "./logistic_regression_model.py", "exec")
            exec(code)



