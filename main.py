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
    default_path = os.getcwd()

    data_processing = False
    logistic = True
    lstm = False
    ews = False
    logistic_ews = False
    explain = False

    if data_processing:
        print("Executing data processing steps")
        ##################
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

        os.chdir(default_path)

        ##################
        # 2. Logistic Regression data processing
        os.chdir("./Logistic")

        # Execute Logistic Regression step 1 - Prepare data using WoE approach
        with open("./prepare_data_logistic.py", "r") as file:
            code = compile(file.read(), "./prepare_data_logistic.py", "exec")
            exec(code)

        os.chdir(default_path)

        ##################
        # 3. LSTM data processing

        os.chdir("./LSTM")

        # Execute LSTM step 1 of data processing for training/validation data
        with open("./data_preparation_1.py", "r") as file:
            code = compile(file.read(), "./data_preparation_1.py", "exec")
            exec(code)

        # Execute LSTM step 1 of data processing for test data
        with open("./data_preparation_1_test.py", "r") as file:
            code = compile(file.read(), "./data_preparation_1_test.py", "exec")
            exec(code)

        # Execute LSTM step 2 of data processing (only for train/validation)
        with open("./data_preparation_2.py", "r") as file:
            code = compile(file.read(), "./data_preparation_2.py", "exec")
            exec(code)

        # Execute LSTM step 3 of data processing for train, validation and testing
        with open("./data_preparation_3.py", "r") as file:
            code = compile(file.read(), "./data_preparation_3.py", "exec")
            exec(code)

        os.chdir(default_path)

        ##################
        # 3. EWS data processing

        os.chdir("./EWS")

        # Execute EWS step 1 of data processing for train, validation and testing (based on step 3 of LSTM)
        with open("./data_preparation_EWS_3.py", "r") as file:
            code = compile(file.read(), "./data_preparation_EWS_3.py", "exec")
            exec(code)

        os.chdir(default_path)

        # 4. Logistic approach to EWS data processing
        os.chdir("./Logistic_EWS")

        # Execute EWS step 1 of data processing for train, validation and testing (based on step 3 of LSTM)
        with open("./prepare_data_logistic_EWS.py", "r") as file:
            code = compile(file.read(), "./prepare_data_logistic_EWS.py", "exec")
            exec(code)

        os.chdir(default_path)

    if logistic:
        print("Executing logistic model steps")
        os.chdir("./Logistic")

        # Execute Logistic Regression model step 1 - Fit and test Logistic Model
        with open("./logistic_regression_model.py", "r") as file:
            code = compile(file.read(), "./logistic_regression_model.py", "exec")
            exec(code)

        # Execute Logistic Regression model step 2 - validate on the Test sample
        with open("./logistic_regression_test.py", "r") as file:
            code = compile(file.read(), "./logistic_regression_test.py", "exec")
            exec(code)

        os.chdir(default_path)

    if lstm:
        print("Executing LSTM model steps")
        os.chdir("./LSTM")

        # Execute LSTM model step 1 - Fit and test LSTM Model
        with open("./LSTM_run.py", "r") as file:
            code = compile(file.read(), "./LSTM_run.py", "exec")
            exec(code)

        # Execute LSTM model step 2 - validate on the Test sample (the same as Logistic Regressor)
        with open("./LSTM_test.py", "r") as file:
            code = compile(file.read(), "./LSTM_test.py", "exec")
            exec(code)

        os.chdir(default_path)

    if ews:
        print("Executing EWS model steps")
        os.chdir("./EWS")

        # # Execute EWS short model step 1 - Fit and test EWS short Model
        with open("./EWS_short_run.py", "r") as file:
            code = compile(file.read(), "./EWS_short_run.py", "exec")
            exec(code)

        # Execute EWS short model step 2 - validate on the Test sample (the same as Logistic Regressor)
        with open("./EWS_short_test.py", "r") as file:
            code = compile(file.read(), "./EWS_short_test.py", "exec")
            exec(code)

        # Execute EWS medium model step 1 - Fit and test EWS medium Model
        with open("./EWS_medium_run.py", "r") as file:
            code = compile(file.read(), "./EWS_medium_run.py", "exec")
            exec(code)

        # Execute EWS medium model step 2 - validate on the Test sample (the same as Logistic Regressor)
        with open("./EWS_medium_test.py", "r") as file:
            code = compile(file.read(), "./EWS_medium_test.py", "exec")
            exec(code)

        os.chdir(default_path)

    if logistic_ews:
        print("Executing logistic EWS model steps")
        os.chdir("./Logistic_EWS")

        # # Execute EWS short model step 1 - Fit and test EWS short Model
        # with open("./logistic_regression_model_EWS.py", "r") as file:
        #     code = compile(file.read(), "./logistic_regression_model_EWS.py", "exec")
        #     exec(code)
        #
        # # Execute EWS short model step 2 - validate on the Test sample (the same as Logistic Regressor)
        # with open("./logistic_regression_test_EWS.py", "r") as file:
        #     code = compile(file.read(), "./logistic_regression_test_EWS.py", "exec")
        #     exec(code)

        # Execute EWS medium model step 1 - Fit and test EWS medium Model
        # with open("./logistic_regression_model_EWS_medium.py", "r") as file:
        #     code = compile(file.read(), "./logistic_regression_model_EWS_medium.py", "exec")
        #     exec(code)

        # Execute EWS medium model step 2 - validate on the Test sample (the same as Logistic Regressor)
        with open("./logistic_regression_test_EWS_medium.py", "r") as file:
            code = compile(file.read(), "./logistic_regression_test_EWS_medium.py", "exec")
            exec(code)

        os.chdir(default_path)

    if explain:
        print("Executing explainability steps")
        os.chdir("./Explainability")

        # Execute Explainability step 1 - Integrated Gradient calculation
        # with open("./integrated_gradient.py", "r") as file:
        #     code = compile(file.read(), "./integrated_gradient.py", "exec")
        #     exec(code)

        # Execute Explainability step 2 - Feature Permutation
        # with open("./feature_permutations.py", "r") as file:
        #     code = compile(file.read(), "./feature_permutations.py", "exec")
        #     exec(code)

        # Execute Explainability step 3 - Feature Ablation
        with open("./feature_ablation.py", "r") as file:
            code = compile(file.read(), "./feature_ablation.py", "exec")
            exec(code)

        # Execute Explainability step 4 - Lime explanations
        # with open("./lime_explainer.py", "r") as file:
        #     code = compile(file.read(), "./lime_explainer.py", "exec")
        #     exec(code)

        os.chdir(default_path)


