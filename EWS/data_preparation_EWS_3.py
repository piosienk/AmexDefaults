from EWS.EWS_helpers import prepare_ews_data_3

# Settings
# time_windows = [2, 3, 4, 5, 6] # Short model
time_windows = [7, 8, 9, 10, 11, 12] # Medium model

# define inputs and outputs
train_input = "../LSTM/Additional_data/intermediate_train_data_split_reshaping.parquet"
train_output = "./Additional_data/data_train_EWS_medium.pickle"

valid_input = "../LSTM/Additional_data/intermediate_validation_data_split_reshaping.parquet"
valid_output = "./Additional_data/data_valid_EWS_medium.pickle"

test_input = data_path + "/test_data_for_LSTM.parquet"
test_output = "./Additional_data/data_test_EWS_medium.pickle"

# Prepare train, validation and test datasets
prepare_ews_data_3(train_input, train_output, time_windows)
prepare_ews_data_3(valid_input, valid_output, time_windows)
prepare_ews_data_3(test_input, test_output, time_windows)
