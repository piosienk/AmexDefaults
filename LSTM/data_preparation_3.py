from LSTM.LSTM_helpers import prepare_lstm_data_3

# define inputs and outputs
train_input = "./Additional_data/intermediate_train_data_split_reshaping.parquet"
train_output = "./Additional_data/data_train_LSTM.pickle"

valid_input = "./Additional_data/intermediate_validation_data_split_reshaping.parquet"
valid_output = "./Additional_data/data_valid_LSTM.pickle"

test_input = data_path + "/test_data_for_LSTM.parquet"
test_output = "./Additional_data/data_test_LSTM.pickle"

# Prepare train, validation and test datasets
prepare_lstm_data_3(train_input, train_output)
prepare_lstm_data_3(valid_input, valid_output)
prepare_lstm_data_3(test_input, test_output)