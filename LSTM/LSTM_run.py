import pickle
import torch
from LSTM.LSTM_model import LSTM


##############

# Define setting for LSTM training
# Standard settings
# num_epochs = 5 # epochs
# learning_rate = 0.0002 # lr
# dropout=0.2

num_epochs = 6 # epochs
learning_rate = 0.0002 # lr
dropout=0.1

input_size = 237 #number of features
hidden_size = 200 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 2 #number of output classes
class_weights=[0.05, 0.95]

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")
###############
# Load data into data loaders
with open('./Additional_data/data_train_LSTM.pickle', 'rb') as file:
    data_train = pickle.load(file)

with open('./Additional_data/data_valid_LSTM.pickle', 'rb') as file:
    data_valid = pickle.load(file)

batch_n= 128
trainloader = torch.utils.data.DataLoader(data_train["x"][:], batch_size=batch_n,
                                          shuffle=False, num_workers=0)
targets = data_train["y"][:].reshape(-1).tolist()

validloader = torch.utils.data.DataLoader(data_valid["x"][:], batch_size=batch_n,
                                          shuffle=False, num_workers=0)
targets_valid = data_valid["y"][:].reshape(-1).tolist()

################
# perform training and validation
lstm_model = LSTM(num_classes, input_size, hidden_size, num_layers, dropout).to(device)

overall_accuracy_list_100, epoch_train_acc_list_100, epoch_valid_acc_list_100 = lstm_model.train_validate_experiment(
                                                                        num_epochs, batch_n, trainloader,
                                                                        validloader, targets, targets_valid, lr=learning_rate,
                                                                        scheduler_step_size = 2, scheduler_gamma=0.5,
                                                                        class_weights=class_weights, n_runs=1
                                                                        )
torch.save(lstm_model, "./Final_models/lstm.pickle")