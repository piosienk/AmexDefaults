import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

# Define LSTM model
from LSTM.LSTM_helpers import calculate_accuracy


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True, bias=True)  # lstm
        self.fc1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc2 = nn.Linear(128, num_classes)  # fully connected 1
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device="cpu"

        x = x.to(device)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)  # hidden state
        nn.init.xavier_uniform_(h_0)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)  # internal state
        nn.init.xavier_uniform_(c_0)

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm1(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.reshape(-1, self.hidden_size)  # reshaping the data for Dense layer next
        x = self.dropout(F.relu(self.fc1(hn)))
        out = self.fc2(x)
        return out

    def train_network(self, epochs, epoch_size, train_loader, valid_loader, targets, valid_targets,
                      optimizer, criterion, scheduler=None, device='cpu', batch_n=128):

        train_accuracy_list = []
        validation_accuracy_list = []

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data.float()
                labels = targets[i * batch_n:(i + 1) * batch_n]
                labels = torch.IntTensor(labels)
                labels = labels.type(torch.LongTensor)

                labels = torch.reshape(labels, (-1,))
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                # loss = criterion(outputs, labels)
                loss = criterion(outputs[:,1].float(), labels.float(), alpha=0.25, reduction="sum")
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % epoch_size == (epoch_size - 1):  # print every epoch_size mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / epoch_size:.3f}')
                    running_loss = 0.0
            if scheduler is not None:
                scheduler.step()
            else:
                pass;

            train_accuracy_list.append(
                calculate_accuracy(self, train_loader, targets, device, data_type='train')[0])
            validation_accuracy_list.append(
                calculate_accuracy(self, valid_loader, valid_targets, device, data_type='valid')[0])

        print('Finished Training')
        return train_accuracy_list, validation_accuracy_list

    def train_validate_experiment(self, epochs, epoch_size, train_loader, valid_loader, targets, valid_targets,
                                  lr, scheduler_step_size, scheduler_gamma,
                                  class_weights, n_runs):

        overall_accuracy_list = []
        epoch_train_acc_list = []
        epoch_valid_acc_list = []

        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"

        for i in range(n_runs):
            print('Run nr {}'.format(i + 1))

            # criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
            criterion = sigmoid_focal_loss
            optimizer = optim.Adam(self.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
            acc_train_list, acc_valid_list = self.train_network(epochs, epoch_size, train_loader,
                                                                             valid_loader, targets, valid_targets,
                                                                             optimizer, criterion,
                                                                             scheduler, device)

            accuracy_overall = calculate_accuracy(
                self, valid_loader, valid_targets, device, data_type='valid')[0]

            overall_accuracy_list.append(accuracy_overall)

            epoch_train_acc_list.append(acc_train_list)
            epoch_valid_acc_list.append(acc_valid_list)

        return overall_accuracy_list, epoch_train_acc_list, epoch_valid_acc_list
