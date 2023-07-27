import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

random_seed = 123
torch.manual_seed(random_seed)


# define our model: LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def load_seg(data_path, variable_name, test_data_size):
    # load the file flights.csv
    df = pd.read_csv(data_path, thousands=",")
    all_data = df[variable_name].values.astype(float)

    # segment the train and test
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]

    # do the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    return train_data_normalized, scaler


# define the function in order to train and predict
def create_inout_sequences(input_data, batch_size=12):
    inout_seq = []
    L = len(input_data)
    for i in range(L - batch_size):
        train_seq = input_data[i:i + batch_size]
        train_label = input_data[i + batch_size:i + batch_size + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def train(train_data_normalized, batch_size=12):
    train_inout_seq = create_inout_sequences(train_data_normalized, batch_size)
    # set the loss, optim and epoch
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 300
    # train
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    return model


def predict(model, train_data_normalized, scaler, batch_size=12, test_data_size=12):
    test_inputs = train_data_normalized[-batch_size:].tolist()
    model.eval()
    for i in range(test_data_size):
        seq = torch.FloatTensor(test_inputs[-batch_size:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[batch_size:]).reshape(-1, 1))
    return actual_predictions


def plot(test_data_size, df, variable_name, actual_predictions, save_path, title):
    x = np.arange(df.shape[0]-test_data_size, df.shape[0], 1)
    plt.title(title)
    plt.ylabel('Values')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(df[variable_name].values.astype(float), label="True value")
    plt.plot(x, actual_predictions, label="Predicted value")
    plt.legend()
    plt.savefig(save_path)
    #plt.show()


def loss(actual_predictions, df, variable_name, test_data_size):
    actual_predictions = actual_predictions.squeeze()
    actual_predictions = [round(x, 2) for x in actual_predictions]
    all_data = df[variable_name].values.astype(float)
    true_data = all_data[-test_data_size:]
    print(f"The true value of last {test_data_size}: {true_data}\n")
    print(f"The predicted value of last {test_data_size}: {actual_predictions}\n")
    print(f"均方误差 = {np.sum(np.square(true_data - actual_predictions))}")


data_path = "data/CNpop2018.csv"
variable_name = "pop"
test_data_size = 20
batch_size = 10
df = pd.read_csv(data_path, thousands=",")

train_data_normalized, scaler = load_seg(data_path, variable_name, test_data_size)
model = LSTM()
model = train(train_data_normalized, batch_size)
actual_predictions = predict(model, train_data_normalized, scaler, batch_size, test_data_size)
plot(test_data_size, df, variable_name, actual_predictions,
     save_path="output/China's pop.png", title="Time vs population")
loss(actual_predictions, df, variable_name, test_data_size)
