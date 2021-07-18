# Power Load forecasting

## Overview
The goal of this implementation is to create a model that can accurately predict the energy usage in the next hour given historical usage data. We will be using both the GRU and LSTM model to train on a set of historical data and evaluate both models on an unseen test set. To do so, we’ll start with feature selection, data-preprocessing, followed by defining, training and eventually evaluating the models.

## Implementation

### Preprocessing
* Getting the time data of each individual time step and generalizing them
    * Hour of the day i.e. 0-23
    * Day of the week i.e. 1-7
    * Month i.e. 1-12
    * Day of the year i.e. 1-365
* Scale the data to values between 0 and 1
    * Algorithms tend to perform better or converge faster when features are on a relatively similar scale and/or close to normally distributed
    * Scaling preserves the shape of the original distribution and doesn't reduce the importance of outliers.
* Group the data into sequences to be used as inputs to the model and store their corresponding labels
    * The sequence length or lookback period is the number of data points in history that the model will use to make the prediction
    * The label will be the next data point in time after the last one in the input sequence
* The inputs and labels will then be split into training and test sets

### Model
 ```
 class GRUNet(nn.Module):
     def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
         super(GRUNet, self).__init__()
         self.hidden_dim = hidden_dim
         self.n_layers = n_layers

         self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
         self.fc = nn.Linear(hidden_dim, output_dim)
         self.relu = nn.ReLU()

     def forward(self, x, h):
         out, h = self.gru(x, h)
         out = self.fc(self.relu(out[:,-1]))
         return out, h

     def init_hidden(self, batch_size):
         weight = next(self.parameters()).data
         hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
         return hidden

 class LSTMNet(nn.Module):
     def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
         super(LSTMNet, self).__init__()
         self.hidden_dim = hidden_dim
         self.n_layers = n_layers

         self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
         self.fc = nn.Linear(hidden_dim, output_dim)
         self.relu = nn.ReLU()

     def forward(self, x, h):
         out, h = self.lstm(x, h)
         out = self.fc(self.relu(out[:,-1]))
         return out, h

     def init_hidden(self, batch_size):
         weight = next(self.parameters()).data
         hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
         return hidden
 ```
### Results
Achieved an sMAPE of .2957% and .2981% on GRU and LSTM networks
<br> Te results can also be visualised as -
![result visualisation](https://github.com/AnunayGupta/Smart-Load-Forecasting/blob/3e5692a687a1c4c0b8ab563b480b88d90f7c090f/Static/Screenshot%202021-07-19%20at%201.13.20%20AM.png)
