from functools import update_wrapper
import sys
from scipy.sparse import data


import torch
import torch.nn as nn
import torch.optim as optim

#  def neuro work
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(95, 50)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        #  self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        #  x = nn.LeakyReLU(x)
        x = self.fc2(x)
        x = self.relu(x)
        #  x = nn.LeakyReLU(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    #  l1 regularization
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

import math
from sklearn.metrics import roc_auc_score

def has_nan(tensor):
    """
    Check if the tensor contains NaN.

    Args:
        tensor: A tensor.

    Returns:
        True if the tensor contains NaN, False otherwise.
    """
    return torch.isnan(tensor).any().item()

#  train function
def train_neuro(net, optimizer, criterion, train_loader, test_loader, epochs, l1_weight):
    #  n = 1;
    upper_bound = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    lower_bound = torch.finfo(torch.float32).eps
    #  eps = 1e-6
    #  upper_bound = 1.0 - eps
    #  lower_bound = eps

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.squeeze()
            

            outputs = torch.where(outputs > upper_bound, upper_bound, outputs)
            outputs = torch.where(outputs < lower_bound, lower_bound, outputs)

            #  min_value = torch.min(outputs).item()
            #  max_value = torch.max(outputs).item()
            #  if min_value < 0 or max_value > 1:
            #      print(outputs)

            loss = criterion(outputs, labels)
            train_loss += loss.item()
            #  print(train_loss)

            to_regularise = []
            for param in net.parameters():
              to_regularise.append(param.view(-1))

            l1 = l1_weight* net.compute_l1_loss(torch.cat(to_regularise)) 
            loss += l1

            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)

        net.eval()
        test_loss = 0.0
        correct = 0
        tp = 0
        #  tp_fp = 0
        tp_fn = 0
        y_true_list = []
        y_score_list = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                #  if has_nan(inputs) or has_nan(labels):
                #      print("Error: NaN value detected.")
                outputs = net(inputs)
                outputs = outputs.squeeze()
                #  print(labels)
                #  print(outputs)

                outputs = torch.where(outputs > upper_bound, upper_bound, outputs)
                outputs = torch.where(outputs < lower_bound, lower_bound, outputs)
                #  print(outputs)
                #  print(labels)

                test_loss += criterion(outputs, labels).item()
                predicted = torch.round(outputs)
 
                correct += (predicted == labels).sum().item()
                tp += ((predicted == labels) & (labels ==1)).sum().item()
                tp_fn += (labels ==1).sum().item()
                #  tp_fp += (outputs == 1).sum().item()

                y_true_list.extend(labels.tolist())  
                y_score_list.extend(outputs.tolist())  
                
        test_loss /= len(test_loader.dataset) 
        test_acc = correct / len(test_loader.dataset)
        auc_roc = roc_auc_score(y_true_list, y_score_list)
        #  print(tp)
        #  print(tp_fn)
                
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Recall: {tp/tp_fn:.4f}, AUC_ROC: {auc_roc:.4f}')




# create dataloaders
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
#  from utils import normalization

def classifier(df,  model_path): 
    #  prepare the tensor
    X = df.drop('is_default', axis=1)
    y = df['is_default'].astype(int)

    X = X.to_numpy() 
    y = y.to_numpy() 

    #  #  load parameters
    #  parameters = pd.read_csv(norm_path)
    #  parameters = parameters[:-1]
    #  parametras = parameters.to_numpy()

    #  X , norm_parameters = normalization(X, parameters)

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=10, shuffle= False)

    #  load net
    net = Net()
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)


    #  compute is_default
    output_list = []
    net.eval()

    with torch.no_grad():
        for inputs, labels  in data_loader:
            y = net(inputs)
            #  print(inputs)
            #  print(y)
            output_list.append(y.detach().numpy())

    output_array = np.concatenate(output_list, axis=0)
    df = df.drop('is_default', axis = 1)
    result = pd.DataFrame(output_array, columns=['is_default'])

    return result



#  load data


import pandas as pd
import numpy as np

#  train_data = pd.read_csv('../../train_data_3.csv')
#  test_data = pd.read_csv('../../test_data_1.csv')


#preprocess test_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import argparse
import pandas as pd
import numpy as np


#  from tqdm import tqdm

#  from preprocess.my_gain import generator
#
