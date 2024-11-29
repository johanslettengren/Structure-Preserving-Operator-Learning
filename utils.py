import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

# Define network class (we use the same base structure for both branch and trunk)
class Network(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            #torch.nn.init.xavier_normal_(self.linears[-1].weight)
            #torch.nn.init.zeros_(self.linears[-1].bias)
        
    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = torch.relu(linear(x))
        x = self.linears[-1](x)
        return x
    
    
class DeepONet(nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, dim_out):
        super(DeepONet, self).__init__()
        
        #self.branches = torch.nn.ModuleList()
        #self.trunks = torch.nn.ModuleList()
        
        self.dim_out = dim_out
        
        # Initialize branch and trunk networks
        self.branches = torch.nn.ModuleList([Network(layer_sizes_branch) for _ in range(self.dim_out)]) 
        self.trunks =  torch.nn.ModuleList([Network(layer_sizes_trunk) for _ in range(self.dim_out)]) 
        
        # bias term
        self.bias = nn.Parameter(torch.zeros(self.dim_out, 1))

    
    def forward(self, x_func, x_loc):
        
        # Forward pass through the networks
        branch_outputs = torch.stack([branch(x_func) for branch in self.branches], dim=0)
        trunk_outputs = torch.stack([trunk(x_loc) for trunk in self.trunks], dim=0)
        
        # Combine the outputs
        #output = torch.sum(branch_outputs * trunk_outputs, dim=2) + self.bias
        output = torch.einsum('ijk,ijk->ij', branch_outputs, trunk_outputs) + self.bias
        
        return output




class Model():
    def __init__(self, x_train, y_train, x_test, y_test, net):     
        
        X_train = self.format_data(x_train[0], x_train[1])
        X_test = self.format_data(x_test[0], x_test[1])
        
        self.sensors = x_train[0].shape[1]
        self.num_training = x_train[0].shape[0] * x_train[1].shape[0]
        self.num_testing = x_test[0].shape[0] * x_test[1].shape[0]
        
        y_train_ = torch.tensor(y_train, dtype=torch.float32).flatten(start_dim=1, end_dim=2).T
        y_test_ = torch.tensor(y_test, dtype=torch.float32).flatten(start_dim=1, end_dim=2).T
                
        self.train_dataset = TensorDataset(X_train, y_train_) 
        self.test_dataset = TensorDataset(X_test, y_test_) 

        self.net = net
        self.vlosshistory = []
        self.tlosshistory = []
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        
    def format_data(self, x_func, x_loc):
        
        x_tensor = torch.tensor(x_func, dtype=torch.float32)
        y_tensor = torch.tensor(x_loc, dtype=torch.float32)
        
        n_x = x_tensor.shape[0]
        n_y = y_tensor.shape[0]

        # Compute all combinations
        x_expanded = x_tensor.unsqueeze(1)
        y_expanded = y_tensor.unsqueeze(0)
        combinations = torch.cat((x_expanded.expand(n_x, n_y, -1), y_expanded.expand(n_x, n_y, -1)), dim=-1)     
        X = combinations.flatten(start_dim=0, end_dim=1) 
        
        return X
                
    def train_one_epoch(self):
            
            running_loss = 0.
            iter = 0
            for data in self.train_loader:
                

                inputs, targets = data
                
                self.optimizer.zero_grad()
                outputs = self.net(inputs[:, :self.sensors], inputs[:, self.sensors:])            
                
                loss = self.loss_fn(outputs, targets.T)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                iter += 1

            return running_loss / iter
        
        
        
        
    def train(self, epochs, batch_size):
        
        test_batch = self.num_training if batch_size is None else batch_size
        train_batch = self.num_training if batch_size is None else batch_size
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=test_batch, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=train_batch, shuffle=True)

        
        self.epochs = []
        
        print('Epoch \t Train loss \t Test loss')
        
    
        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(epochs):
            
            self.net.train(True)
            avg_loss = self.train_one_epoch()

        
            if epoch % int(epochs / 10) == int(epochs / 10) - 1:
                self.net.eval()
                running_vloss = 0.0
                iter = 0
                with torch.no_grad():
                    for i, vdata in enumerate(self.test_loader):
                        vinputs, vtargets = vdata
                        voutputs = self.net(vinputs[:, :self.sensors], vinputs[:, self.sensors:])
                        vloss = self.loss_fn(voutputs, vtargets.T)
                        running_vloss += vloss
                        iter += 1
                

                avg_vloss = running_vloss / iter
                self.vlosshistory.append(avg_vloss)
                self.tlosshistory.append(avg_loss)
                self.epochs.append(epoch)
                
                print('{} \t {:.2e} \t {:.2e}'.format(epoch + 1, avg_loss, avg_vloss))
                
            
            #self.optimizer.param_groups[0]['lr'] *= 0.9
            
            
            
    def plot_losshistory(self):
        # Plot the loss trajectory
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.epochs, self.tlosshistory, label='Training loss')
        ax.plot(self.epochs, self.vlosshistory, label='Test loss')
        ax.set_title("Training Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        plt.show()
            
    def predict(self, x_func, x_loc):
        
        inputs = self.format_data(x_func, x_loc)
        return self.net(inputs[:, :self.sensors], inputs[:, self.sensors:])
                
        