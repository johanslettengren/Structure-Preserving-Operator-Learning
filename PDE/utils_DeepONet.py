import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class branch(torch.nn.Module):
    def __init__(self, layer_sizes, dim, K, activation):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh}[activation]
        self.K = K
        self.dim = dim
                
        layer_sizes = layer_sizes + [K*dim]
        # Create layers
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Initialize weights (normal distribution)
            torch.nn.init.xavier_normal_(layer.weight)
            # Initialize biases (zeros)
            torch.nn.init.zeros_(layer.bias)
            self.linears.append(layer)
        
    # Branch forward pass
    def forward(self, x_func):
        z = x_func
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
        z = self.linears[-1](z).view(-1, self.dim, self.K)
        
        return z
        
class trunk(torch.nn.Module):
    def __init__(self, layer_sizes, K, activation):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh}[activation]
        
        layer_sizes = layer_sizes + [K]
        
        # Create layers
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Initialize weights (normal distribution)
            torch.nn.init.xavier_normal_(layer.weight)
            # Initialize biases (zeros)
            torch.nn.init.zeros_(layer.bias)
            self.linears.append(layer)
        
    # Trunk forward pass
    def forward(self, x_loc):
        z = x_loc
        for linear in self.linears:
            z = self.activation(linear(z))
        
        return z

    
    
# DeepONet class 
class DeepONet(torch.nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, K=2, dim=2, activation='tanh'):
        super(DeepONet, self).__init__()
        
        # Final output dimension
        self.dim = dim
        # Initialize branch networks
        self.branch = branch(layer_sizes_branch, dim=dim, K=K, activation=activation) 
        # Initialize one trunk network
        self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)

    # DeepONet forward pass
    def forward(self, x_func, x_loc):
        
        # Get one output for each branch
        # Shape (#u, dim, K): K = branch & trunk output dim
        # #u is the same as the number of boundary values
        branch_outputs = self.branch(x_func)
        
        # Get trunk ouput
        trunk_outputs  = self.trunk(x_loc)
        # Shape (#t, K): K = branch & trunk output dim
        
        # Get network output when energy is not conserved
        output = torch.einsum("udK,tK->utd", branch_outputs, trunk_outputs)
        
        return output


# Compile and train model + data
class Model():
    
    def __init__(self, x_train, y_train, x_test, y_test, net, lr=0.001, Tmax=5):     
        
        # Training data
        self.x_train = (self.format(x_train[0]), self.format(x_train[1]))
        self.y_train = self.format(y_train)
        
        # Testing data
        self.x_test = (self.format(x_test[0]), self.format(x_test[1]))
        self.y_test = self.format(y_test)
        
        # Network
        self.net = net
        
        # Loss history
        self.tlosshistory = []  # training loss
        self.vlosshistory = []  # validation/test loss

        # Initialize optimizer (default adam)
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(net.parameters(), lr=lr)
        
        # Set loss function (MSE default)
        self.mse = torch.nn.MSELoss()
        self.loss_fn = lambda pred, target : torch.nn.MSELoss()(pred, target)
        
        
    # Format data as torch tensor with dtype=float32    
    def format(self, x, requires_grad=False):
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        return x.to(torch.float32).requires_grad_(requires_grad)
        
    # Train network
    def train(self, iterations):
        
        # Train step history
        self.steps = []
        
        # Set net to train mode
        self.net.train(True)
        
        # Training process updates
        print('Step \t Train loss \t Test loss')

        for iter in range(iterations):
            
            # Train
            self.optimizer.zero_grad()
            prediction = self.net(*self.x_train)
            loss = self.loss_fn(prediction, self.y_train)
            loss.backward()
            self.optimizer.step()
            tloss = loss.item()

            # Test
            if iter % 1000 == 999:
                # Set net to evalutation mode
                self.net.eval()
                
                # Don't calculate gradients
                with torch.no_grad():
                    outputs = self.net(*self.x_test)  
                    vloss = self.loss_fn(outputs, self.y_test).item()
    
                # Save loss history
                self.vlosshistory.append(vloss)
                self.tlosshistory.append(tloss)
                self.steps.append(iter)
                self.net.train(True)
                
                print('{} \t [{:.2e}] \t [{:.2e}]'.format(iter + 1, tloss, vloss))    
                
                
                        
    def plot_losshistory(self, dpi=100):
        # Plot the loss trajectory
        _, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.plot(self.steps, self.tlosshistory, '.-', label='Training loss')
        ax.plot(self.steps, self.vlosshistory, '.-', label='Test loss')
        ax.set_title("Training Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        plt.show()
        
        
    
    # Predict output using DeepONet
    def predict(self, x_func, x_loc):
        
        return self.net(self.format(x_func), self.format(x_loc))
                
        