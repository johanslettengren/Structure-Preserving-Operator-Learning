import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from Fourier import *

class branch(torch.nn.Module):
    def __init__(self, layer_sizes, N, K, activation='tanh', real=True):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh}[activation]
        self.K = K
        self.N = N
        
        self.num_coeffs = N+1 if real else 2*N+1
                
        # Input layer takes 2N+1 Fourier coefficients
        # Ouputs K times 2N+1 'branch vectors'
        layer_sizes = [self.num_coeffs] + layer_sizes + [K*self.num_coeffs]
        
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
        z = self.linears[-1](z).view(-1, self.num_coeffs, self.K)
        
        return z
        
class trunk(torch.nn.Module):
    def __init__(self, layer_sizes, K, activation='tanh'):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu':torch.relu, 'tanh':torch.tanh}[activation]
        
        layer_sizes = [1] + layer_sizes + [K]
        
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


class SDeepONet(torch.nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, K, N, activation='tanh', real=True):
        super(SDeepONet, self).__init__()
        
        # Final output dimension
        self.N = N
        # Initialize branch network 
        self.branch = branch(layer_sizes_branch, N=N, K=K, activation=activation, real=real) 
        # Initialize trunk network
        self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)


    # DeepONet forward pass
    def forward(self, x_func, x_loc):
        
        
        
        # Get one output for each branch (dim many)
        # Shape (#f, 2N+1, K): K = branch & trunk output dim
        # #f is the same as the number of boundary values
        branch_outputs = self.branch(x_func)
        
        # Get trunk ouput (only one)  
        trunk_outputs  = self.trunk(x_loc)
        # Shape (#t, K): K = branch & trunk output dim
        
        # Get network output when energy is not conserved
        output = torch.einsum("udK,tK->utd", branch_outputs, trunk_outputs)
        
        
        return output
    
    
    
    
class Model():

    def __init__(self, N, num_iv, num_t, net, lr=0.001):    
        
        self.N = N
        
        # Training data
        target_coeffs, a, x0, t = self.get_data(num_iv, num_t)
        input_coeffs = self.inital_values(a, x0)
        
        self.x_train = (self.format(input_coeffs), self.format(t))
        self.y_train = self.format(target_coeffs)
        
        # Testing data
        target_coeffs, a, x0, t = self.get_data(num_iv, num_t)
        input_coeffs = self.inital_values(a, x0)
        
        self.x_test = (self.format(input_coeffs), self.format(t))
        self.y_test = self.format(target_coeffs)
        
        # Network
        self.net = net
        
        # Loss history
        self.tlosshistory = []  # training loss
        self.vlosshistory = []  # validation/test loss

        # Initialize optimizer (default adam)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        
        # Set loss function 
        self.loss_fn = torch.nn.MSELoss()
            
    
    # Format data as torch tensor with dtype=float32    
    def format(self, x, requires_grad=False):
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        return x.to(torch.float32).requires_grad_(requires_grad)
    

    def get_data(self, num_iv, num_t):

        f = lambda x, a, t, x0: 3*a**2 / np.cosh((a*(((x + x0) + 1 - a**2*t) % 2 - 1))/2)**2
        
        #for i in range(num_samples):
        x = np.linspace(-1, 1, 2*self.N+1)[None, None, :]
        t = np.linspace(0, 1, num_t)[None, :, None]
        a = np.random.uniform(10, 25, size=(num_iv, 1, 1))
        x0 = np.random.uniform(-1, 1, size=(num_iv, 1, 1))
        
        return np.fft.rfft(f(x, a, t, x0), axis=-1), a[...,0], x0[...,0], t[0,:,0]
    
    def inital_values(self, a, x0):

        f = lambda x, a, x0: 3 * a**2 / np.cosh((a * (((x + x0) + 1) % 2 - 1))/2)**2
        x = np.linspace(-1, 1, 2*self.N+1)[None, :]

        return np.fft.rfft(f(x, a, x0))

    
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
            #prediction_0 = self.net(self.x_train[0], torch.tensor([[0]], dtype=torch.float32)).squeeze(1)
            # loss_0 = self.loss_fn(prediction_0, self.x_train[0])
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
            
    