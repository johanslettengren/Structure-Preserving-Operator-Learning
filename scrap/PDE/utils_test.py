import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from scipy.fftpack import dst, idst

from torch.autograd.functional import jacobian
sys.path.append('..')  # Go up one directory to where utils.py is located
from utils import *




# Fully connected neural network
class FNN(torch.nn.Module):
    def __init__(self, layer_sizes, N, activation='tanh', mode='vanilla'):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus' : torch.nn.Softplus()}[activation]        
                
        self.N = N
        
        self.mode = mode
        
        if self.mode == 'vanilla':
            layer_sizes = [1] + layer_sizes + [(2*N+1)*(2*N+1)]
            
        else:
            layer_sizes = [1] + layer_sizes + [(2*N+1)*(2*N+1)]
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
            
        z = self.linears[-1](z)
        
        if self.mode == 'vanilla':
            z = z.view(-1, 2*self.N+1, 2*self.N+1)
        else:
            z = z.view(-1, 2*self.N+1, 2*self.N+1)
        return z

# Spectral DeepONet (levrages Fourier series representation of solution)
class SDeepONet(torch.nn.Module):
    def __init__(self, layer_sizes, N, activation='softplus', mode='vanilla'):
        super(SDeepONet, self).__init__()
        
        torch.manual_seed(0)
        
        # Final output dimension
        self.N = N
        # Initialize branch network 
        self.net = FNN(layer_sizes, N=N, activation=activation, mode=mode)
        
        self.k = torch.arange(-self.N, self.N+1, dtype=torch.float32)[None,:]
        self.l = torch.arange(-self.N, self.N+1, dtype=torch.float32)[:,None]
        self.x = torch.linspace(0, 1, 2*self.N+1, dtype=torch.float32)[:,None]
        self.t = torch.linspace(0, 1, 2*self.N+1, dtype=torch.float32)[None, None, :]
    
        self.ekx = (torch.sin(np.pi * self.k * self.x))[None,...]
        
        #(torch.exp(np.pi * 1j * self.k * self.x) * 2**(self.k > 0.5))[None,...]
        
        
        
        self.flt = lambda c : torch.sin(np.pi * c[..., None] * self.l[None,...] * self.t) +\
            torch.cos(np.pi * c[..., None] * self.l[None,...] * self.t)
        
        self.mode = mode
    
    # DeepONet forward pass
    def forward(self, c):
        
        
        if self.mode == 'vanilla':
            u = self.net(c)
        elif self.mode == 'fourier':
            w = torch.view_as_complex(self.net(c))
            v = torch.fft.ifft(w, axis=1)
            u = torch.fft.irfft(v, n=2*self.N+1)
        
        elif self.mode == 'pde':
            
        
            
        
            w = self.net(c)#.to(torch.cfloat)
            
            print(w)
            
            #w = torch.view_as_complex(self.net(c))

            
            
            
            # w_last_N = w[..., 1:].conj() 
            # w_flipped = torch.flip(w_last_N, dims=[-1])
            # w_full = torch.cat([w_flipped, w], dim=-1)
            

            
            
            v = torch.einsum("clk,oxk->clx", w, self.ekx)
            
            
            
            
            
            
            #v = torch.fft.irfft(w, n=2*self.N+1, axis=-1)
        
            
            f = self.flt(c)
        
        
            u = torch.einsum("clt,clx->ctx", f, v) 
    
        
        
        return u




    
    
class Model():

    def __init__(self, N, num_iv, layer_sizes, lr=0.001, mode='vanilla'):    
        
        torch.manual_seed(0)
        np.random.seed(0)
        
        # Number of test- and training-points
        self.num_iv = num_iv

        # 2N + 1 is the number of coefficients in use
        self.N = N
        
        # analytical solution
        self.f = lambda x, c, t: np.sin(np.pi * x) * np.cos(c * np.pi * t) + np.sin(4 * np.pi * x) * np.cos(4 * c * np.pi * t) / 2


  
        
        # Spectral DeepONet to be trained
        self.net = SDeepONet(layer_sizes=layer_sizes, N=N, mode=mode)
        
                
        # Training data
        targets, c = self.get_data(num_iv)
                
        self.x_train = self.format(c)
        self.y_train = targets
        
        # Testing data
        targets, c = self.get_data(num_iv)
        self.x_test = self.format(c)
        self.y_test = self.format(targets)
        
        self.mse = torch.nn.MSELoss() 
        self.loss_fn = self.mse      
            
        # Initialize optimizer (default adam)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
           self.optimizer,
          lr_lambda=lambda epoch: 0.5 ** (epoch // 10000)
        )
        
    
    # Format data as torch tensor with dtype=float32    
    def format(self, x, requires_grad=False):
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        return x.to(torch.float32).requires_grad_(requires_grad)

    # Generates data for training and testing
    def get_data(self, num_iv):
        
        x = np.linspace(0, 1, 2*self.N+1)[None, None, :]
        t = np.linspace(0, 1, 2*self.N+1)[None, :, None]
        c = np.random.uniform(0.1, 1, size=(num_iv, 1, 1))
    
        y = self.f(x, c, t)
        y = torch.tensor(y, dtype=torch.float32)
        
        return y, c[...,0]

    
    # Train network
    def train(self, iterations):
        
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** (epoch // 5000))
        
        self.steps = []   
        self.tlosshistory = []
        self.vlosshistory = [] 
        
        # Set net to train mode
        self.net.train(True)
        
        # Training process updates
        print('Step \t Train loss \t Test loss')

        for iter in range(iterations):
            
            
            # Train
            self.optimizer.zero_grad()
            prediction = self.net(self.x_train)
            
            
            loss = self.loss_fn(prediction, self.y_train)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Test
            if iter % 1000 == 999:
                # Set net to evalutation mode
                self.net.eval()
                
                # Don't calculate gradients
                with torch.no_grad():
                    outputs = self.net(self.x_test)  
                    vloss = self.loss_fn(outputs, self.y_test).item()
                    tloss = loss.item()
    
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
    def predict(self, c):
        
        c = c[:,None]
                
        output = self.net(self.format(c))
        # coeffs = torch.view_as_complex(output_coeffs)
        
        #output_func = np.fft.ifft(output_coeffs.detach(), n=2*N+1)
        
        return output
            
    