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
    def __init__(self, layer_sizes, activation='tanh'):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus' : torch.nn.Softplus()}[activation]        
                
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
        return z

# Branch net
class branch(torch.nn.Module):
    def __init__(self, layer_sizes, N, K, Fourier, activation='softplus'):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus' : torch.nn.Softplus()}[activation]
        self.K = K
        self.N = N
        
        
        self.num_coeffs = N+1 if Fourier else 2*N+1
            
                
        # Input layer takes 2N+1 Fourier coefficients
        # Ouputs K times 2N+1 'branch vectors'
        layer_sizes = [1] + layer_sizes + [2*K*self.num_coeffs]
        
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
        z = self.linears[-1](z).view(-1, self.num_coeffs, self.K, 2)
        
        return z
        
# Trunk net 
class trunk(torch.nn.Module):
    def __init__(self, layer_sizes, K, activation='softplus'):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus' : torch.nn.Softplus()}[activation]

        
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
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
        z = self.linears[-1](z)
        
        return z

# Spectral DeepONet (levrages Fourier series representation of solution)
class SDeepONet(torch.nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, N, K, activation='relu', conserve=False, 
                 predict_fourier=False, Fourier=True):
        super(SDeepONet, self).__init__()
        
        torch.manual_seed(0)
        
        # Whether to enforce conservation law
        self.conserve = conserve
        
        # Final output dimension
        self.N = N
        # Initialize branch network 
        self.branch = branch(layer_sizes_branch, N=N, K=K, Fourier=Fourier, activation=activation) 
        # Initialize trunk network
        self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)
        
        self.alpha = torch.nn.Parameter(torch.ones(K))


        
        # Integral weight network
        # self.weight = parameter([2, 40, 40, 40, 1], activation=activation)
        
        #self.I = lambda a : 18 * a**3 * (np.tanh(a/2) - np.tanh(a/2)**3 / 3 - np.tanh(-a/2) + np.tanh(-a/2)**3 / 3 )
        
        
        #k = torch.arange(self.N + 1)
        
        # Weights for "P-matrix" in bilinear form
        self.A = torch.sqrt(torch.concatenate((torch.tensor([1]), 2*torch.ones(N)))).to(torch.cfloat)[None, :]
        self.B = 2*np.pi*np.sqrt(2)*torch.arange(N+1).to(torch.cfloat)[None, :]
        
        
        # Additional parameter (may or may nor be used)
        self.D = torch.nn.Parameter(torch.tensor([0.0]))
        
        # Whether to predict ceofficients or function values
        self.predict_fourier = predict_fourier
        
        # whether to predict Fourier coefficients or sine coefficients
        self.Fourier = Fourier
        
        self.fft = np.fft.rfft if Fourier else dst
        self.ifft = np.fft.irfft if Fourier else idst
    


    # DeepONet forward pass
    def forward(self, x_func, x_loc):
        
        
        
        # Get one output for each branch (dim many)
        # Shape (#f, 2N+1, K): K = branch & trunk output dim
        # #f is the same as the number of boundary values
        branch_outputs = self.branch(x_func)

            
        # If conservation law is enforced
        if self.conserve:
        
            # turn 2-d vectors into complex numbers
            # has shape (num_iv, num_coeffs, K)
            branch_outputs = torch.view_as_complex(branch_outputs)
    
            # orthogonalise branch outputs
            
            #Q, R = torch.linalg.qr(branch_outputs * self.d)
            
            
            # Orthonormalise wrt to bilinear form
            # branch_outputs = (Q / self.d) # maybe should multiply by 2N + 1?
            # # Rescale (so that norm matches energy) and redefine branch outputs
            # # branch_outputs *= (x_func * np.pi / np.sqrt(2))[..., None]
            # #print(torch.linalg.norm(branch_outputs*self.d, axis=1))
            # alpha_norm = torch.linalg.norm(R@self.trunk.alpha.to(torch.cfloat), axis=-1)[:, None]
            
            # #branch_outputs *= np.pi / np.sqrt(2) / alpha_norm
            
                        
            # #print((torch.linalg.norm(branch_outputs*self.d, axis=1)[...,None] * alpha_norm))
            # #print(torch.linalg.norm(branch_outputs*self.d, axis=1) * alpha_norm)
            # #print(torch.linalg.norm(R@self.trunk.alpha.to(torch.cfloat), axis=-1)[:, None, None].shape)
            
            # # Transform trunk coordinates to match new branch
            # trunk_outputs = torch.einsum("ukK,tK->utK", R, trunk_outputs) 
            # M = torch.linalg.norm(branch_outputs*self.w[None,:,None], axis=1) * self.trunk.alpha[None,:] 
            # M = torch.linalg.norm(M, axis=-1)[:, None]
            
            # f = np.pi * torch.sin(x_func*x_loc.T * torch.sqrt((alpha_norm**2 - 1) / M)) / x_loc.T / torch.sqrt(2*(alpha_norm**2 - 1))
            
            
            # #/ torch.sqrt(2*(alpha_norm**2 - 1)))[..., None]
            # trunk_outputs *= f[...,None]
            

            alpha = self.alpha.to(torch.cfloat)
            
            
            
            #print(alpha[None, None, :].shape)
            x = torch.sum(branch_outputs * alpha[None, None, :], axis=-1)
            
            a = torch.linalg.norm(self.A * x, axis=-1)[:, None]
            b = torch.linalg.norm(self.B * x, axis=-1)[:, None]
                        
                        
            y = np.pi * torch.sin(x_func * b * x_loc.T / a) / b / np.sqrt(2)
            #y_t = grad(y, x_loc, grad_outputs=torch.ones_like(y), create_graph=True)[0]

            trunk_outputs = (y.unsqueeze(-1) * self.alpha).to(torch.cfloat)
            
            
            # Get network output
            output = torch.einsum("cNK,ctK->ctN", branch_outputs, trunk_outputs)            
            
            def print_energy():
                c = x_func[0]
                N = output[0,0,:]

                grads_r = torch.stack([grad(Ni, x_loc, grad_outputs=torch.ones_like(Ni), create_graph=True)[0].squeeze(-1) for Ni in N.real])
                grads_i = torch.stack([grad(Ni, x_loc, grad_outputs=torch.ones_like(Ni), create_graph=True)[0].squeeze(-1) for Ni in N.imag])            
                grads = torch.view_as_complex(torch.concat((grads_r, grads_i), axis=-1))  
                
                norm_t = torch.linalg.norm(self.A * grads, axis=-1)
                norm_x = torch.linalg.norm(self.B * N, axis=-1)

                E_N = (norm_t**2 + c**2 * norm_x**2).item()
                E = (c**2 * np.pi**2 / 2).item()
                
                print(f'A: \t {E}.')
                print(f'N: \t {E_N}.')
            
            #print_energy()
        
            # Convert complex output to real output
            if self.predict_fourier:
                output = torch.view_as_real(output)
                return output
            
        else:
            
            # Get trunk ouput (only one)  
            trunk_outputs  = self.trunk(x_loc)
            # Shape (#t, K): K = branch & trunk output dim
            # Get network output when conservation law is not enforced
            output = torch.einsum("udKc,tK->utdc", branch_outputs, trunk_outputs)
            if self.predict_fourier:
                return output
            output = torch.view_as_complex(output.contiguous())
            

        
        #print(torch.linalg.norm(output, axis=-1))
        output = torch.fft.irfft(output, n=2*self.N+1) if self.Fourier else self.torch_idst(output)
        
        
        return output
    
    def torch_idst(self, X):
        """
        Compute the Inverse Discrete Sine Transform (IDST-I) using PyTorch FFT.
        Equivalent to scipy.fftpack.idst(X, type=1)
        """
        N = X.shape[-1]
        
        # Create the extended sequence for FFT trick
        X_ext = torch.cat([torch.tensor([0.0]), X, torch.tensor([0.0]), -X.flip(dims=[-1])], dim=-1)

        # Compute the FFT
        x_reconstructed = torch.fft.fft(X_ext, dim=-1)

        # Extract imaginary part and scale
        return (1 / (2 * N)) * x_reconstructed.imag[..., 1:N+1]





    
    
class Model():

    def __init__(self, N, K, num_iv, num_t, layer_sizes_branch, layer_sizes_trunk, conserve=False, lr=0.001, 
                 predict_fourier=False, Fourier=True):    
        
        torch.manual_seed(0)
        np.random.seed(0)
        
        # Number of test- and training-points
        self.num_iv = num_iv
        self.num_t = num_t

        # 2N + 1 is the number of coefficients in use
        self.N = N
        # Whether to enforce conservation law
        self.conserve = conserve
        # Whether to predict coefficients or function values
        self.predict_fourier = predict_fourier
        
        # Weights for "integration" via coefficients
        d = torch.concatenate((torch.tensor([1]), np.sqrt(2)*torch.ones(self.N)))
        # "Integration" via coefficients 
        self.integrate = lambda coeffs : 2*(torch.linalg.norm(d * coeffs, axis=2)**2).unsqueeze(-1)
        
        # Spectral DeepONet to be trained
        self.net = SDeepONet(layer_sizes_branch, layer_sizes_trunk, N=N, K=K, conserve=conserve, 
                             predict_fourier=predict_fourier, Fourier=Fourier)
        
        self.fft = self.net.fft
        
        # Calculates initial condition based on x, a, t and x0
        self.f = lambda x, c, t: np.sin(np.pi * x) * np.cos(c * np.pi * t) + np.sin(4 * np.pi * x) * np.cos(4 * c * np.pi * t) / 2
        
        # Training data
        targets, c, t = self.get_data(num_iv, num_t)
                
        self.x_train = (self.format(c), self.format(t, requires_grad=True))
        self.y_train = targets
        
        # Testing data
        targets, c, t = self.get_data(num_iv, num_t)
        self.x_test = (self.format(c), self.format(t))
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
    def get_data(self, num_iv, num_t):
        
        x = np.linspace(0, 1, 2*self.N+1)[None, None, :]
        t = np.linspace(1e-3, 1, num_t)[None, :, None]
        c = np.random.uniform(0.1, 1, size=(num_iv, 1, 1))
    
        y = self.f(x, c, t)
        
        # Convert to coefficients if Fourier coeffs are predictes
        if self.predict_fourier:
            coeffs = self.fft(y, axis=-1)             
            y = torch.view_as_real(torch.tensor(coeffs, dtype=torch.cfloat)).to(torch.float32)
        else:
            y = torch.tensor(y, dtype=torch.float32)
        
        return y, c[...,0], t[0,...]

    
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
            prediction = self.net(*self.x_train)
            
            #print(self.x_train[1])
            
            loss = self.loss_fn(prediction, self.y_train) #+ self.pinn_loss()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Test
            if iter % 1000 == 999:
                # Set net to evalutation mode
                self.net.eval()
                
                # Don't calculate gradients
                with torch.no_grad():
                    outputs = self.net(*self.x_test)  
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
    def predict(self, c, t):
        
        c, t = c[:,None], t[:,None]
                
        output = self.net(self.format(c), self.format(t))
        # coeffs = torch.view_as_complex(output_coeffs)
        
        #output_func = np.fft.ifft(output_coeffs.detach(), n=2*N+1)
        
        return output
            
    