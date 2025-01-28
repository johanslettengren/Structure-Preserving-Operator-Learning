import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
#from Fourier import *

class FNN(torch.nn.Module):
    def __init__(self, layer_sizes, N, K, activation='softplus', real=True):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus' : torch.nn.Softplus()}[activation]
        self.K = K
        
        self.num_coeffs = N+1 if real else 2*N+1
                
        layer_sizes = [self.num_coeffs] + layer_sizes + [2*self.num_coeffs]
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

class branch(torch.nn.Module):
    def __init__(self, layer_sizes, N, K, activation='softplus', real=True):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus' : torch.nn.Softplus()}[activation]
        self.K = K
        self.N = N
        
        self.num_coeffs = N+1 if real else 2*N+1
                
        # Input layer takes 2N+1 Fourier coefficients
        # Ouputs K times 2N+1 'branch vectors'
        layer_sizes = [2] + layer_sizes + [2*K*self.num_coeffs]
        
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
        for linear in self.linears:
            z = self.activation(linear(z))
        
        return z


class SDeepONet(torch.nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, N, K, activation='softplus', real=True, conserve=False):
        super(SDeepONet, self).__init__()
        
        torch.manual_seed(0)
        
        self.conserve = conserve
        
        # Final output dimension
        self.N = N
        # Initialize branch network 
        self.branch = branch(layer_sizes_branch, N=N, K=K, activation=activation, real=real) 
        # self.branches = torch.nn.ModuleList([FNN(layer_sizes_branch, N=N, K=K) for _ in range(K)])
        # Initialize trunk network
        self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)
        
        self.I = lambda a : 18 * a**3 * (np.tanh(a/2) - np.tanh(a/2)**3 / 3 - np.tanh(-a/2) + np.tanh(-a/2)**3 / 3 )
        
        d = torch.concatenate((torch.tensor([1]), np.sqrt(2)*torch.ones(N)))
        self.P = torch.diag(d).to(torch.complex64)
        self.P_inv = torch.diag(1/d).to(torch.complex64)



    # DeepONet forward pass
    def forward(self, x_func, x_loc):
        
        
        
        # Get one output for each branch (dim many)
        # Shape (#f, 2N+1, K): K = branch & trunk output dim
        # #f is the same as the number of boundary values
        branch_outputs = self.branch(x_func)
        # branch_outputs = torch.stack([branch(x_func) for branch in self.branches], axis=2)
        
        # Get trunk ouput (only one)  
        trunk_outputs  = self.trunk(x_loc)
        # Shape (#t, K): K = branch & trunk output dim
        
        
        
        if self.conserve:
        
            
        
            
            # turn 2-d vectors into complex numbers
            branch_outputs = torch.view_as_complex(branch_outputs)
    
            # orthogonalise branch outputs
            Q, R = torch.linalg.qr(self.P @ branch_outputs)
                        
            
            # Rescale (so that norm matches energy) and redefine branch outputs
            a = x_func[...,0]
            branch_outputs = (self.P_inv @ Q) * torch.sqrt(self.I(a)[:,None,None] / 2) * (2*self.N+1)
            #print(branch_outputs.shape)
            #print(branch_outputs.shape)
            
            #print(torch.sum(torch.abs(branch_outputs)**2, axis=1))
            #print(torch.linalg.norm(branch_outputs, axis=-1))
            
            #branch_outputs = torch.view_as_real(branch_outputs)
                    
            # Find the corresponding coordinates
            trunk_outputs = torch.einsum("ukK,tK->utK", R, trunk_outputs)
            # Normalise and redfine trunk outputs
            #trunk_outputs = trunk_outputs / torch.sqrt(torch.sum(torch.abs(trunk_outputs)**2, axis=2))[...,None]
            trunk_outputs = torch.nn.functional.normalize(trunk_outputs, dim=2)
            # print(trunk_outputs.shape)
            # print(branch_outputs.shape)
            # print(torch.sum(torch.abs(trunk_outputs)**2, axis=2))
            
            # Get corresponding network output (trunk_outputs has one more dimension than usual)
            output = torch.einsum("udK,utK->utd", branch_outputs, trunk_outputs)
            #print(output.shape)
            # print(torch.sum(torch.abs(output)**2, axis=2))
            output = torch.view_as_real(output)
            
        else:
            # Get network output when energy is not conserved
            output = torch.einsum("udKc,tK->utdc", branch_outputs, trunk_outputs)
        
        
        return output
    
    
    
    
class Model():

    def __init__(self, N, num_iv, num_t, net, lr=0.001):    
        
        torch.manual_seed(14)
        np.random.seed(14)
        
        self.N = N
        
        self.f = lambda x, a, t, x0: 3 * a**2 / np.cosh((a * (((x + x0) + 1 - a**2 * t) % 2 - 1)) / 2)**2
        
        # Training data
        target_coeffs, a, x0, t = self.get_data(num_iv, num_t)
                        
        self.x_train = (self.format(np.concatenate((a, x0), axis=1)), self.format(t))
        self.y_train = target_coeffs
        
        # Testing data
        target_coeffs, a, x0, t = self.get_data(num_iv, num_t)
        
        self.x_test = (self.format(np.concatenate((a, x0), axis=1)), self.format(t))
        self.y_test = self.format(target_coeffs)
        
        # Network
        self.net = net
        
        # Loss history
        self.tlosshistory = []  # training loss
        self.vlosshistory = []  # validation/test loss

        # Initialize optimizer (default adam)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: 0.5 ** (epoch // 1000)
        )
        
        self.mse = torch.nn.MSELoss() 
        
        # Set loss function 
        self.loss_fn = self.mse
        
        self.I = lambda a : 18 * a**3 * (np.tanh(a/2) - np.tanh(a/2)**3 / 3 - np.tanh(-a/2) + np.tanh(-a/2)**3 / 3 )

    
    
    def weighted_loss(self, pred, target):
        
        
       
        c = torch.ones_like(pred)
        c[...,0] = 0.1        
        pred, target = c*pred, c*target
        loss = self.mse(pred, target)
        return loss
  
         
    # Format data as torch tensor with dtype=float32    
    def format(self, x, requires_grad=False):
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        return x.to(torch.float32).requires_grad_(requires_grad)
    
    
    def analytical_solution(self, a, x0, t):
        
        x = np.linspace(-1, 1, 2*self.N+1)[None, None, :]
        a, x0, t = a[:, None, None], x0[:, None, None], t[None, :, None]
        
        return self.f(x, a, t, x0), x[0,0,:]

    def get_data(self, num_iv, num_t):
        
        #for i in range(num_samples):
        x = np.linspace(-1, 1, 2*self.N+1)[None, None, :]
        t = np.linspace(0, 1, num_t)[None, :, None]
        a = np.random.uniform(0, 1, size=(num_iv, 1, 1))
        x0 = np.random.uniform(-1, 1, size=(num_iv, 1, 1))
    
        coeffs = np.fft.rfft(self.f(x, a, t, x0), axis=-1)
                
        coeffs = torch.view_as_real(torch.tensor(coeffs, dtype=torch.cfloat)).to(torch.float32)
        
        return coeffs, a[...,0], x0[...,0], t[0,...]
    
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
            #self.scheduler.step()

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
        
    def plot_conservation(self, a, x0, dpi=100):
                    
        # Energy test boundary values
        analytical_integral =  self.I(a) 
        
        t = torch.linspace(0, 1, 100, requires_grad=True)[:,None]        
        iv =  torch.tensor([[a, x0]], dtype=torch.float32)
        
        # Calculate total energy
        u = self.net(iv, t)   
        u = torch.view_as_complex(u.contiguous()) / (2*self.N+1)
        #print(u.shape)
        #print(torch.sum(torch.abs(u)**2, axis=2))
        u = u[0,...]
        u = torch.concatenate((u, u[...,1:]), dim=1)
        
        #print(u)
  
        integral = 2*torch.sum(torch.abs(u)**2, axis=1).detach()
        #integral = 2*torch.linalg.norm(u, axis=-1).detach()
        t= t.detach()
        
        _, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        
        ax.set_title('$\int_{\mathbb{T}} u^2(x,t)dt$ over time for $(a, x_0)$=' + f'({a}, {x0})')
        ax.plot(t, torch.ones_like(t)*analytical_integral, alpha=0.5, linewidth=5, label='True')
        ax.plot(t, integral, '--', alpha=0.8, linewidth=3,  label='DeepONet')
        ax.legend()
        ax.grid(True)
        
        ax.set_xlabel("t")
        ax.set_ylabel("$\int_{\mathbb{T}} u^2(x, t)dx$")
        plt.show()
    
    # Predict output using DeepONet
    def predict(self, a, x0, t):
        
        a, x0, t = a[:,None], x0[:,None], t[:,None]
                
        output_coeffs = self.net(self.format(np.concatenate((a, x0), axis=1)), self.format(t))
        coeffs = torch.view_as_complex(output_coeffs)
        
        #output_func = np.fft.ifft(output_coeffs.detach(), n=2*N+1)
        
        return coeffs
            
    