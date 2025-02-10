import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.autograd.functional import jacobian
sys.path.append('..')  # Go up one directory to where utils.py is located
from utils import *






# Traininable parameter (a function of on any of a, x0 and t)
class parameter(torch.nn.Module):
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
        # n, m = x_func.shape[0], x_loc.shape[0]
        # x_func = x_func.repeat_interleave(m, dim=0)
        # x_loc = x_loc.repeat(n, 1)
        # z = torch.cat([x_func, x_loc], dim=1)

        z = x_func
        
        
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
        z = self.linears[-1](z)
        return z #z.view(n, m, 1)

# Branch net
class branch(torch.nn.Module):
    def __init__(self, layer_sizes, N, K, activation='softplus', real=True, pred_c0=True):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus' : torch.nn.Softplus()}[activation]
        self.K = K
        self.N = N
        
        
        self.num_coeffs = N+1 if real else 2*N+1
        
        if not pred_c0:
            self.num_coeffs -= 1
            
                
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
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, N, K, weight=None, activation='relu', 
                 real=True, conserve=False, predict_fourier=False, pred_c0=True, L=2):
        super(SDeepONet, self).__init__()
        
        torch.manual_seed(0)
        
        # Width of interval [-L/2, L/2]
        self.L = L
        # Whether to enforce conservation law
        self.conserve = conserve
        # Whether to predict zeroth coefficient (failed test)
        self.pred_c0 = pred_c0
        
        # Final output dimension
        self.N = N
        # Initialize branch network 
        self.branch = branch(layer_sizes_branch, N=N, K=K, activation=activation, real=real, pred_c0=pred_c0) 
        # self.branches = torch.nn.ModuleList([FNN(layer_sizes_branch, N=N, K=K) for _ in range(K)])
        # Initialize trunk network
        self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)
        
        # Integral weight network
        self.weight = parameter([2, 40, 40, 40, 1], activation=activation)
        
        #self.I = lambda a : 18 * a**3 * (np.tanh(a/2) - np.tanh(a/2)**3 / 3 - np.tanh(-a/2) + np.tanh(-a/2)**3 / 3 )
        
        
        #k = torch.arange(self.N + 1)
        
        # Weights for "P-matrix" in bilinear form
        w = torch.concatenate((torch.tensor([1]), 2*torch.ones(N)))
        self.d = torch.sqrt(w)[None, :, None]
        
        # Additional parameter (may or may nor be used)
        self.p = torch.nn.Parameter(torch.tensor([1.0]))
        
        # Whether to predict Fourier ceofficients or function values
        self.predict_fourier = predict_fourier



    # DeepONet forward pass
    def forward(self, x_func, x_loc):
        
        
        
        # Get one output for each branch (dim many)
        # Shape (#f, 2N+1, K): K = branch & trunk output dim
        # #f is the same as the number of boundary values
        branch_outputs = self.branch(x_func)
        
        # Get trunk ouput (only one)  
        trunk_outputs  = self.trunk(x_loc)
        # Shape (#t, K): K = branch & trunk output dim
            
        # If conservation law is enforced
        if self.conserve != False:
            
        
            # turn 2-d vectors into complex numbers
            # has shape (num_iv, num_coeffs, K)
            branch_outputs = torch.view_as_complex(branch_outputs)
    
            # orthogonalise branch outputs
            
            Q, R = torch.linalg.qr(branch_outputs * self.d)
            
            
            a = x_func[...,0]
            # Orthonormalise wrt to bilinear form
            branch_outputs = (Q / self.d) * (2*self.N+1) / np.sqrt(self.L)
            # Rescale (so that norm matches energy) and redefine branch outputs
            branch_outputs = branch_outputs * torch.sqrt((self.I2(a))[:, None, None])
            
            # Transform trunk coordinates to match new branch
            trunk_outputs = torch.einsum("ukK,tK->utK", R, trunk_outputs)
            # Normalise new trunk
            trunk_outputs = torch.nn.functional.normalize(trunk_outputs, dim=-1)
            
            # Whether to train branch norm or not (not currently in use)
            if self.conserve == 'trained':
                #branch_outputs = branch_outputs * 1.2 #(1+self.weight(x_func)[:, None])
                pass

            # Get network output
            output = torch.einsum("udK,utK->utd", branch_outputs, trunk_outputs)
        
            # Convert complex output to real output
            if self.predict_fourier:
                output = torch.view_as_real(output)
                return output
            
        else:
            # Get network output when conservation law is not enforced
            output = torch.einsum("udKc,tK->utdc", branch_outputs, trunk_outputs)
            if self.predict_fourier:
                return output
            output = torch.view_as_complex(output.contiguous())
            if not self.pred_c0:
                c0 = self.c0(x_func[:,0]).repeat(50,1)[...,None]
                output = torch.concatenate((c0, output), axis=-1)

        
        #print(torch.linalg.norm(output, axis=-1))
        output = torch.fft.irfft(output, n=2*self.N+1)
        
        return output
    
    # Calculates analytical integral of u^2 (based on initial condition)
    def I2(self, a):
        
        if isinstance(a, torch.Tensor):
            tanh = torch.tanh
        else:
            tanh = np.tanh

        return 18 * a**3 * (tanh(self.L*a/4) - tanh(self.L*a/4)**3 / 3 - (-tanh(self.L*a/4)) + (-tanh(self.L*a/4))**3 / 3)
    
    # Calculates analytical integral of u_x (not currently in use)
    def gradI(self, a):
        if isinstance(a, torch.Tensor):
            cosh, tanh = torch.cosh, torch.tanh
        else:
            cosh, tanh = np.cosh, np.tanh

        p_ = lambda x : (cosh(2*x) + 4) * tanh(x) ** 3 / cosh(x)**2
        
        return 18 * a**5 * (p_(a/2) - p_(-a/2)) / 15
    
    # Calculates analytical value of zeroth coefficient (not currently in use)
    def c0(self, a):
        if isinstance(a, torch.Tensor):
            tanh = torch.tanh
        else:
            tanh = np.tanh
        
        return 3 * a * (tanh(a/2) - tanh(-a/2)) * (2*self.N+1)
    
    
class Model():

    def __init__(self, N, K, num_iv, num_t, layer_sizes_branch, layer_sizes_trunk, conserve=False, lr=0.001, 
                 predict_fourier=False, pred_c0=True, L=2):    
        
        torch.manual_seed(14)
        np.random.seed(14)
        
        # Number of test- and training-points
        self.num_iv = num_iv
        self.num_t = num_t
        
        # Length of interval [-L/2, L/2]
        self.L = L

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
                             predict_fourier=predict_fourier, pred_c0=pred_c0, L=self.L)

        # Calculates analytical integral of u^2
        self.I2 = self.net.I2
        
        # Calculates initial condition based on x, a, t and x0
        self.f = lambda x, a, t, x0: 3 * a**2 / np.cosh((a * (((x + x0) + self.L/2 - a**2 * t) % self.L - self.L/2)) / 2)**2
        
        # Training data
        targets, a, x0, t = self.get_data(num_iv, num_t)
        x_train = self.format(np.concatenate((a, x0), axis=1))
        self.x_train = (x_train, self.format(t))
        self.y_train = targets
        
        # Collocation points (for PINN)
        targets, a, x0, t = self.get_data(5, 5)
        x_col = self.format(np.concatenate((a, x0), axis=1))     
        self.x_col = (x_col, self.format(t, requires_grad=True))
        
        # Testing data
        targets, a, x0, t = self.get_data(num_iv, num_t)
        x_test = self.format(np.concatenate((a, x0), axis=1), requires_grad=True)
                        
        self.x_test = (x_test, self.format(t))
        self.y_test = self.format(targets)
        
        self.mse = torch.nn.MSELoss() 
        self.loss_fn = self.mse      
            
        # Initialize optimizer (default adam)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
           self.optimizer,
          lr_lambda=lambda epoch: 0.5 ** (epoch // 10000)
        )


    # def c(self, t):
    #     return self.net(self.x_train[0], t)
  
    # Physics informed loss (using fourier coefficients for computing u_x)
    # Not currently used 
    """def pinn_loss(self):
        
        c = self.net(*self.x_col)
        c = torch.view_as_complex(c.contiguous())
        t = self.x_col[1]
        c_t = torch.zeros_like(c)
        
        for i in range(c.shape[0]):
            for j in range(c.shape[2]):
                    cc = c[i,:,j]
                    c_t[i,:,j] = grad(cc, t, grad_outputs=torch.ones_like(cc), create_graph=True)[0][0,:]
        
    
        
        k = torch.arange(self.N+1)[None, None, :]
        c_x = 1j * k * c
        c_xxx = (-1j) * k**3 * c
        
        
        pde = torch.view_as_real(c_t + c*c_x + c_xxx)
        
        
        
        loss = self.mse(pde, torch.zeros_like(pde))
        
        return loss"""
        
         
    
    # Format data as torch tensor with dtype=float32    
    def format(self, x, requires_grad=False):
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        return x.to(torch.float32).requires_grad_(requires_grad)

    # Generates data for training and testing
    def get_data(self, num_iv, num_t):
        
        x = np.linspace(-self.L/2, self.L/2, 2*self.N+1)[None, None, :]
        t = np.linspace(0, 1, num_t)[None, :, None]
        a = np.random.uniform(0.4, 0.8, size=(num_iv, 1, 1))
        x0 = np.random.uniform(-2, 2, size=(num_iv, 1, 1))
    
        y = self.f(x, a, t, x0)
        
        # Convert to coefficients if Fourier coeffs are predictes
        if self.predict_fourier:
            coeffs = np.fft.rfft(y, axis=-1)        
            # if self.conserve:
            #     analytic_integral = self.I2(a)
            #     truncated_integral = np.asarray(self.integrate(coeffs / (2*self.N+1)))
            #     coeffs = coeffs*np.sqrt(analytic_integral / truncated_integral)            
            y = torch.view_as_real(torch.tensor(coeffs, dtype=torch.cfloat)).to(torch.float32)
        else:
            y = torch.tensor(y, dtype=torch.float32)
        
        return y, a[...,0], x0[...,0], t[0,...]
    
    # Get initial values corresponding to x, a and x0
    # Not currently used as we predict based directly on a and x0
    """def inital_values(self, a, x0):

        f = lambda x, a, x0: 3 * a**2 / np.cosh((a * (((x + x0) + 1) % 2 - 1))/2)**2
        x = np.linspace(-1, 1, 2*self.N+1)[None, :]

        return self.format(np.fft.rfft(f(x, a, x0)))"""

    
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
        
    # Implemented in notebook for more flexibility
    """def plot_conservation(self, a, x0, dpi=100):
                    
        # Analytical integral of u^w
        analytical_integral =  self.I2(a) 
        
        t = torch.linspace(0, 1, 100, requires_grad=True)[:,None]        
        iv =  torch.tensor([[a, x0]], dtype=torch.float32)
        
        #
        u = self.net(iv, t)   
        u = torch.view_as_complex(u.contiguous()) / (2*self.N+1)
        u = u[0,...]
        u = torch.concatenate((u, u[...,1:]), dim=1)  
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
        plt.show()"""    
        
    # Predict output using DeepONet
    def predict(self, a, x0, t):
        
        a, x0, t = a[:,None], x0[:,None], t[:,None]
                
        output = self.net(self.format(np.concatenate((a, x0), axis=1)), self.format(t))
        # coeffs = torch.view_as_complex(output_coeffs)
        
        #output_func = np.fft.ifft(output_coeffs.detach(), n=2*N+1)
        
        return output
            
    