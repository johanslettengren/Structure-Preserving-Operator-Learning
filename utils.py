import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad


# Fully connected neural network
class FNN(torch.nn.Module):
    def __init__(self, layer_sizes, activation='softplus'):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus' : torch.nn.Softplus()}[activation]        
                
        layer_sizes = layer_sizes
        
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
    def __init__(self, layer_sizes, dim, K, activation='tanh'):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh,'softplus' : torch.nn.Softplus()}[activation]
        self.K = K
        self.dim = dim
                
        layer_sizes = [2] + layer_sizes + [dim*K]
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
    def __init__(self, layer_sizes, K, activation='softplus'):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus':torch.nn.Softplus()}[activation]
        
        layer_sizes = [1] + layer_sizes + [2]
        
        # Create layers
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Initialize weights (normal distribution)
            torch.nn.init.xavier_normal_(layer.weight)
            # Initialize biases (zeros)
            #torch.nn.init.zeros_(layer.bias)
            self.linears.append(layer)
        
    # Trunk forward pass
    def forward(self, x_loc):
        z = x_loc
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)        
        
        
        #z = torch.concatenate((z, 1/z), axis=-1)
        

        return z

    
    
# DeepONet class 
class DeepONet(torch.nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, K=2, dim=2, activation='softplus', conserve_energy=False, symplectic=False):
        super(DeepONet, self).__init__()
        
        
        self.symplectic = False if conserve_energy else symplectic
        
        # Final output dimension
        self.dim = dim
        # Initialize branch networks (dim many)
        # self.branches = torch.nn.ModuleList([FNN(layer_sizes_branch, activation) for _ in range(K)]) 
        self.branches = torch.nn.ModuleList([FNN([1] + layer_sizes_branch + [1], activation) for _ in range(2)]) 
        #self.branches = torch.nn.ModuleList([FNN([1] + layer_sizes_branch + [2], activation) for _ in range(2)]) 
        
        self.branch = branch(layer_sizes_branch, dim=dim, K=K, activation=activation) 
        #self.branch = branch(layer_sizes_branch, dim=dim, K=K, activation=activation) 
        # Initialize one trunk network (using split trunk strategy)
        self.trunk = trunk(layer_sizes_trunk, K=K)
        # Initialize bias term
        # Not used!
        # self.bias = torch.nn.Parameter(torch.zeros(1, 1, self.dim))
        # Whether or not to conserve energy (using orthogonalisation of branch and normalisation of trunk)
        self.conserve_energy = conserve_energy

    # DeepONet forward pass
    def forward(self, x_func, x_loc):
        
        
        # Get one output for each branch (dim many)
        # Shape (#u, dim, K): K = branch & trunk output dim
        # #u is the same as the number of boundary values
        #branch_outputs = torch.stack([self.branches[i](x_func[:,-i,None]) for i in range(len(self.branches))], axis=-1)
        
        # Get trunk ouput (only one)  
        trunk_outputs  = self.trunk(x_loc)
        
        #print(trunk_outputs.shape)
        
        if self.symplectic: 
            
            bs = [self.branches[i](x_func[:,i,None]) for i in range(len(self.branches))]            
            b1 = torch.stack(bs, axis=1)[...,0]
            mask = torch.tensor([-1, 1])[None, :] 
            b2 = mask * b1.flip(dims=[-1])
            branch_outputs = torch.stack((b1, b2), axis=-1)      
            trunk_outputs = torch.nn.functional.normalize(trunk_outputs, dim=-1)
            
            #branch_outputs = torch.stack([self.branches[i](x_func[:,i,None]) for i in range(len(self.branches))], axis=-1)
                  
    
        
        else:
            branch_outputs = self.branch(x_func)            
        
        if self.conserve_energy:
            # orthogonalise branch outputs
            Q, R = torch.linalg.qr(branch_outputs)
            # Rescale (so that norm matches energy) and redefine branch outputs
            branch_outputs = Q*torch.linalg.norm(x_func, dim=1, keepdim=True).unsqueeze(-1)
            # Find the corresponding coordinates
            trunk_outputs = torch.einsum("ukK,tK->utK", R, trunk_outputs)
            # Normalise and redfine trunk outputs
            trunk_outputs = torch.nn.functional.normalize(trunk_outputs, dim=2)
            # Get corresponding network output (trunk_outputs has one more dimension than usual)
            output = torch.einsum("udK,utK->utd", branch_outputs, trunk_outputs)
        else:
            # Get network output when energy is not conserved
            output = torch.einsum("udK,tK->utd", branch_outputs, trunk_outputs)
        
              
        
        """ If needed in the future
        # trunk_outputs = trunk_outputs.unsqueeze(0).expand(self.dim, *trunk_outputs.shape)        
        # equivalent to torch.stack([self.trunk.activation(self.trunk(x_loc)) for _ in range(self.dim)], dim=0)"""
        
        
        return output


# Compile and train model + data
class Model():
    
    def __init__(
        self, x_train, y_train, x_test, y_test, x_col, net, 
        optimizer='adam', lr=0.001, Px=None, PI_loss_fn=None, Tmax=5):     
        
        # Training data
        self.x_train = (self.format(x_train[0]), self.format(x_train[1]))
        self.y_train = self.format(y_train)
        
        # Testing data
        self.x_test = (self.format(x_test[0]), self.format(x_test[1]))
        self.y_test = self.format(y_test)
        

        self.x_col = (self.format(x_col[0]), self.format(x_col[1]))
        

        
        
        #self.Px_func = self.format(Px[0], requires_grad=True) if Px is not None else None
        #self.Px_loc = self.format(Px[1], requires_grad=True) if Px is not None else None
        
        # Network
        self.net = net
        
        # Loss history
        self.tlosshistory = []  # training loss
        self.vlosshistory = []  # validation/test loss

        # Initialize optimizer (default adam)
        optimizer = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}[optimizer]
        self.optimizer = optimizer(net.parameters(), lr=lr)
        
        # Set loss function (MSE default)
        self.mse = torch.nn.MSELoss()
        self.PI_loss_fn = PI_loss_fn
        
        self.loss_fn = self.MSE_loss if PI_loss_fn is None else self.PINN_loss
        
        self.Tmax = Tmax
        
        
    # Format data as torch tensor with dtype=float32    
    def format(self, x, requires_grad=False):
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        return x.to(torch.float32).requires_grad_(requires_grad)
    
    def MSE_loss(self, prediction, targets):
        return self.mse(prediction, targets)
    
    def PINN_loss(self, prediction, targets):
        
        torch.set_grad_enabled(True)
        # Calculate physics informed loss
        u = self.net(self.Px_func, self.Px_loc)
        Ploss = self.PI_loss_fn(u, self.Px_loc)
        Ploss = self.mse(Ploss, torch.zeros_like(Ploss))
        return self.mse(prediction, targets) + Ploss
        
        
    # Train network
    def train(self, iterations):
        
        # Train step history
        self.steps = []
        
        # Set net to train mode
        self.net.train(True)
        
        # Training process updates
        print('Step \t Train loss \t Test loss')

        for iter in range(iterations):
            
                        
            # Physics informed loss
            #u = self.net(self.Px_func, self.Px_loc)
            #Ploss = self.PI_loss_fn(u, self.Px_loc)
            #Ploss = self.loss_fn(Ploss, torch.zeros_like(Ploss))
            #loss = self.loss_fn(prediction, self.y_train) + Ploss
            
            # Train
            self.optimizer.zero_grad()
            prediction = self.net(*self.x_train)
            loss = self.loss_fn(prediction, self.y_train)

            if self.net.symplectic:
                prediction_0 = self.net(*self.x_col)
                loss_0 = self.loss_fn(prediction_0, self.x_col[0][:, None, :])
                loss += loss_0
            
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
                    
                    if self.net.symplectic:
                        prediction_0 = self.net(*self.x_col)
                        vloss_0 = self.loss_fn(prediction_0, self.x_col[0][:, None, :])
                        vloss += vloss_0
                    
                    
    
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
        
    def plot_energy(self, q0, p0, dpi=100):
        
        # Energy test boundary values
        energy_bv =  torch.tensor([[q0, p0]], dtype=torch.float32).requires_grad_(True)
        
        # Energy test time-domain
        energy_t = torch.linspace(0, self.Tmax, 100, requires_grad=True).reshape(-1, 1).requires_grad_(True)
        
        # Calculate total energy
        u = self.net(energy_bv, energy_t)   
        x, y = u[..., 0], u[..., 1]        
        E = (x**2 + y**2)/2
                
        _, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        t = energy_t.detach()
        
        ax.set_title(f'Total Energy of DeepONet Prediction $(q_0, p_0)$=({q0}, {p0})')
        ax.plot(t, torch.ones_like(t)*(q0**2 + p0**2)/2, alpha=0.5, linewidth=5, label='True energy')
        ax.plot(t, E[0,...].detach(), '--', alpha=0.8, linewidth=3,  label='DeepONet energy')
        ax.legend()
        ax.grid(True)
        
        ax.set_xlabel("t")
        ax.set_ylabel("Total energy")
        plt.show()
        
    def test_symplecticity(self, T, num_pts=100):

        J = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)
        #norm_J = torch.linalg.norm(J)
        
        grid_points = torch.linspace(-1, 1, num_pts, requires_grad=True, dtype=torch.float32)
        mesh = torch.cartesian_prod(grid_points, grid_points)
        #print(mesh)
        
        t = torch.tensor([T], requires_grad=True, dtype=torch.float32).reshape(-1, 1)
        
        delta_J = []
        
        #print(torch.autograd.functional.jacobian(self.net, (mesh, t)))
        
        for idx in range(num_pts**2):
            bv = mesh[idx].unsqueeze(0)
            N = self.net(bv, t)
            
            
            
            N1, N2 = N[0,0,0],  N[0,0,1]                
            row1 = grad(N1, bv, retain_graph=True)[0].squeeze()
            row2 = grad(N2, bv, retain_graph=True)[0].squeeze()
            Jac = torch.stack((row1, row2))
            #det = row1[0]*row2[1] - row1[1]*row2[0]
            #print(Jac)
            delta_J.append(np.linalg.norm(Jac.T @ J @ Jac -J))
            
                
        delta_J = np.array(delta_J).reshape(num_pts, num_pts)
        plt.figure(figsize=(8, 6))
        plt.imshow(delta_J, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        plt.colorbar(label='$\Delta J$')
        plt.title("$\Delta J = || \\nabla \mathcal{N}(t)^T J \\nabla \mathcal{N}(t) - J||/||J||$ over $(q,p)$-grid, for t=" + f"{T}")
        plt.xlabel('q')
        plt.ylabel('p')
        plt.show()
        
    
    # Predict output using DeepONet
    def predict(self, x_func, x_loc):
        
        return self.net(self.format(x_func, requires_grad=True), self.format(x_loc, requires_grad=True))
                
        