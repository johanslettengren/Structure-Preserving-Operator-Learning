import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# Fully connected neural network
class FNN(torch.nn.Module):
    def __init__(self, layer_sizes, activation):
        super().__init__()
        
        # Set activation function
        self.activation = activation 
                
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

# Fully connected neural network
class gradientbranch(torch.nn.Module):
    def __init__(self, layer_sizes, dim, activation):
        super().__init__()
        
        # Set activation function
        self.activation = activation
        self.dim = dim
                
        layer_sizes = [dim] + layer_sizes + [1]
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
    def forward(self, momenta):
        
        z = momenta
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)
        
        f = grad(z, momenta, grad_outputs=torch.ones_like(z), retain_graph=True)[0]
                
        return f

class branch(torch.nn.Module):
    def __init__(self, layer_sizes, dim, K, activation):
        super().__init__()
        
        # Set activation function
        self.activation = activation
        self.dim = dim
        self.K = K
                
        layer_sizes = [dim] + layer_sizes + [dim*K]
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
    def forward(self, momenta):
        

        
        z = momenta
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z).view(*z.shape[:-1], self.dim, self.K)
             

        return z
        
class trunk(torch.nn.Module):
    def __init__(self, layer_sizes, K, activation):
        super().__init__()
        
        self.activation = activation
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
    def forward(self, t):
        z = t
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)                

        return z

    
    
# DeepONet class 
class DeepONet(torch.nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, K=2, N=1, activation='tanh', scheme=None, L=3):
        super(DeepONet, self).__init__()
        
        # Whether to QR factorise the branch
        self.scheme = scheme
        
        self.N = N
        
        I = torch.eye(N)
        O = torch.zeros_like(I)

        self.J = torch.cat([torch.cat([O, I], dim=1), torch.cat([-I, O], dim=1) ], dim=0)
        #self.J = torch.zeros_like(self.J)

        activation = {'relu' : torch.relu, 'tanh':torch.nn.Tanh(), 'softplus' : torch.nn.Softplus(), 'htanh' : torch.nn.Hardtanh()}[activation]   
            
        if scheme in ['preorth', 2]:
            self.branch = branch(layer_sizes_branch, dim=2*N, K=1, activation=activation)
            self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)        

        elif scheme in ['symplectic', 'symp', 3]:
            self.L = L
            branch_net = branch(layer_sizes_branch, K=1, dim=1, activation=activation) 
            self.branch = torch.nn.ModuleList([torch.nn.ModuleList([branch_net for _ in range(N)]) for _ in range(L)])
            self.trunk = torch.nn.ModuleList([trunk(layer_sizes_trunk, K=1, activation=activation) for _ in range(L)])    
            self.a = torch.nn.ParameterList([0.01 * torch.nn.Parameter(torch.randn(2,N)) for _ in range(L)])
            

        else:  
            self.branch = branch(layer_sizes_branch, dim=2*N, K=K, activation=activation) 
            self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)        
        
    


    # DeepONet forward pass
    def forward(self, momenta, t):
        
        
        
        
        if self.scheme in ['QR', 1]:
            trunk_outputs  = self.trunk(t)
            branch_outputs = self.branch(momenta)       
            # orthogonalise branch outputs
            Q, R = torch.linalg.qr(branch_outputs)            
            # Rescale (so that norm matches energy) and redefine branch outputs
            branch_outputs = Q*torch.linalg.norm(momenta, dim=-1)[:,None,None]
            # Find the corresponding coordinates
            trunk_outputs = torch.einsum("ukK,tK->utK", R, trunk_outputs)
            # Normalise and redfine trunk outputs
            trunk_outputs = torch.nn.functional.normalize(trunk_outputs, dim=2)
            # Get corresponding network output (trunk_outputs has one more dimension than usual)
            output = torch.einsum("udK,utK->utd", branch_outputs, trunk_outputs)
            
        elif self.scheme in ['preorth', 2]:
            trunk_outputs  = self.trunk(t)
            branch_outputs = self.branch(momenta)       
            b1 = self.branch(momenta).squeeze(-1)
            #J = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)

            b2 = torch.einsum ('ud, dD -> uD', b1, self.J)
            branch_outputs = torch.stack((b1, b2), axis=-1)    
            
            trunk_outputs = torch.nn.functional.normalize(trunk_outputs, dim=-1)
            output = torch.einsum("udK,tK->utd", branch_outputs, trunk_outputs)   
            
        elif self.scheme in ['symplectic', 'symp', 3]:
            
            
            momenta = momenta[:,None,:]
            
            
        
            for i in range(self.L):
                
                
                
                a = self.a[i]
                alpha = self.trunk[0](t)[None,...]
                
                
                N = momenta.shape[-1]//2
                
                p = momenta[...,:N]
                q = momenta[...,N:]                

                arg = a[0,None,None,:] * p + a[1,None,None,:] * q                        
                
                
                psi = torch.concatenate([self.branch[i][j](arg[...,None,j]).squeeze(-1) for j in range(N)], dim=-1)

                branch = torch.concatenate((a[1,None,None,:]*psi, -a[0,None,None,:]*psi), axis=-1)
                momenta = momenta + alpha * branch
                
            output = momenta
                                            
        else:
            trunk_outputs  = self.trunk(t)
            trunk_outputs = torch.ones_like(trunk_outputs) #torch.nn.functional.normalize(trunk_outputs, dim=-1)
            branch_outputs = self.branch(momenta)               
            output = torch.einsum("udK,tK->utd", branch_outputs, trunk_outputs)     
                
        return output



# Compile and train model + data
class Model():
    
    def __init__(
        self, x_train, y_train, x_test, y_test, net, lr=0.001):     
        
        # Training data
        self.x_train = (self.format(x_train[0], requires_grad=True), self.format(x_train[1], requires_grad=True))
        self.y_train = self.format(y_train)
        
        # Testing data
        self.x_test = (self.format(x_test[0], requires_grad=True), self.format(x_test[1], requires_grad=True))
        self.y_test = self.format(y_test)
                
        self.bestloss = 1000000
        
        # Network
        self.net = net
        
        # Loss history
        self.tlosshistory = []  # training loss
        self.vlosshistory = []  # validation/test loss

        # Initialize optimizer (default adam)
        optimizer = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}['adam']
        self.optimizer = optimizer(net.parameters(), lr=lr)


        
        # Set loss function (MSE default)
        self.mse = torch.nn.MSELoss()
        self.loss_fn = self.mse 
        
        
    # Format data as torch tensor with dtype=float32    
    def format(self, x, requires_grad=False):
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        return x.to(torch.float32).requires_grad_(requires_grad)
        
        
    # Train network
    def train(self, iterations):
        
        # Train step history
        self.steps = []
        self.net.train(True)
        
        print('Step \t Train loss \t Test loss')
        
        for iter in range(iterations):
            
            # Train
            self.optimizer.zero_grad()
            outputs = self.net(*self.x_train)   
            
            loss = self.loss_fn(outputs, self.y_train)

            loss.backward()
            self.optimizer.step()
            tloss = loss.item()

            # Test
            if iter % (iterations // 10) == iterations / 10 - 1:
                # Set net to evalutation mode
                self.net.eval()
                
                # Don't calculate gradients
                # with torch.no_grad():
                    
                    
                outputs = self.net(*self.x_test)  
                vloss = self.loss_fn(outputs, self.y_test).item()
                
                announce_new_best = ''
                if vloss < self.bestloss:
                    announce_new_best = 'New best model!'
                    self.bestloss = vloss
                    torch.save(self.net.state_dict(), "best_model.pth")  # Save model weights                    
                    
    
                # Save loss history
                self.vlosshistory.append(vloss)
                self.tlosshistory.append(tloss)
                self.steps.append(iter)
                self.net.train(True)
                
                print('{} \t [{:.2e}] \t [{:.2e}] \t {}'.format(iter + 1, tloss, vloss, announce_new_best))    
                
        self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        self.net.eval()
        
    
    # Predict output using DeepONet
    def predict(self, momenta, t):
        
        momenta = self.format(momenta, requires_grad=True)
        t = self.format(t, requires_grad=True)
        
        u = self.net(momenta, t)

        
        return u
    
    
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
        
    def plot_energy(self, q=1, p=1, Tmax=1, dpi=100):
 
        bv = np.array([[q, p]], dtype=np.float32)
        t = np.linspace(0, Tmax, 20).reshape(-1, 1)
        u = self.predict(bv, t)
    
         
        E = torch.linalg.norm(u, axis=-1)[0,:]**2/2
        _, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.set_title(f'Total Energy of DeepONet Prediction $(q_0, p_0)$=({q}, {p})')
        ax.plot(t, np.ones_like(t)*(q**2 + p**2)/2, alpha=0.5, linewidth=5, label='True energy')
        ax.plot(t, E.detach(), '--', alpha=0.8, linewidth=3,  label='DeepONet energy')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("t")
        ax.set_ylabel("Total energy")
        plt.show()
        
    def plot_symplecticity(self, T=1, num_pts=100):

                
            grid_points = torch.linspace(-1, 1, num_pts, requires_grad=True, dtype=torch.float32)
            mesh = torch.cartesian_prod(grid_points, grid_points)
            t = torch.tensor([T], requires_grad=True, dtype=torch.float32).reshape(-1, 1)
            u = self.net(mesh, t)[:,0,:]


            D = torch.stack([grad(u[:,i], mesh, grad_outputs=torch.ones_like(u[:,i]), retain_graph=True)[0] for i in range(u.shape[1])], axis=-1)
            J = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)

            DTJD = torch.matmul(D.transpose(1, 2), torch.matmul(J, D))
            deltaJ = torch.linalg.norm(DTJD - J, ord=2, dim=(1, 2)).numpy().reshape(num_pts, num_pts)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(deltaJ, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
            plt.colorbar(label='$\Delta J$')
            plt.title("$\Delta J$ over $(q,p)$-grid, for t=" + f"{T}")
            plt.xlabel('q')
            plt.ylabel('p')
            plt.show()
            
    def plot_full_predictions(self, solver, Tmax, num_iv=3):
        
        iv = np.random.uniform(low=-1, high=1, size=(num_iv, 2)).astype(np.float32)
        t = np.linspace(0, Tmax, 100)
        pred  = self.predict(iv, t.reshape(-1, 1))
        true = solver(iv, t.reshape(-1, 1))
        
        self.plot_predictions(pred, true, t, iv)
        
    def plot_rollout_predictions(self, solver, h=0.01, iters=1000, num_iv=3):


        T = h * iters
        h = torch.tensor([[h]], dtype=torch.float32)
        
        iv = np.random.uniform(low=-1, high=1, size=(num_iv, 2)).astype(np.float32)
        conj_momenta = self.predict(self.format(iv), h)
        
        pred = conj_momenta.clone()
        
        for _ in range(iters):
            conj_momenta = self.predict(conj_momenta.squeeze(1), h)
            pred = torch.concatenate((pred, conj_momenta), axis=1)
            
        t = np.linspace(0, T, iters+1)
        true = solver(iv, t.reshape(-1, 1))
                        
        self.plot_predictions(pred, true, t, iv, title='Rollout prediction')

    
    def plot_predictions(self, pred, true, t, iv, title='Prediction'):


        # Plot prediction vs. solution
        fig, axes = plt.subplots(2, 2, figsize=(20, 4), dpi=150)

        axes[0][0].set_title(title)
        axes[0][1].set_title('Error')
        
        handles, labels = [], []
        
        for k in range(axes.shape[0]):
            for l in range(pred.shape[0]):
                
                label = '({:.1f}, {:.1f})'.format(iv[l, 0], iv[l, 1])
                axes[k][0].plot(t, true[l, :, k], label=label, alpha=0.5, linewidth=5)
                axes[k][0].plot(t, pred[l, :, k].detach(), '--', c='k')
                line, = axes[k][1].plot(t, true[l, :, k] - pred[l, :, k].detach(), label=label, linewidth=2)

                # Store handles and labels only once
                if k == 0:
                    handles.append(line)
                    labels.append(label)

            axes[k][0].grid(True)
            axes[k][1].grid(True)

        axes[k][0].set_xlabel("t")
        axes[k][1].set_xlabel("t")

        # Create a shared legend
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10)

        plt.show()
        
        
        
                
        