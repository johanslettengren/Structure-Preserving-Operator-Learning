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
        
        
    # Branch forward pass
    def forward(self, momenta):

        z = momenta
        
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)
        return z

class branch(torch.nn.Module):
    def __init__(self, layer_sizes, K, N, activation, dim=2):
        super().__init__()
        
        # Set activation function
        self.activation = activation
        self.dim = dim
        self.N = N
        self.K = K
                
        layer_sizes = [dim*N] + layer_sizes + [dim*K*N]
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
                
        z = momenta.reshape(*momenta.shape[:-2], -1)
        
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
       
        z = self.linears[-1](z)
        z = z.view(*z.shape[:-1], self.N, self.dim, self.K)
        
             

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
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, K=2, activation='tanh', scheme=None, L=3, N=1):
        super(DeepONet, self).__init__()
        
        # Whether to QR factorise the branch
        self.scheme = scheme
        
        I = torch.eye(N)
        O = torch.zeros_like(I)

        self.J = torch.cat([torch.cat([O, I], dim=1), torch.cat([-I, O], dim=1) ], dim=0)
        
        #self.J = torch.tensor([[O, I], [-I, 0]], dtype=torch.float32)

        activation = {'relu' : torch.relu, 'tanh':torch.tanh, 'softplus' : torch.nn.Softplus(), 'htanh' : torch.nn.Hardtanh()}[activation]   
        
            
        if scheme in ['preorth', 2]:
            self.branch = branch(layer_sizes_branch, K=1, N=N, activation=activation) 
            self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)        

        elif scheme in ['symplectic', 'symp', 3]:
            self.branch = torch.nn.ModuleList([branch(layer_sizes_branch, 1, N, activation, dim=1) for _ in range(L)]) 
            self.trunk = trunk(layer_sizes_trunk, K=1, activation=activation)     
            self.a = torch.nn.ParameterList([0.01*torch.nn.Parameter(torch.randn(2, )) for _ in range(L)])
            self.L = L

        else:  
            self.branch = branch(layer_sizes_branch, K=K, N=N, activation=activation) 
            self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)        
        
    


    # DeepONet forward pass
    def forward(self, momenta, t):
        
        
        
        if self.scheme in ['QR', 1]:
            trunk_outputs  = self.trunk(t)
            branch_outputs = self.branch(momenta)  
            
            # orthogonalise branch outputs
            Q, R = torch.linalg.qr(branch_outputs)
            
            # Rescale (so that norm matches energy) and redefine branch outputs
            branch_outputs = Q * torch.linalg.norm(momenta, dim=-1)[:,None,None]
            # Find the corresponding coordinates
            

            
            trunk_outputs = torch.einsum("uNkK,tK->utNK", R, trunk_outputs)
            
            # Normalise and redfine trunk outputs
            trunk_outputs = torch.nn.functional.normalize(trunk_outputs, dim=-1)
            # Get corresponding network output (trunk_outputs has one more dimension than usual)
    
            output = torch.einsum("uNdK,utNK->utNd", branch_outputs, trunk_outputs)
            
        elif self.scheme in ['preorth', 2]:
            trunk_outputs  = self.trunk(t)
            branch_outputs = self.branch(momenta)       
            b1 = self.branch(momenta).squeeze(-1)
            print(b1.shape)
            print(self.J.shape)
            b2 = torch.einsum ('uNd, dD -> uND', b1, self.J)          
            print(b2.shape)  
            branch_outputs = torch.stack((b1, b2), axis=-1)   
            
            trunk_outputs = torch.nn.functional.normalize(trunk_outputs, dim=-1)
            output = torch.einsum("uNdK,tK->utNd", branch_outputs, trunk_outputs)   
            
        elif self.scheme in ['symplectic', 'symp', 3]:
            
            momenta = momenta[:,None,...]
            
            
            for i in range(self.L):
                a = self.a[i]
                alpha = self.trunk(t)[None,:,:,None]   
                psi = self.branch[i](torch.einsum('utNd, d -> utN', momenta, a)[...,None]).squeeze(-1)                
                momenta = momenta + (alpha * (self.J @ a) * psi)
                
                
                
            output = momenta
                                            
        else:
            trunk_outputs  = self.trunk(t)
            branch_outputs = self.branch(momenta)
            output = torch.einsum("uNdK,tK->utNd", branch_outputs, trunk_outputs)     
    
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
            pred = self.net(*self.x_train)   
            loss = self.loss_fn(pred, self.y_train)


            loss.backward()
            self.optimizer.step()
            tloss = loss.item()

            # Test
            if iter % (iterations // 10) == iterations / 10 - 1:
                # Set net to evalutation mode
                self.net.eval()
                
                # Don't calculate gradients
                with torch.no_grad():
                    
                    
                    pred = self.net(*self.x_test)  
                    vloss = self.loss_fn(pred, self.y_test).item()
                    
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