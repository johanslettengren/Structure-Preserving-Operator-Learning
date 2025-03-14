import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

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
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, K=2, N=1, activation='tanh', scheme=None, L=3, M=3):
        
        super(DeepONet, self).__init__()
        
        
        # Whether to QR factorise the branch
        self.scheme = scheme
        
        self.N = N
        
        I = torch.eye(N)
        O = torch.zeros_like(I)

        self.J = torch.cat([torch.cat([O, I], dim=1), torch.cat([-I, O], dim=1) ], dim=0)

        activation = {'relu' : nn.ReLU(), 'tanh':torch.nn.Tanh(), 'softplus' : torch.nn.Softplus(), 'htanh' : torch.nn.Hardtanh()}[activation]   
            
        if scheme in ['preorth', 2]:
            self.branch = branch(layer_sizes_branch, dim=2*N, K=1, activation=activation)
            self.trunk = trunk(layer_sizes_trunk, K=K, activation=activation)        

        elif scheme in ['symplectic', 'symp', 3]:
            self.L = L
            self.M = M
            branch_net = branch([10, 10, 10], K=1, dim=1, activation=activation) 
            #gradbranch_net = gradientbranch(layer_sizes_branch, dim=N, activation=activation)
            self.branch = torch.nn.ModuleList([torch.nn.ModuleList([branch_net for _ in range(N)]) for _ in range(L)])
            #self.gradbranch = torch.nn.ModuleList([gradbranch_net for _ in range(L)])
            #self.trunk = torch.nn.ModuleList([trunk(layer_sizes_trunk, K=1, activation=activation) for _ in range(L)])    
            self.a = torch.nn.ParameterList([np.sqrt(1e-3) * torch.nn.Parameter(torch.randn(2,N)) for _ in range(L)])
            self.v = torch.nn.ParameterList([np.sqrt(1e-3) * torch.nn.Parameter(torch.randn(2)) for _ in range(L)])
            self.trunk = torch.nn.Parameter(torch.randn(2))
            

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
            
            
            momenta = momenta[:,None,...]
        
            for i in range(self.L):

                a = self.a[i]
                trunk = self.trunk[0](t)[0,0]
                N = momenta.shape[-1]//2
                
                q = momenta[...,:N]
                p = momenta[...,N:]                

                arg = a[0,None,None,:] * q + a[1,None,None,:] * p                        
                psi = torch.concatenate([self.branch[i][j](arg[...,None,j]).squeeze(-1) for j in range(N)], dim=-1)
                

                branch = torch.concatenate((a[1,None,None,:]*psi, -a[0,None,None,:]*psi), axis=-1)
                momenta = momenta + self.trunk[0] * branch
                
            #for i in range(self.M):

                # v = self.v[i]
                # N = momenta.shape[-1]//2
                
                # q = momenta[...,:N]
                # p = momenta[...,N:]                

                # arg = v[0] * q + v[1] * p                        
                # psi = self.gradbranch[i](arg).squeeze(-1)
                
                
                # branch = torch.concatenate((v[1]*psi, -v[0]*psi), axis=-1)
                # momenta = momenta + self.trunk[0] * branch

            output = momenta
                                            
        else:
            trunk_outputs  = self.trunk(t)
            trunk_outputs = torch.ones_like(trunk_outputs) #torch.nn.functional.normalize(trunk_outputs, dim=-1)
            branch_outputs = self.branch(momenta)        
            if len(branch_outputs.shape) == 4:    
                output = torch.einsum("uZdK,tK->utZd", branch_outputs, trunk_outputs)
            else: 
                output = torch.einsum("udK,tK->utd", branch_outputs, trunk_outputs)
                
        return output



# Compile and train model + data
class DeepONetModel():
    
    def __init__(self, x_train, y_train, x_test, y_test, net, lr=0.001):  
                
        N = x_train[0].shape[-1]
        # Training data
        self.x_train = (self.format(x_train[0], requires_grad=True), self.format(x_train[1], requires_grad=True))
        self.y_train = (self.format(y_train[...,:N]), self.format(y_train[...,N:]))
        
        # Testing data
        self.x_test = (self.format(x_test[0]), self.format(x_test[1]))
        self.y_test = (self.format(y_test[...,:N]), self.format(y_test[...,N:]))
                
        self.bestvloss = 1000000
        self.bestloss = 1000000
        
        self.q_scale = torch.mean(torch.abs(self.y_train[0]-self.x_train[0])**2)
        self.p_scale = torch.mean(torch.abs(self.y_train[1]-self.x_train[1])**2)
        
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
                with torch.no_grad():
                    
                    outputs = self.net(*self.x_test)  
                    vloss = self.loss_fn(outputs, self.test)
                    
                    announce_new_best = ''
                    if vloss < self.bestvloss:
                        announce_new_best = 'New best model!'
                        self.bestvloss = vloss
                        torch.save(self.net.state_dict(), "best_model.pth")  # Save model weights                    
                        
        
                    # Save loss history
                    self.vlosshistory.append(vloss)
                    self.tlosshistory.append(tloss)
                    self.steps.append(iter)
                    self.net.train(True)
                
                print('{} \t [{:.2e}] \t [{:.2e}] \t {}'.format(iter + 1, tloss, vloss, announce_new_best))    
                
        # print('Best training loss:', self.bestloss.item())
        self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        self.net.eval()
        
    
    # Predict output using DeepONet
    def predict(self, momenta, t):
        
        momenta = self.format(momenta, requires_grad=True)
        t = self.format(t, requires_grad=True)
        
        u = self.net(momenta, t)

        
        return u