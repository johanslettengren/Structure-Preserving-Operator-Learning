import torch
import torch.nn as nn

class SNN(nn.Module):
    """Symplectic Neural Network

    Args:
        layer_sizes : width of each symplectic layer
        N : dimension of prediction
        s : scale of parameter 'a'
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, N, s, activation):
        super().__init__()
        
        # possible activation functions
        # NN fits an activation using a neural network with ReLU activation
        dict = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs,
            'NN' : FNN([1, 5, 5, 5, 1], 'relu')}  
        
        self.activation = dict[activation]  
        
        # Linear layers
        self.linears = nn.ModuleList() 
        
        # Other parameters
        self.bias, self.a, self.D = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
                
        # Create parameters
        for i in range(len(layer_sizes)):
            
            layer = nn.Linear(N, layer_sizes[i], bias=True)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
            self.linears.append(layer)
            self.bias.append(nn.Parameter(torch.zeros(N)))
            self.D.append(nn.Parameter(torch.randn(layer_sizes[i], 1)))
            
            # Initialize at appropriate scale
            self.a.append(s * nn.Parameter(torch.randn(2)))
            
    def forward(self, p, q):
        """Forward pass"""
        for l, linear in enumerate(self.linears):            
            
            # Get parameters a and b
            a, b = self.a[l][0], self.a[l][1]  
            
            # Linear combination of p and q
            z = a * p + b * q
                    
            # Apply linear layer and activation
            # .unsqueeze needed if activation is NN
            z = self.activation(linear(z).unsqueeze(-1)).squeeze(-1)
            
            # Apply transpose wieght matrix, diagonal matrix and second bias
            z = torch.matmul(z, self.D[l] * linear.weight) + self.bias[l]
            # Update p and q accordingly
            p = p + b * z
            q = q - a * z

        return torch.cat((p, q), axis=-1)
    
    
class SNN1D(nn.Module):
    """Symplectic Neural Network in 1D case
    
    Has slightly simpler architecture than the general SNN.

    Args:
        num_layers : number of symplectic layers
        NN_layer_sizes : shape of the 'activation' NN
        h : prediction time-step
        activation : activation function for NN
    """
    def __init__(self, num_layers, NN_layer_size, h, activation):
        super().__init__() 
 
        self.h = h
        self.a = nn.ParameterList()    
        self.layers = nn.ModuleList()
                
        # Create parameters
        for _ in range(num_layers):
            
            layer = FNN([1] + NN_layer_size + [1], activation)
            self.layers.append(layer)
            
            # Initialize at appropriate scale 
            self.a.append(h * nn.Parameter(torch.randn(2)))
        
    def forward(self, p, q):
        """Forward pass"""
        for l, layer in enumerate(self.layers):            
            
            # Get paramaters
            a, b = self.a[l][0], self.a[l][1]       
            
            # Take linear combination
            z = a * p + b * q
        
            # Apply NN
            z = layer(z)
                        
            # Update accordingly
            p = p + b * z
            q = q - a * z

        return torch.cat((p, q), axis=-1)

class vRO(nn.Module):
    """Rollout Net Built on Vanilla NN

    Args:
        layer_size : shape of NN
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, activation):
        
        super().__init__()
        
        self.NN = FNN(layer_sizes, activation)
            
    def forward(self, p, q):
        """Forward pass"""
        z = torch.cat((p, q), axis=-1)
        
        return self.NN(z)
    
class FNN(nn.Module):
    """Fully Connected Neural Network

    Args:
        layer_sizes : network shape
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, activation):
        
        super().__init__()
        
        dict = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs}   
        
        self.activation = dict[activation] 
                                
        layer_sizes = layer_sizes
        
        # Create layers
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
    def forward(self, z):
        """Forward pass"""
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)
        return z
    
def predict_rollout(net, X, Tmax, h):
        """Roll out prediction"""
        
        N = X[0].shape[-1]
        
        # Number of updates
        iters = int(Tmax / h)
        
        X = (X[0][:,None,:], X[1][:,None,:])
        
        p, q = X
        
        
        for _ in range(iters):
            
            out = net(*X)
            X = (out[...,:N], out[...,N:])
            p = torch.cat((p, X[0]), axis=1)
            q = torch.cat((q, X[1]), axis=1)
        
        return torch.cat((p, q), axis=-1)