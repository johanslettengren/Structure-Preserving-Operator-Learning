import torch
import torch.nn as nn

class Branch(nn.Module):
    """Branch Network for DeepONet

    Args:
        layer_sizes : shape of network
        dim : dimension of input
        K : number of branch vectors
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, dim, K, activation):
        super().__init__()
        
        self.activation = activation
        self.dim = dim
        self.K = K
                
        # We will use one network to predict all branch vectors simultaneuosly
        layer_sizes = [dim] + layer_sizes + [dim*K]
        
        # Create layers
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
        
    def forward(self, z):
        """Branch forward pass"""

        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        # Reshape to [batch_size, dim, K] 
        # K branch vectors of same dimension as input
        z = self.linears[-1](z).view(*z.shape[:-1], self.dim, self.K) 

        return z
    
class ComplexBranch(nn.Module):
    """Branch Network for DeepONet

    Args:
        layer_sizes : shape of network
        dim : dimension of input
        K : number of branch vectors
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, dim, K, activation):
        super().__init__()
        
        self.activation = activation
        self.dim = dim
        self.K = K
                
        # We will use one network to predict all branch vectors simultaneuosly
        layer_sizes = [dim] + layer_sizes + [dim*K]
        
        # Create layers
        self.real_linears = nn.ModuleList()
        self.imag_linears = nn.ModuleList()
        
        self.real_biases = nn.ParameterList()
        self.imag_biases = nn.ParameterList()
        
        for i in range(1, len(layer_sizes)):
            real_layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i], bias=False)
            imag_layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i], bias=False)
            
            nn.init.xavier_normal_(real_layer.weight)
            nn.init.xavier_normal_(imag_layer.weight)
            
            self.real_linears.append(real_layer)
            self.imag_linears.append(imag_layer)
            
            self.real_biases.append(nn.Parameter(torch.zeros(layer_sizes[i])))
            self.imag_biases.append(nn.Parameter(torch.zeros(layer_sizes[i])))
        
    def forward(self, z):
        """Branch forward pass"""
        
        z_i, z_r = z[...,:self.dim], z[...,self.dim:]

        for l in range(len(self.imag_linears[:-1])):
            
            z_i_ = self.activation(self.real_linears[l](z_i) + self.imag_linears[l](z_r) + self.imag_biases[l])
            z_r  = self.activation(self.real_linears[l](z_r) - self.imag_linears[l](z_i) + self.real_biases[l])
            z_i  = z_i_ 
            

        z_i_ = self.real_linears[-1](z_i) + self.imag_linears[-1](z_r) + self.imag_biases[-1]
        z_r  = self.real_linears[-1](z_r) - self.imag_linears[-1](z_i) + self.real_biases[-1]
        z_i  = z_i_
        
        # Reshape to [batch_size, dim, K] 
        # K branch vectors of same dimension as input
        z_i = z_i.view(*z.shape[:-1], self.dim, self.K) 
        z_r = z_r.view(*z.shape[:-1], self.dim, self.K) 

        return z_r + 1j * z_i
        
class Trunk(nn.Module):
    """Trunk Network for DeepONet

    Args:
        layer_sizes : shape of network
        K : number of trunk coefficients (same as number of branch vectors)
        activation : nonlinear activation function
    """
    def __init__(self, layer_sizes, K, activation):
        super().__init__()
        
        self.activation = activation
        layer_sizes = [1] + layer_sizes + [K]
        
        # Create layers
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Initialize weights (normal distribution)
            nn.init.xavier_normal_(layer.weight)
            # Initialize biases (zeros)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)

    def forward(self, z):
        """Trunk forward pass"""
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)                

        return z

    
    
class DeepONet(nn.Module):
    """Deep Operator Neural Network

    Args:
        layer_sizes_branch : shape of branch net
        layer_sizes_trunk : shape of trunk net
        K : number of branch vectors
        dim : dimension of input / output
        activation : nonlinear activation function for both branch and trunk
        scheme : scheme used for predicting outputs (e.g. vanilla or conservative)
    """
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, K, dim, activation, scheme=None):
        
        super().__init__()
        
        self.scheme = scheme 
        
        activation = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs}[activation]   

        # Create branch- and trunk-net
        self.branch = Branch(layer_sizes_branch, dim, K, activation) 
        self.trunk = Trunk(layer_sizes_trunk, K, activation)        
        

    def forward(self, iv, t):
        """DeepONet forward pass

        Args:
            iv : initial values to predict from
            t : times to predict to

        Returns:
            Prediction of solutions of initial values given by iv at times given by t
        """
        # Get branch and trunk output
        trunk_outputs, branch_outputs  = self.trunk(t), self.branch(iv)
        
        # Conservative forward pass
        if self.scheme in ['QR', 1]:
                        
            # Orthogonormalise branch outputs
            Q, R = torch.linalg.qr(branch_outputs)    
                
            # Rescale branch to match inital norm
            branch_outputs = Q*torch.linalg.norm(iv, dim=-1)[:,None,None]
            
            # Transform trunk coordinates to match the new branch vectors
            trunk_outputs = torch.einsum("ukK,tK->utK", R, trunk_outputs)
            
            # Normalise trunk coordinates
            trunk_outputs = nn.functional.normalize(trunk_outputs, dim=-1)
            
            # Get network output by multiplying 'trunk vector' by 'branch matrix'
            output = torch.einsum("udK,utK->utd", branch_outputs, trunk_outputs)
            
        # Vanilla forward pass                                  
        else:
            
            # Get network output by multiplying 'trunk vector' by 'branch matrix'
            output = torch.einsum("udK,tK->utd", branch_outputs, trunk_outputs)
                
        return output
    
    
class ComplexDeepONet(nn.Module):
    """Deep Operator Neural Network

    Args:
        layer_sizes_branch : shape of branch net
        layer_sizes_trunk : shape of trunk net
        K : number of branch vectors
        dim : dimension of input / output
        activation : nonlinear activation function for both branch and trunk
        scheme : scheme used for predicting outputs (e.g. vanilla or conservative)
    """
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, K, dim, activation, scheme=None):
        
        super().__init__()
        
        self.dim = dim
        self.scheme = scheme
        
        activation = {'relu' : nn.ReLU(), 'tanh':nn.Tanh(), 'softplus' : nn.Softplus(), 'htanh' : nn.Hardtanh()}[activation]   

        # Create branch- and trunk-net
        self.complex_branch = ComplexBranch(layer_sizes_branch, dim // 2, K, activation) 
        self.real_trunk = Trunk(layer_sizes_trunk, K, activation)
        self.imag_trunk = Trunk(layer_sizes_trunk, K, activation) 

    def forward(self, iv, t):
        """DeepONet forward pass

        Args:
            iv : initial values to predict from
            t : times to predict to

        Returns:
            Prediction of solutions of initial values given by iv at times given by t
        """
                
        # Get real and complex part of branch output
        B = self.complex_branch(iv)
        
        # Get real and complex part of trunk output
        trunk = self.real_trunk(t) + 1j * self.imag_trunk(t)
        
        if self.scheme == 'QR':
                        
            # Get (complex) norm of initial values
            norm_iv = torch.linalg.norm(iv[...,self.dim//2:] + 1j * iv[...,:self.dim//2], dim=-1)
            
            # Orthogonormalise branch outputs
            Q, R = torch.linalg.qr(B)    
                
            # Rescale branch to match inital norm
            B = Q * norm_iv[:,None,None]
            
            # Transform trunk coordinates to match the new branch vectors
            trunk = torch.einsum("ukK,tK->utK", R, trunk)
            
            # Normalise trunk coordinates
            trunk = nn.functional.normalize(trunk, dim=-1)
            
            # Get network output by multiplying 'trunk vector' by 'branch matrix'
            out = torch.einsum("udK,utK->utd", B, trunk)
        
        else:
            # Multiply branch matrix by trunk vector
            out = torch.einsum("udK,tK->utd", B, trunk)
        
        return torch.cat((torch.imag(out), torch.real(out)), axis=-1)