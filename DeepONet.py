import torch
import torch.nn as nn

class branch(nn.Module):
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
        
class trunk(nn.Module):
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
        
        activation = {'relu' : nn.ReLU(), 'tanh':nn.Tanh(), 'softplus' : nn.Softplus(), 'htanh' : nn.Hardtanh()}[activation]   

        # Create branch- and trunk-net
        self.branch = branch(layer_sizes_branch, dim, K, activation) 
        self.trunk = trunk(layer_sizes_trunk, K, activation)        
        

    def forward(self, iv, t):
        """DeepONet forward pass

        Args:
            iv : initial values to predict from
            t : times to predict to

        Returns:
            Prediction of solutions of initial values given by iv at times given by t
        """
        
        # Conservative forward pass
        if self.scheme in ['QR', 1]:
            
            # Get branch- and trunk output
            trunk_outputs, branch_outputs  = self.trunk(t), self.branch(iv)
                        
            # Orthogonormalise branch outputs
            Q, R = torch.linalg.qr(branch_outputs)    
                
            # Rescale branch to match inital norm
            branch_outputs = Q*torch.linalg.norm(iv, dim=-1)[:,None,None]
            
            # Transform trunk coordinates to match the new branch vectors
            trunk_outputs = torch.einsum("ukK,tK->utK", R, trunk_outputs)
            
            # Normalise trunk coordinates
            trunk_outputs = nn.functional.normalize(trunk_outputs, dim=2)
            
            # Get network output by multiplying 'trunk vector' by 'branch matrix'
            output = torch.einsum("udK,utK->utd", branch_outputs, trunk_outputs)
            
        # Vanilla forward pass                                  
        else:
            
            # Get branch- and trunk output
            trunk_outputs  = self.trunk(t)
            branch_outputs = self.branch(iv)  
            
            # Get network output by multiplying 'trunk vector' by 'branch matrix'
            output = torch.einsum("udK,tK->utd", branch_outputs, trunk_outputs)
                
        return output
    
    
# I = torch.eye(dim // 2)
# O = torch.zeros_like(I)
# self.J = torch.cat([torch.cat([O, I], dim=1), torch.cat([-I, O], dim=1) ], dim=0)

# if scheme in ['preorth', 2]:
#     self.branch = branch(layer_sizes_branch, dim, K, activation=activation)
#     self.trunk = trunk(layer_sizes_trunk, K, activation=activation)     
    
    
# elif self.scheme in ['preorth', 2]:
#     trunk_outputs  = self.trunk(t)
#     branch_outputs = self.branch(momenta)       
#     b1 = self.branch(momenta).squeeze(-1)
#     #J = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)

#     b2 = torch.einsum ('ud, dD -> uD', b1, self.J)
#     branch_outputs = torch.stack((b1, b2), axis=-1)    
    
#     trunk_outputs = nn.functional.normalize(trunk_outputs, dim=-1)
#     output = torch.einsum("udK,tK->utd", branch_outputs, trunk_outputs)   