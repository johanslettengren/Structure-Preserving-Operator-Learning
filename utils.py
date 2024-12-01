import torch
import matplotlib.pyplot as plt

# Fully connected NN (use same structure for branch and trunk)
class FNN(torch.nn.Module):
    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        
        # Set activation function
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh}[activation]

        # Create layers
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Initialize weights (normal distribution)
            torch.nn.init.xavier_normal_(layer.weight)
            # initialize biases (zeros)
            torch.nn.init.zeros_(layer.bias)
            self.linears.append(layer)
        
    # FNN forward pass
    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x
    
# DeepONet class 
class DeepONet(torch.nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, dim_out=1, activation='relu'):
        super(DeepONet, self).__init__()
        
        # Final output dimension
        self.dim_out = dim_out
                
        # Initialize branch networks (dim_out many)
        self.branches = torch.nn.ModuleList([FNN(layer_sizes_branch, activation) for _ in range(self.dim_out)]) 
        # Initialize one trunk network (using split trunk strategy)
        self.trunk = FNN(layer_sizes_trunk, activation)
        # Initialize bias term
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, self.dim_out))

    # DeepONet forward pass
    def forward(self, x_func, x_loc):
        
        # Get one output for each branch (dim_out many)
        branch_outputs = torch.stack([branch(x_func) for branch in self.branches], dim=0)  
        # Shape (dim_out, #u, i): i = branch & trunk output dim
        
        # Get trunk ouput (only one)       
        trunk_output = self.trunk.activation(self.trunk(x_loc))
        # Shape (#t, i): i = branch & trunk output dim
        
        """ If needed in the future
        # trunk_outputs = trunk_outputs.unsqueeze(0).expand(self.dim_out, *trunk_outputs.shape)        
        # equivalent to torch.stack([self.trunk.activation(self.trunk(x_loc)) for _ in range(self.dim_out)], dim=0)"""
        
        # Equivalent to b @ t.T for each output, but in batch
        # Dot product trunk output with each branch output to get final output
        output = torch.einsum("dui,ti->utd", branch_outputs, trunk_output) + self.bias 
        # Shape (#u, #t, dim_out)

        return output


# Compile and train model + data
class Model():
    
    def __init__(
        self, x_train, y_train, x_test, y_test, net, 
        optimizer='adam', lr=0.001, Px=None, PI_loss_fn=None):     
        
        # Training data
        self.x_train = (self.format(x_train[0]), self.format(x_train[1]))
        self.y_train = self.format(y_train)
        
        # Testing data
        self.x_test = (self.format(x_test[0]), self.format(x_test[1]))
        self.y_test = self.format(y_test)
        
        self.Px_func = self.format(Px[0], requires_grad=True) if Px is not None else None
        self.Px_loc = self.format(Px[1], requires_grad=True) if Px is not None else None
        
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
                # Save loss history
                self.vlosshistory.append(vloss)
                self.tlosshistory.append(tloss)
                self.steps.append(iter)
                self.net.train(True)
                
                print('{} \t [{:.2e}] \t [{:.2e}]'.format(iter + 1, tloss, vloss))            
            
            
    def plot_losshistory(self, dpi=100):
        # Plot the loss trajectory
        _, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        ax.plot(self.steps, self.tlosshistory, label='Training loss')
        ax.plot(self.steps, self.vlosshistory, label='Test loss')
        ax.set_title("Training Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        plt.show()
    
    # Predict output using DeepONet
    def predict(self, x_func, x_loc):
        
        return self.net(self.format(x_func), self.format(x_loc))
                
        