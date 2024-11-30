import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define network class (we use the same base structure for both branch and trunk)
class FNN(nn.Module):
    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        
        self.activation = {'relu' : torch.relu, 'tanh':torch.tanh}[activation]

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            self.linears.append(layer)
        
    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x
    
    
class DeepONet(nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, dim_out=1, activation='relu'):
        super(DeepONet, self).__init__()
        
        self.dim_out = dim_out
                
        # Initialize branch and trunk networks
        self.branches = torch.nn.ModuleList([FNN(layer_sizes_branch, activation) for _ in range(self.dim_out)]) 
        self.trunks =  torch.nn.ModuleList([FNN(layer_sizes_trunk, activation) for _ in range(self.dim_out)]) 
        
        # bias term
        self.bias = nn.Parameter(torch.zeros(1, 1, self.dim_out))

    
    def forward(self, x_func, x_loc):
            
        branch_outputs = torch.stack([branch(x_func) for branch in self.branches], dim=0)  # [dim_out, batch_size_branch, branch_dim]
        trunk_outputs = torch.stack([trunk.activation(trunk(x_loc)) for trunk in self.trunks], dim=0)
        
        # Equivalent to b @ t.T for each output, but in batch
        out = torch.einsum("obi,oti->bto", branch_outputs, trunk_outputs) + self.bias # [batch_size, dim_out]

        return out



class Model():
    
    def __init__(self, x_train, y_train, x_test, y_test, net, optimizer, loss_fn=nn.MSELoss()):     
        
        self.x_train = (self.format(x_train[0]), self.format(x_train[1]))
        self.y_train = self.format(y_train)
        
        self.x_test = (self.format(x_test[0]), self.format(x_test[1]))
        self.y_test = self.format(y_test)
        
        self.net = net
        self.vlosshistory = []  # validation/test loss
        self.tlosshistory = []  # training loss
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def format(self, x):
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        return x.to(torch.float32)
    
                
    def train_one_iteration(self):
        
            self.optimizer.zero_grad()
            prediction = self.net(self.x_train[0], self.x_train[1])   
            loss = self.loss_fn(prediction, self.y_train)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        
        
    def train(self, iterations):
        
        self.steps = []
        
        print('Step \t Train loss \t Test loss')

        for iter in range(iterations):
            
            self.net.train(True)
            tloss = self.train_one_iteration()

            if iter % 1000 == 999:
                
                self.net.eval()
                with torch.no_grad():
                    outputs = self.net(self.x_test[0], self.x_test[1])   
                    vloss = self.loss_fn(outputs, self.y_test).item()
                
                self.vlosshistory.append(vloss)
                self.tlosshistory.append(tloss)
                self.steps.append(iter)
                
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
            
    def predict(self, x_func, x_loc):
        
        return self.net(self.format(x_func), self.format(x_loc))
                
        