import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SympNet(torch.nn.Module):
    def __init__(self, layer_sizes, N, h, activation='tanh'):
        super().__init__()

        dict = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink()}  
        
        self.activation = dict[activation]   
        
        # Create layers
        self.a = nn.ParameterList()
        self.linears = nn.ModuleList()
        
        self.biases = nn.ParameterList()
        self.diags = nn.ParameterList()
        
        for i in range(len(layer_sizes)):
            
            real_layer = nn.Linear(N, layer_sizes[i], bias=False)
            imag_layer = nn.Linear(N, layer_sizes[i], bias=False)
            
            nn.init.normal_(real_layer.weight, std=h)
            nn.init.normal_(imag_layer.weight, std=h)
            
            self.linears.append(nn.ModuleList([real_layer, imag_layer]))
            self.biases.append(nn.ParameterList([nn.Parameter(torch.zeros(layer_sizes[i], 2)), nn.Parameter(torch.zeros(N, 2))]))    
            self.diags.append(h * nn.Parameter(torch.randn(layer_sizes[i])))
            self.a.append(h * nn.Parameter(torch.randn(4)))


        
            
    # Trunk forward pass
    def forward(self, q_r, q_i, p_r, p_i):
        
        ### ADD BACK DIAG !
                
        for l in range(len(self.linears)):            
            
            a_r, a_i, b_r, b_i = self.a[l][0], self.a[l][1], self.a[l][2], self.a[l][3]
            
            real = (a_r * q_r - a_i * q_i) + (b_r * p_r - b_i * p_i)
            imag = (a_r * q_i + a_i * q_r) + (b_r * p_i + b_i * p_r)
            
            real_ = self.activation(self.linears[l][0](real) - self.linears[l][1](imag) + self.biases[l][0][:,0])
            imag_ = self.activation(self.linears[l][0](imag) + self.linears[l][1](real) + self.biases[l][0][:,1])
                        
            real = torch.matmul(real_, self.linears[l][0].weight) - torch.matmul(imag_, self.linears[l][1].weight) + self.biases[l][1][:,0]
            imag = torch.matmul(imag_, self.linears[l][0].weight) + torch.matmul(real_, self.linears[l][1].weight) + self.biases[l][1][:,1]
            
            q_r = q_r + (b_r * real - b_i * imag)
            q_i = q_i + (b_r * imag + b_i * real)
            
            p_r = p_r - (a_r * real - a_i * imag)
            p_i = p_i - (a_r * imag + a_i * real)


        return (q_r, q_i, p_r, p_i)
    
class SkipNet(torch.nn.Module):
    def __init__(self, layer_sizes, N, h, activation):
        
        super().__init__()
        
        dict = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink()}  
        
        self.activation = dict[activation] 
        
        # Create layers
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            
            L1 = nn.Linear(2*N, layer_sizes[i])
            nn.init.xavier_normal_(L1.weight, gain=h)
            torch.nn.init.zeros_(L1.bias)
            
            L2 = nn.Linear(layer_sizes[i], 2*N)
            nn.init.xavier_normal_(L2.weight)
            torch.nn.init.zeros_(L2.bias)
            layer = nn.Sequential(L1, self.activation, L2)
        
            self.layers.append(layer)
            
    def forward(self, q, p):
        
        N = q.shape[-1]
        
        z = torch.cat((q, p), dim=-1)  
        
        for layer in self.layers:
            z = z + layer(z) 
        
        return (z[...,:N], z[...,N:])


# Fully connected neural network
class FNN(torch.nn.Module):
    def __init__(self, layer_sizes, N, activation):
        
        super().__init__()
        
        dict = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink()}   
        
        self.activation = dict[activation] 
                        
        layer_sizes = [2*N] + layer_sizes + [2*N]
        
        # Create layers
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Initialize weights (normal distribution)
            nn.init.xavier_normal_(layer.weight)
            # Initialize biases (zeros)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
    def forward(self, q, p):
        
        N = q.shape[-1]
        
        z = torch.cat((q, p), dim=-1)  
    
        
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)
 
    

        return (z[...,:N], z[...,N:])


        
        
    
            
            



# Compile and train model + data
class SympModel():
    
    def __init__(self, x_train, y_train, x_test, y_test, net, lr=0.001):  
                
        # Training data
        self.x_train = tuple(self.format(x, requires_grad=True) for x in x_train)
        self.y_train = tuple(self.format(y) for y in y_train)
        
        # Testing data
        self.x_test = tuple(self.format(x) for x in x_test)
        self.y_test = tuple(self.format(y) for y in y_test)
                
        self.bestvloss = 1000000
        self.bestloss = 1000000
        
        self.scales = tuple(torch.mean(torch.abs(self.y_train[i]-self.x_train[i])**2) for i in range(4))
        
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
            
            out = self.net(*self.x_train)  
                        
            loss = sum(self.loss_fn(out[i], self.y_train[i]) / self.scales[i] for i in range(4))
            
        
            #loss = self.loss_fn(outputs, self.y_train)

            loss.backward(retain_graph=True)
            self.optimizer.step()
            tloss = loss.item()

            # Test
            if iter % 1000 == 999:
                # Set net to evalutation mode
                self.net.eval()
                
                # Don't calculate gradients
                with torch.no_grad():
                    
                    
                    out = self.net(*self.x_test)  
                    vloss = sum(self.loss_fn(out[i], self.y_test[i]) / self.scales[i] for i in range(4))
                                        
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
                
        print('Best testing loss:', self.bestvloss.item())
        self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        self.net.eval()
        # print(self.net.linears[0].bias)
        # print([a for a in self.net.a])
        
    
    # Predict output using DeepONet
    def predict(self, X, h, Tmax=1):
        
        iters = int(Tmax / h)

        preds = list(X)
        
        momenta = tuple(self.format(x) for x in X)

        for _ in range(iters):
            
            momenta = self.net(*momenta)
            preds = [torch.cat((preds[i], momenta[i]), axis=0) for i in range(4)]

        
        
        return preds
    
    
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