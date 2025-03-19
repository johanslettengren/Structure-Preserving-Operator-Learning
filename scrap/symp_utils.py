import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SympNet(torch.nn.Module):
    def __init__(self, layer_sizes, N, h, activation='abs'):
        super().__init__()

        dict = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs,
            'FNN' : FNN([1, 5, 5, 5, 1], 'sigmoid')}  
        
        
        self.activation = dict[activation]  
        
        # Create layers
        self.linears = nn.ModuleList() 
        self.biases = nn.ParameterList()
        
        self.diag_layer = nn.ParameterList()
        
        self.bias_1 = nn.ParameterList()
        self.bias_2 = nn.ParameterList()
        
        self.a = nn.ParameterList()
        
        self.h = h
        
        self.d = nn.ParameterList()
                
        for i in range(len(layer_sizes)):
            
            layer = nn.Linear(N, layer_sizes[i], bias=True)
            
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
            self.linears.append(layer)
            self.biases.append(nn.Parameter(torch.zeros(N)))
            self.bias_1.append(nn.Parameter(torch.zeros(N)))
            self.bias_2.append(nn.Parameter(torch.zeros(N)))
            self.a.append(h * nn.Parameter(torch.randn(2)))
            
            self.d.append(nn.Parameter(torch.randn(layer_sizes[i], 1)))
            
            self.diag_layer.append(nn.Parameter(torch.randn(1, N)))

        
            
    # Trunk forward pass
    def forward(self, p, q):
        
                
        for l, linear in enumerate(self.linears):            
            
            a, b = self.a[l][0], self.a[l][1]       
            z = a * p + b * q
                    
            z = self.activation(linear(z).unsqueeze(-1)).squeeze(-1)
            z = torch.matmul(z, self.d[l] * linear.weight) + self.biases[l]
            
            p = p + b * z
            q = q - a * z
        

        return (p, q)

class ComplexSympNet(torch.nn.Module):
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
            
            real_layer = nn.Linear(N, layer_sizes[i], bias=True)
            imag_layer = nn.Linear(N, layer_sizes[i], bias=True)
            
            nn.init.normal_(real_layer.weight, std=h)
            nn.init.normal_(imag_layer.weight, std=h)
            
            nn.init.zeros_(real_layer.bias)
            nn.init.zeros_(imag_layer.weight)
            
            self.linears.append(nn.ModuleList([real_layer, imag_layer]))
            self.biases.append(nn.Parameter(torch.zeros(N))) 
            self.diags.append(h * nn.Parameter(torch.randn(layer_sizes[i], 1)))
            self.a.append(h * nn.Parameter(torch.randn(4)))

    def forward(self, p, q):
        
        q_r, q_i, p_r, p_i = torch.real(q), torch.imag(q), torch.real(p), torch.imag(p)
                        
        for l in range(len(self.linears)):            
            
            a_r, a_i, b_r, b_i = self.a[l][0], self.a[l][1], self.a[l][2], self.a[l][3]
            
            real = (a_r * q_r - a_i * q_i) + (b_r * p_r - b_i * p_i)
            imag = (a_r * q_i + a_i * q_r) + (b_r * p_i + b_i * p_r)
            
            real_ = self.activation(self.linears[l][0](real) - self.linears[l][1](imag))
            imag_ = self.activation(self.linears[l][0](imag) + self.linears[l][1](real))
                        
            real = torch.matmul(real_, self.diags[l] * self.linears[l][0].weight) - torch.matmul(imag_, self.diags[l] * self.linears[l][1].weight) #+ self.biases[l][:,0]
            imag = torch.matmul(imag_, self.diags[l] * self.linears[l][0].weight) + torch.matmul(real_, self.diags[l] * self.linears[l][1].weight) + self.biases[l]
            
            q_r = q_r + (b_r * real - b_i * imag)
            q_i = q_i + (b_r * imag + b_i * real)
            
            p_r = p_r - (a_r * real - a_i * imag)
            p_i = p_i - (a_r * imag + a_i * real)

            q = q + q_r + 1j * q_i
            p = p + p_r + 1j * p_i

        return (p, q)
    
    
class SymJacNet(torch.nn.Module):
    def __init__(self, layer_sizes, N, h, activation='abs'):
        super().__init__()

        dict = {
            'relu' : nn.ReLU(), 
            'tanh' : nn.Tanh(), 
            'softplus' : nn.Softplus(), 
            'htanh' : nn.Hardtanh(), 
            'sigmoid' : nn.Sigmoid(),
            'hsigmoid' : nn.Hardsigmoid(), 
            'tanhshrink' : nn.Tanhshrink(),
            'abs' : torch.abs,
            'FNN' : FNN([1, 5, 5, 5, 1], 'sigmoid')}  
        
        
        self.activation = dict[activation]  
        
        # Create layers
        self.linears = nn.ModuleList() 
        self.biases = nn.ParameterList()
                
        self.h = h

        self.d = nn.ParameterList()
                
        for i in range(len(layer_sizes)):
            
            layer = nn.Linear(2*N, layer_sizes[i], bias=True)
            
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
            self.linears.append(layer)
            self.biases.append(nn.Parameter(torch.zeros(2*N)))            
            self.d.append(nn.Parameter(torch.randn(layer_sizes[i], 1)))

        
            
    # Trunk forward pass
    def forward(self, p, q):
        
        z = torch.cat((p, q), axis=-1)
                
        for l, linear in enumerate(self.linears):            
        
            z = self.activation(linear(z).unsqueeze(-1)).squeeze(-1)

            z = torch.matmul(z, self.d[l] * linear.weight) + self.biases[l]
            
            p = p - self.h * z[...,z.shape[-1] // 2:]
            q = q + self.h * z[...,:z.shape[-1] // 2]
        

        return (p, q)  
    
class EquivariantNet(torch.nn.Module):
    def __init__(self, layer_sizes, N, h, activation='abs'):
        super().__init__()
        
        
        self.h = h
        self.phasenet =FNN([N] + layer_sizes + [N], activation)
        self.modnet =FNN([N] + layer_sizes + [N], activation)

        
            
    # Trunk forward pass
    def forward(self, p, q):
        
        xi = (q + 1j * p) / torch.sqrt(torch.tensor(2))
        
        phase = torch.angle(xi)
        mod = torch.abs(xi)
        
        
        m = phase.mean(-1).unsqueeze(-1)
                
        phase = self.phasenet(phase - m) + m
        mod = torch.abs(self.modnet(mod)) 
        
        #print(mod * torch.sin(phase) * torch.sqrt(torch.tensor(2)))
                
        #print(xi.shape)
        
        p = p + self.h * mod * torch.sin(phase) * torch.sqrt(torch.tensor(2))
        q = q + self.h * mod * torch.cos(phase) * torch.sqrt(torch.tensor(2))
        

        return (p, q)    

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
            nn.init.normal_(L1.weight, gain=h)
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

class VanillaRollOut(torch.nn.Module):
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
                        
        #layer_sizes = [2*N] + layer_sizes + [2*N]
        
        layer_sizes = layer_sizes
        
        # Create layers
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Initialize weights (normal distribution)
            nn.init.xavier_normal_(layer.weight)
            # Initialize biases (zeros)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
    def forward(self, p, q):
        
        z = torch.cat((p, q), axis=-1)
    
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)
        
        z = z / torch.linalg.norm(z, axis=-1, keepdim=True)
    

        return (z[..., :z.shape[-1]//2], z[..., z.shape[-1]//2:])

# Fully connected neural network
class FNN(torch.nn.Module):
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
                        
        #layer_sizes = [2*N] + layer_sizes + [2*N]
        
        layer_sizes = layer_sizes
        
        # Create layers
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Initialize weights (normal distribution)
            nn.init.xavier_normal_(layer.weight)
            # Initialize biases (zeros)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)
            
    def forward(self, z):
    
        for linear in self.linears[:-1]:
            z = self.activation(linear(z))
            
        z = self.linears[-1](z)
    

        return z



# Compile and train model + data
class SympModel():
    
    def __init__(self, x_train, y_train, x_test, y_test, net, lr=0.001, loss='fourier'):  
        
                
        # Training data
        self.x_train = x_train
        self.y_train = y_train[0] if (loss == 'fourier') else y_train[1]
        
        # Testing data
        self.x_test = x_test
        self.y_test = y_test[0] if (loss == 'fourier') else y_test[1]
                
        self.bestvloss = torch.tensor(torch.inf)    
                        
        # Network
        self.net = net
        
        # Loss history
        self.tlosshistory = []  # training loss
        self.vlosshistory = []  # validation/test loss

        # Initialize optimizer (default adam)
        optimizer = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}['adam']
        self.optimizer = optimizer(net.parameters(), lr=lr)

        N = y_train[0].shape[-1]
        M = y_train[1].shape[-1]
        
        self.ifft = torch.fft.ifft if (N == M) else torch.fft.irfft
        
        self.loss = self.fourier_loss if (loss == 'fourier') else self.spatial_loss  
        
        xi = (x_train[1] + 1j * x_train[0]) / torch.sqrt(torch.tensor(2))
        
        #xi = torch.concat((torch.zeros_like(xi[:,None,0]), xi), axis=-1)
        
        
        self.scale = torch.linalg.norm(xi - y_train[0]) if (loss == 'fourier') \
            else torch.linalg.norm(self.ifft(xi, axis=-1) - y_train[1])
    
        print(self.scale)
        
    
    def rel_l2(self, prediction, target):
        return torch.linalg.norm(prediction - target) / self.scale #torch.linalg.norm(target)
        
    
    def fourier_loss(self, output, target):
        
        xi = (output[1] + 1j * output[0]) / torch.sqrt(torch.tensor(2))
                
        # energy_term = torch.linalg.norm(torch.abs(xi) - torch.abs(target))
        return self.rel_l2(xi, target)
    
    def spatial_loss(self, output, target):
        
        xi = (output[1] + 1j * output[0]) / torch.sqrt(torch.tensor(2))
        xi = torch.concat((torch.zeros_like(xi[:,None,0]), xi), axis=-1)
        u = self.ifft(xi, axis=-1)
        
        return self.rel_l2(u, target)
        
        
        
    # Format data as torch tensor with dtype=float32    
    def format(self, x, requires_grad=False):
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        return x.to(torch.float32).requires_grad_(requires_grad)
    
    
        
    # Train network
    def train(self, iterations):
        
        # Train step history
        self.steps = []
        self.net.train(True)
        
        print('step \t train loss \t test loss')
        
        for iter in range(iterations):
                                    
            # Train
            self.optimizer.zero_grad()
            
            out = self.net(*self.x_train) 
            
            loss = self.loss(out, self.y_train) #+ sum(p.abs().sum() for p in self.net.parameters())
            

            loss.backward(retain_graph=True)
            self.optimizer.step()
            tloss = loss.item()

            # Test
            if iter % 100 == 99:
                # Set net to evalutation mode
                self.net.eval()
                
                # Don't calculate gradients
                with torch.no_grad():
                    
                    out = self.net(*self.x_test)  
                    vloss = self.loss(out, self.y_test)
                                        
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
                
                print('{} \t [{:.2e}] \t [{:.2e}] \t {}'.\
                    format(iter + 1, tloss, vloss, announce_new_best))    
                
        print('Best testing loss:', self.bestvloss.item())
        self.net.load_state_dict(torch.load("best_model.pth", weights_only=True))
        self.net.eval()
        
    
    # Predict output using DeepONet
    def predict(self, X, h, Tmax=1):
        
        iters = int(Tmax / h)
        
        p, q = X


        for _ in range(iters):
            
            X = self.net(*X)
            p = torch.cat((p, X[0]), axis=0)
            q = torch.cat((q, X[1]), axis=0)
        
        return (p, q)
    
    
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