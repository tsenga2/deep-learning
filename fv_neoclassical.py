# Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm

# Plotting settings
fontsize = 14
ticksize = 14
figsize  = (12, 4.5)
params   = {'font.family':'serif',
    "figure.figsize":figsize,
    'figure.dpi': 80,
    'figure.edgecolor': 'k',
    'font.size': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': ticksize,
    'ytick.labelsize': ticksize
}
plt.rcParams.update(params)

# Set up the model parameters
class Params:
    def __init__(self,
                 alpha = 1.0/3.0,
                 beta = 0.9,
                 delta = 0.1,
                 k_0 = 1.0,
                ):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.k_0 = k_0

# Define functions
def f(k):
    alpha = Params().alpha
    return k**alpha
def f_prime(k):
    alpha = Params().alpha
    return  alpha*(k**(alpha -1))
def u_prime(c):
    out = c.pow(-1)
    return out

class SS: #steady state
    def __init__(self):
        self.delta = Params().delta
        self.beta = Params().beta
        self.alpha = Params().alpha
        base = ((1.0/self.beta)-1.0+self.delta)/self.alpha
        exponent = 1.0/(self.alpha-1)
        self.k_ss = base**exponent
        self.c_ss = f(self.k_ss)-self.delta*self.k_ss
        
class Grid_data:
    def __init__(self,
                 max_T = 32,
                 batch_size = 8):
        self.max_T = max_T
        self.batch_size = batch_size
        self.time_range = torch.arange(0.0, self.max_T , 1.0)
        self.grid = self.time_range.unsqueeze(dim = 1)

class Data_label(Dataset):

    def __init__(self,data):
        self.time = data
        self.n_samples = self.time.shape[0]

    def __getitem__(self,index):
            return self.time[index]

    def __len__(self):
        return self.n_samples

train_data = Grid_data().grid
train_labeled = Data_label(train_data)
train = DataLoader(dataset = train_labeled, batch_size = 8 , shuffle = True )

class NN(nn.Module):
    def __init__(self,
                 dim_hidden = 128,
                layers = 4,
                hidden_bias = True):
        super().__init__()
        self.dim_hidden= dim_hidden
        self.layers = layers
        self.hidden_bias = hidden_bias

        torch.manual_seed(123)
        module = []
        module.append(nn.Linear(1,self.dim_hidden, bias = self.hidden_bias))
        module.append(nn.Tanh())

        for i in range(self.layers-1):
            module.append(nn.Linear(self.dim_hidden,self.dim_hidden, bias = self.hidden_bias))
            module.append(nn.Tanh())

        module.append(nn.Linear(self.dim_hidden,2))
        module.append(nn.Softplus(beta = 1.0)) #The softplus layer ensures c>0,k>0

        self.q = nn.Sequential(*module)


    def forward(self, x):
        out = self.q(x) # first element is consumption, the second element is capital
        return  out

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

q_hat= NN()
learning_rate = 1e-3
optimizer = torch.optim.Adam(q_hat.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

delta = Params().delta
beta = Params().beta
k_0 = Params().k_0

num_epochs = 1001

for epoch in range(num_epochs):
    for i, time in enumerate(train):
        time_zero = torch.zeros([1,1])
        time_next = time+1
        c_t = q_hat(time)[:,[0]]
        k_t = q_hat(time)[:,[1]]
        c_tp1 = q_hat(time_next)[:,[0]]
        k_tp1 = q_hat(time_next)[:,[1]]
        k_t0 = q_hat(time_zero)[0,1]

        res_1 = c_t-f(k_t)-(1-delta)*k_t + k_tp1 #Budget constraint
        res_2 = (u_prime(c_t)/u_prime(c_tp1)) - beta*(f_prime(k_tp1)+1-delta) #Euler
        res_3 = k_t0-k_0 #Initial Condition

        loss_1 = res_1.pow(2).mean()
        loss_2 = res_2.pow(2).mean()
        loss_3 = res_3.pow(2).mean()
        loss = 0.1*loss_1+0.8*loss_2+0.1*loss_3

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    scheduler.step()

    if epoch == 0:
         print('epoch' , ',' , 'loss' , ',', 'loss_bc' , ',' , 'loss_euler' , ',' , 'loss_initial' ,
               ',', 'lr_rate')
    if epoch % 100 == 0:
          print(epoch,',',"{:.2e}".format(loss.detach().numpy()),',',
                "{:.2e}".format(loss_1.detach().numpy()) , ',' , "{:.2e}".format(loss_2.detach().numpy())
               , ',' , "{:.2e}".format(loss_3.detach().numpy()), ',', "{:.2e}".format(get_lr(optimizer)) )

time_test = Grid_data().grid
c_hat_path = q_hat(time_test)[:,[0]].detach()
k_hat_path = q_hat(time_test)[:,[1]].detach()

plt.subplot(1, 2, 1)

plt.plot(time_test,k_hat_path, color='k',  label = r"Approximate capital path")
plt.axhline(y=SS().k_ss, linestyle='--',color='k', label="k Steady State")
plt.ylabel(r"k(t)")
plt.xlabel(r"Time(t)")
plt.ylim([Params().k_0-0.1,SS().k_ss+0.1 ])
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(time_test,c_hat_path,label= r"Approximate consumption path")
plt.axhline(y=SS().c_ss, linestyle='--',label="c Steady State")
plt.xlabel(r"Time(t)")
plt.ylim([c_hat_path[0]-0.1,SS().k_ss+0.1 ])
plt.tight_layout()
plt.legend(loc='best')
plt.tight_layout()
plt.show()
