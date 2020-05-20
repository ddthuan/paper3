import numpy as np
from matplotlib import pyplot as plt

def plot_filters_single_channel(t):
    
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
   
    plt.tight_layout()
    plt.show()
    
from model import Scatt_OneOrder

model = Scatt_OneOrder(8,1,2)
model.to("cuda:0")
model_name = model.__class__.__name__

#0k, no eps
#PATH = './model/Scatt_OneOrder_cifar10_level1_Smooth_best.pth'

#Ok
PATH = './model/Scatt_OneOrder_random_level1_Smooth_epsConstant.pth'



# Load pre-traing model
model.load_state_dict(torch.load(PATH))
model.eval()
 
plot_filters_single_channel(model.phi.weight.detach().cpu())
plot_filters_single_channel(model.psi_real.weight.detach().cpu())
plot_filters_single_channel(model.psi_imag.weight.detach().cpu())
    
    