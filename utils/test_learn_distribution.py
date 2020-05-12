from libcore import *


def G(n, s):
    x = torch.cat([torch.arange(0, n/2, 1), torch.arange(-n/2,0,1)], dim=0)
    [Y,X] = torch.meshgrid(x,x)
    h = torch.exp((-X**2-Y**2)/(2*s**2))
    h = h/torch.sum(h)
    return h

# =============================================================================
# for i in range(1,10):
#     filter_g = G(5,i)
#     print(filter_g)
# =============================================================================

import numbers
import math
    
class VarFilter(nn.Module):
    def __init__(self, channels, kernel_size):
        super(VarFilter, self).__init__()
        print("torch:")
        self.sigma = nn.Parameter(torch.randn(1, requires_grad=True))
        #self.sigma = nn.parameter.Parameter(torch.tensor(gamma, requires_grad=True))
        
        #sigma = 1
        sigma2 = torch.cat([self.sigma, self.sigma], axis=0)
        
        print('sigma: ', sigma2)
            
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * 2
        
        kernel = 1
        
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        

        for size, std, mgrid in zip(kernel_size, sigma2, meshgrids):
            #print("size: {}, std: {}, meshgrid: {}".format(size, std, mgrid))
            
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)
    
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
        print(kernel)
        
        self.register_buffer('weight', kernel)
        self.groups = channels
        
        self.conv = F.conv2d
        
        #print(kernel)
    
    def forward(self, x):
        return self.conv(x, weight=self.weight, groups=self.groups)
        

C = 3
ksize = 5

data = torch.rand(3, C, 100, 100)
data = F.pad(data, (2, 2, 2, 2), mode='reflect')

gModel = VarFilter(C, ksize)
y = gModel(data)
print("y shape: ", y.shape)