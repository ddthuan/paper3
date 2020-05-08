from libcore import *

class CConv(nn.Module):
    def __init__(self, kernel_size):
        super(CConv, self).__init__()
        self.kernel_size = kernel_size
        self.real = nn.Parameter(torch.randn(kernel_size))
        self.imag = nn.Parameter(torch.randn(kernel_size))
        self.conv = nn.Conv2d(1,4, kernel_size=kernel_size, stride=(2,2), padding=(1,1), bias=False)
    
    def forward(self, x):
        
        filters = torch.stack([
                self.real[None] * self.real[:,None],
                self.real[None] * self.imag[:,None],
                self.imag[None] * self.real[:,None],
                self.imag[None] * self.imag[:,None]
                ], dim=0)
    
        self.conv.weight.data = filters[:, None].data
        
        out = self.conv(x)
        return out

conv = CConv(3)
x = torch.randn(1,1,32,32)
y = conv(x)