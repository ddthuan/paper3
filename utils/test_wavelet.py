from libcore import *

w=pywt.Wavelet('bior2.2')

dec_hi = torch.tensor(w.dec_hi[::-1]) 
dec_lo = torch.tensor(w.dec_lo[::-1])
rec_hi = torch.tensor(w.rec_hi)
rec_lo = torch.tensor(w.rec_lo)

filters = torch.stack([dec_lo[None] * dec_lo[:, None],
                       dec_lo[None] * dec_hi[:, None],
                       dec_hi[None] * dec_lo[:, None],
                       dec_hi[None] * dec_hi[:, None]], dim=0)

inv_filters = torch.stack([rec_lo[None] * rec_lo[:, None],
                           rec_lo[None] * rec_hi[:, None],
                           rec_hi[None] * rec_lo[:, None],
                           rec_hi[None] * rec_hi[:, None]], dim=0)

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

conv = CConv(7)
x = torch.randn(1,1,32,32)
y = conv(x)

# =============================================================================
# k = torch.randn(3,3)
# print(k.repeat(4,1,1)[:, None])
# =============================================================================
