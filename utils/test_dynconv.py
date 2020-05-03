from libcore import *

class DynConv_Order1(nn.Module):
    def __init__(self, C, K, S):
        # Cin : so kenh vao
        # C : tuong ung voi J trong scattering
        
        super(DynConv_Order1, self).__init__()
        
        # Dynamic Channel 3, 6, 8, .... ~ J in scattering
        self.C = C
        #self.C = (int)(nn.Parameter(torch.randint(3,32,(1,)).type(torch.float32)))
        
        self.a1 = nn.Parameter(torch.randn(1))
        self.a2 = nn.Parameter(torch.randn(1))
        self.eps = nn.Parameter(torch.randn(1))
        
        self.kernel_size = K
        
        self.lp_conv = nn.Conv2d(1, 1, K, S, padding=(int)((self.kernel_size-1)/2))
        
        self.real_conv = nn.Conv2d(1, self.C, K, S, padding=(int)((self.kernel_size-1)/2))
        self.imag_conv = nn.Conv2d(1, self.C, K, S, padding=(int)((self.kernel_size-1)/2))
    
    def forward(self, x):     
        B, Cin, W, H = x.shape
        x = x.view(-1, 1, W, H)
        
        lp = self.lp_conv(x) * self.a1
        
        real = self.real_conv(x)
        imag = self.imag_conv(x)
        hp = torch.sqrt(real**2 + imag**2 + self.eps)* self.a2
        
        x = torch.cat([lp, hp], dim=1)
        
        return x.view(-1, (1+self.C)*Cin, x.shape[2], x.shape[3])

C = 8
# =============================================================================
# DynConv_Order2 = nn.Sequential(OrderedDict([
#         ('order1', DynConv_Order1(C, 3, 1)),
#         ('order2', DynConv_Order1(C, 3, 2))
#         ]))
# 
# =============================================================================

DynConv_Order3 = nn.Sequential(OrderedDict([
        ('order1', DynConv_Order1(C, 3, 1)),
        ('order2', DynConv_Order1(C, 3, 2)),
        ('order3', DynConv_Order1(C, 3, 2))
        ]))        

# Scatter order 4    
DynConv_Order4 = nn.Sequential(OrderedDict([
        ('order1', DynConv_Order1(C, 3, 1)),
        ('order2', DynConv_Order1(C, 3, 2)),
        ('order3', DynConv_Order1(C, 3, 1)),
        ('order4', DynConv_Order1(C, 3, 2))
        ]))
    
# =============================================================================
# model1 = DynConv_Order1(C, 7, 2)
# print(model1)
# x0 = torch.randn(10,3, 32, 32)
# y = model1(x0.detach())
# print(y.shape)
# 
# =============================================================================

x0 = torch.randn(5,3,64,64)
y4 = DynConv_Order4(x0.detach())
print(y4.shape)
print(DynConv_Order4[0].lp_conv.weight)
print(DynConv_Order4[1].lp_conv.weight)
print(DynConv_Order4[2].lp_conv.weight)
print(DynConv_Order4[3].lp_conv.weight)


# =============================================================================
# C0 = torch.randint(3,32,(1,))        
# C = nn.Parameter(torch.randint(3,32,(1,)).type(torch.float32))
# C_int = (int)(C)
# print(C)
# print(C_int)
# =============================================================================
