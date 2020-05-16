import torch
from torch import nn
from torch.nn import functional as F


class Scatt_OneOrder(nn.Module):
    def __init__(self, psi_num, C, S):
        super(Scatt_OneOrder, self).__init__()
        self.psi_num = psi_num
        #self.channel_in = channel_in
        #self.stride = S
        
        self.phi = nn.Conv2d(C,C,kernel_size=7, stride=S, padding=(3,3), bias=False, groups=C)
        
        self.psi_real = nn.Conv2d(C, out_channels= C * self.psi_num, kernel_size=7, stride=S, padding=(3,3), bias=False, groups=C)
        self.psi_imag = nn.Conv2d(C, out_channels= C * self.psi_num, kernel_size=7, stride=S, padding=(3,3), bias=False, groups=C)
        
        self.filt_phi_smooth = self.phi.weight.data.repeat(self.psi_num, 1, 1, 1)
        
        print('smooth phi filter: ',self.filt_phi_smooth.shape)
        
        self.pad = nn.ZeroPad2d(3)
        
    def forward(self, x):
        # The zero order scattering transform.
        s0Phi = self.phi(x)
        print('S0: ', s0Phi.shape)
        # The one order scattering transform:
        # 1. Module oparation ( real part & imaginary part )
        #psi = torch.sqrt(self.psi1_real(x)**2 + self.psi1_imag(x)**2 + self.eps)
        
        u1Psi = torch.sqrt(self.psi_real(x)**2 + self.psi_imag(x)**2)
        print('U1 psi: ', u1Psi.shape)
        
        # mean filter for the output of psi
        u1Psi_smooth = F.conv2d(self.pad(u1Psi), self.filt_phi_smooth, bias=None, groups = u1Psi.size(1))
        print('U1 psi smooth: ', u1Psi_smooth.shape)
        
        s1 = torch.cat([s0Phi, u1Psi_smooth], axis = 1)
        print('S1: ', s1.shape)
        return s1
        
        
class Scatt_TwoOrder(nn.Module):
    def __init__(self, psi_num, C, S):
        super(Scatt_TwoOrder, self).__init__()
        self.psi_num = psi_num
        
        self.phi = nn.Conv2d(C, C, kernel_size=7, stride=2, padding=(3,3), bias=False, groups=C)
        
        self.psi_real = nn.Conv2d(C, out_channels = C * self.psi_num, kernel_size=7, stride=S, padding=(3,3), bias=False, groups=C)
        self.psi_imag = nn.Conv2d(C, out_channels = C * self.psi_num, kernel_size=7, stride=S, padding=(3,3), bias=False, groups=C)
        
        self.filt_phi_smooth_u1 = self.phi.weight.data.repeat(self.psi_num, 1, 1, 1)
        
        self.filt_phi_downsample = self.phi.weight.data.repeat(self.psi_num+1, 1, 1, 1)
        self.filt_psi_real = self.psi_real.weight.data.repeat(self.psi_num+1, 1, 1, 1)
        self.filt_psi_imag = self.psi_imag.weight.data.repeat(self.psi_num+1, 1, 1, 1)        
        
        self.filt_phi_smooth_u2 = self.phi.weight.data.repeat((self.psi_num * (self.psi_num + 1)), 1, 1, 1)
        
        print('smooth L1 phi filter: ', self.filt_phi_smooth_u1.shape)
        
        # for next order.        
        print('downsample phi filter: ', self.filt_phi_downsample.shape)        
        
        print('real psi filter ', self.filt_psi_real.shape)
        print('imag psi filter: ', self.filt_psi_imag.shape)
        
        print('smooth L2 phi filter: ', self.filt_phi_smooth_u2.shape)
        
        self.pad = nn.ZeroPad2d(3)
        
    def forward(self, x):
        # The zero order scattering transform.
        s0Phi = self.phi(x)
        print('S0: ', s0Phi.shape)
        # The one order scattering transform:
        # 1. Module oparation ( real part & imaginary part )
        #psi = torch.sqrt(self.psi1_real(x)**2 + self.psi1_imag(x)**2 + self.eps)
        
        u1Psi = torch.sqrt(self.psi_real(x)**2 + self.psi_imag(x)**2)
        #print('psi: ', sOrder1_psi.shape)
        
        # mean filter for the output of psi; for only smooth
        u1Psi_smooth = F.conv2d(self.pad(u1Psi), self.filt_phi_smooth_u1, bias=None, groups = u1Psi.size(1))
        print('U1: ', u1Psi_smooth.shape)
        
        s1 = torch.cat([s0Phi, u1Psi_smooth], axis = 1)
        print('S1: ', s1.shape)
                
        s1Phi = F.conv2d(self.pad(s1), self.filt_phi_downsample, bias=None, stride=2, groups=s1.size(1))
        print('S1 phi: ', s1Phi.shape)
        
        u2Psi = torch.sqrt(
                F.conv2d(self.pad(s1), self.filt_psi_imag, bias=None, stride=2, groups = s1.size(1))**2 +
                F.conv2d(self.pad(s1), self.filt_psi_real, bias=None, stride=2, groups = s1.size(1))**2
                )
        print('U2 psi: ', u2Psi.shape)
        
        u2Psi_smooth = F.conv2d(self.pad(u2Psi), self.filt_phi_smooth_u2, bias=None, groups=u2Psi.size(1))
        print('U2 psi smooth: ', u2Psi_smooth.shape)
        
        s2 = torch.cat([s1Phi, u2Psi_smooth], axis=1)
        print('S2 :', s2.shape)
        
        return s2

channel_in = 3
x = torch.randn(1, channel_in,32,32)
m = Scatt_TwoOrder(6,channel_in, 2)
y = m(x)

print("==================test F.con2d =============")
conv = nn.Conv2d(2,4,3,stride=(1,1), padding=(1,1), groups=2)
print('weight: ', conv.weight.shape)
