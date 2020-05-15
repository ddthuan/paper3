from libscat import *
from torch.nn.modules.utils import _pair

# Luu y mang neural khong phu thuoc kich thuoc dau vao.
# Net need training
class PriorNet_Basic(nn.Module):
    def __init__(self):
        super(PriorNet_Basic, self).__init__()
        
        self.lowpass = nn.Conv2d(1, 1, 7, padding=3, stride=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)
        
        self.real = nn.Conv2d(1, 8, 7, padding=3, stride=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)
        
        self.imag = nn.Conv2d(1, 8, 7, padding=3, stride=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)
        
        self.eps = 1e-2
    
    def forward(self, x):        
        # Lowpass
        lp = self.pool1(self.relu1(self.bn1(self.lowpass(x))))
        
        # Bandpass
        #real = self.bn2(self.real(x))
        real = self.pool2(self.relu2(self.bn2(self.real(x))))
        #imag = self.bn3(self.imag(x))
        imag = self.pool3(self.relu3(self.bn3(self.imag(x))))
        #mag = torch.sqrt(real**2 + imag**2 + self.eps)
        mag = torch.sqrt(real**2 + imag**2 + self.eps)
        
        return torch.cat([lp, mag], axis=1)

    
class PriorNet_Complex(nn.Module):
    def __init__(self):
        super(PriorNet_Complex, self).__init__()
        self.lowpass = nn.Conv2d(1, 1, 7, padding=3, stride=2, bias=False)
        self.real = nn.Conv2d(1, 8, 7, padding=3, stride=2, bias=False)
        self.imag = nn.Conv2d(1, 8, 7, padding=3, stride=2, bias=False)
        #self.eps = 1e-2
        self.eps = nn.Parameter(torch.rand(1))
    
    def forward(self, x):
        lp = self.lowpass(x)
        real = self.real(x)
        imag = self.imag(x)
        mag = torch.sqrt(real**2 + imag**2 + self.eps)
        
        #print('EPS: -->', self.eps.data.item())
        
        return torch.cat([lp, mag], axis=1)


class PriorNet_Complex_SizeKernel(nn.Module):
    def __init__(self, kernel_size):
        super(PriorNet_Complex_SizeKernel, self).__init__()
        self.kernel_size = kernel_size
        self.lowpass = nn.Conv2d(1, 1, self.kernel_size, padding=(int)((self.kernel_size-1)/2), stride=2, bias=False)
        self.real = nn.Conv2d(1, 8, self.kernel_size, padding=(int)((self.kernel_size-1)/2), stride=2, bias=False)
        self.imag = nn.Conv2d(1, 8, self.kernel_size, padding=(int)((self.kernel_size-1)/2), stride=2, bias=False)
        self.eps = nn.Parameter(torch.rand(1))
    
    def forward(self, x):
        lp = self.lowpass(x)
        real = self.real(x)
        imag = self.imag(x)
        mag = torch.sqrt(real**2 + imag**2 + self.eps)
        
        #print('EPS: -->', self.eps.data.item())
        
        return torch.cat([lp, mag], axis=1)

class PriorNet_Complex_SizeKernel_Ex(nn.Module):
    def __init__(self, kernel_size):
        super(PriorNet_Complex_SizeKernel_Ex, self).__init__()
        self.kernel_size = kernel_size

        self.lowpass = nn.Conv2d(1, 1, self.kernel_size, padding=(int)((self.kernel_size-1)/2), stride=2, bias=False)
        
# =============================================================================
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                      nn.init.zeros_(m.bias)
# =============================================================================
        
        self.real = nn.Conv2d(1, 8, self.kernel_size, padding=(int)((self.kernel_size-1)/2), stride=2, bias=False)
        self.imag = nn.Conv2d(1, 8, self.kernel_size, padding=(int)((self.kernel_size-1)/2), stride=2, bias=False)
        
        self.eps = nn.Parameter(torch.rand(1))
        self.pen1 = nn.Parameter(torch.rand(1))
        #self.pen2 = nn.Parameter(torch.rand(1))
    
    def forward(self, x):        
        lp = self.lowpass(x) * self.pen1
        
        real = self.real(x)
        imag = self.imag(x)
        mag = torch.sqrt(real**2 + imag**2 + self.eps) * (1- self.pen1)
        
        return torch.cat([lp, mag], axis=1)
    
    
class PriorNet_Complex_Im(nn.Module):
    def __init__(self):
        super(PriorNet_Complex_Im, self).__init__()
        self.lowpass = nn.Conv2d(1, 1, 7, padding=3, stride=1, bias=False)
        self.pool1 = nn.AvgPool2d(2)
        
        self.real = nn.Conv2d(1, 8, 7, padding=3, stride=2, bias=False)
        #self.pool2 = nn.AvgPool2d(2)
        
        self.imag = nn.Conv2d(1, 8, 7, padding=3, stride=2, bias=False)
        #self.pool3 = nn.AvgPool2d(2)
        
        #self.eps = 1e-2
        self.eps = nn.Parameter(torch.rand(1))
    
    def forward(self, x):        
        lp = self.pool1(self.lowpass(x))
        real = self.real(x)
        imag = self.imag(x)
        
        #real = self.pool2(self.real(x))
        #imag = self.pool3(self.imag(x))
        mag = torch.sqrt(real**2 + imag**2 + self.eps)
        
        #print('EPS: -->', self.eps.data.item())
        
        return torch.cat([lp, mag], axis=1)
    
    
class PriorNet_Invariant(nn.Module):
    def __init__(self):
        super(PriorNet_Invariant, self).__init__()
        
        self.lowpass = nn.Conv2d(1, 1, 7, padding=3, stride=2)
        
        self.real = nn.Conv2d(1, 8, 7, padding=3, stride=2)
        self.imag = nn.Conv2d(1, 8, 7, padding=3, stride=2)
        
    def forward(self, x):        
        # Lowpass
        lp = self.lowpass(x)
        
        # Bandpass
        real = self.real(x)
        imag = self.imag(x)
        mag = torch.sqrt(real**2 + imag**2)
        
        return torch.cat([lp, mag], axis=1)    
    
    
class Model_Standard(nn.Module):
    def __init__(self):
        super(Model_Standard, self).__init__()
        self.lowpass = nn.Conv2d(1, 1, 7, padding=3, stride=2)
        self.bandpass = nn.Conv2d(1, 8, 7, padding=3, stride=2)
    def forward(self, x):        
        lp = self.lowpass(x)
        bp = self.bandpass(x)
        return torch.cat([lp, bp], axis=1)     

class Model_Raw(nn.Module):
    def __init__(self):
        super(Model_Raw, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, 7, padding=3, stride=1)
        #self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(2)
        
# =============================================================================
#         self.conv1 = nn.Conv2d(1, 9, 7, padding=3, stride=2)
#         self.relu1 = nn.ReLU(inplace=True)
# =============================================================================
        
    def forward(self, x):        
        #return self.pool1(self.relu1(self.conv1(x)))
        #return self.conv1(x)
        return self.pool1(self.conv1(x))
    


# =============================================================================
# class PriorNet_Invariant_L2(nn.Module):
#     def __init__(self):
#         super(PriorNet_Invariant, self).__init__()
#         
#         self.bl1_lowpass = nn.Conv2d(1, 1, 7, padding=3, stride=2)
#         self.bl1_real = nn.Conv2d(1, 8, 7, padding=3, stride=2)
#         self.bl1_imag = nn.Conv2d(1, 8, 7, padding=3, stride=2)
#         
#         self.bl1_lowpass = nn.Conv2d(1, 1, 7, padding=3, stride=2)
#         self.bl1_real = nn.Conv2d(1, 8, 7, padding=3, stride=2)
#         self.bl1_imag = nn.Conv2d(1, 8, 7, padding=3, stride=2)
#         
#         
#     def forward(self, x):        
#         # Lowpass
#         lp = self.lowpass(x)
#         
#         # Bandpass
#         real = self.real(x)
#         imag = self.imag(x)
#         mag = torch.sqrt(real**2 + imag**2)
#         
#         torch.cat([], axis=1)
#         
#         return torch.cat([lp, mag], axis=1)    
# =============================================================================
    
class Scatt_OrderOne(nn.Module):
    def __init__(self):
        super(Scatt_OrderOne, self).__init__()
        self.lp = nn.Conv2d(1,1,kernel_size=7, stride=1, padding=(3,3), bias=False)
        
        self.psi_real = nn.Conv2d(1,8, kernel_size=7, stride=1, padding=(3,3), bias=False)
        self.psi_imag = nn.Conv2d(1,8, kernel_size=7, stride=1, padding=(3,3), bias=False)
        
        
        
        self.eps = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        # The zero order scattering transform.
        sOrder0 = self.lp(x)
        
        # The one order scattering transform:
        # 1. Module oparation ( real part & imaginary part )
        #psi = torch.sqrt(self.psi1_real(x)**2 + self.psi1_imag(x)**2 + self.eps)
        
        sOrder1 = torch.sqrt(self.psi1_real(x)**2 + self.psi1_imag(x)**2 + self.eps)
        
        #sOrder1 = self.lp(torch.sqrt(self.psi1_real(x)**2 + self.psi1_imag(x)**2))
        
        # = [sOrder0, sOrder1]
        out_Order1 = torch.cat([sOrder0, sOrder1], axix = 1)
        
        return out_Order1


class Scatt_OneOrder_Ex(nn.Module):
    def __init__(self, psi_num):
        super(Scatt_OneOrder_Ex, self).__init__()
        self.psi_num = psi_num
        
        self.phi = nn.Conv2d(1,1,kernel_size=7, stride=2, padding=(3,3), bias=False)
        
        self.psi_real = nn.Conv2d(1, out_channels=self.psi_num, kernel_size=7, stride=2, padding=(3,3), bias=False)
        self.psi_imag = nn.Conv2d(1, out_channels=self.psi_num, kernel_size=7, stride=2, padding=(3,3), bias=False)
        
        self.filt_phi = self.phi.weight.data.repeat(self.psi_num, 1, 1, 1)
        
        self.pad = nn.ZeroPad2d(3)
        
    def forward(self, x):
        # The zero order scattering transform.
        sOrder0 = self.phi(x)
        
        # The one order scattering transform:
        # 1. Module oparation ( real part & imaginary part )
        #psi = torch.sqrt(self.psi1_real(x)**2 + self.psi1_imag(x)**2 + self.eps)
        
        sOrder1_psi = torch.sqrt(self.psi_real(x)**2 + self.psi_imag(x)**2)
        #print('psi: ', sOrder1_psi.shape)
        
        # mean filter for the output of psi
        sOrder1_psi_mean = F.conv2d(self.pad(sOrder1_psi), self.filt_phi, groups = sOrder1_psi.size(1))
        #print('psi mean: ', sOrder1_psi_mean.shape)        
        
        # = [sOrder0, sOrder1]
        out_Order1 = torch.cat([sOrder0, sOrder1_psi_mean], axis = 1)
        
        return out_Order1
    
class Scatt_TwoOrder_Ex(nn.Module):
    def __init__(self, psi_num):
        super(Scatt_TwoOrder_Ex, self).__init__()
        self.psi_num = psi_num
        
        self.phi = nn.Conv2d(1,1,kernel_size=7, stride=2, padding=(3,3), bias=False)
        
        self.psi_real = nn.Conv2d(1, out_channels=self.psi_num, kernel_size=7, stride=2, padding=(3,3), bias=False)
        self.psi_imag = nn.Conv2d(1, out_channels=self.psi_num, kernel_size=7, stride=2, padding=(3,3), bias=False)
        
        self.filt_phi = self.phi.weight.data.repeat(self.psi_num, 1, 1, 1)
        self.filt_psi_real = self.psi_real.weight.data.repeat(self.psi_num,1,1,1)
        self.filt_psi_imag = self.psi_imag.weight.data
        
        print(self.filt_psi_real.shape)
        
        self.pad = nn.ZeroPad2d(3)
        
    def forward(self, x):
        # The zero order scattering transform.
        sOrder0 = self.phi(x)
        
        # The one order scattering transform:
        # 1. Module oparation ( real part & imaginary part )
        #psi = torch.sqrt(self.psi1_real(x)**2 + self.psi1_imag(x)**2 + self.eps)
        
        sOrder1_psi = torch.sqrt(self.psi_real(x)**2 + self.psi_imag(x)**2)
        #print('psi: ', sOrder1_psi.shape)
        
        # mean filter for the output of psi
        sOrder1_psi_mean = F.conv2d(self.pad(sOrder1_psi), self.filt_phi, groups = sOrder1_psi.size(1))
        print('psi mean: ', sOrder1_psi_mean.shape)        
        
        
        
# =============================================================================
#         sOrder0_2 = F.conv2d(self.pad(sOrder0), self.filt_phi)
#         
#         sOrder2_psi = torch.sqrt(
#                 F.conv2d(self.pad(sOrder1_psi), self.filt_psi_real)**2 +
#                 F.conv2d(self.pad(sOrder1_psi), self.filt_psi_imag)**2
#                 )
#         
#         sOrder2_psi_mean = F.conv2d(self.pad(sOrder2_psi), self.filt_phi)
#         
#         # = [sOrder0, sOrder1]
#         out_Order2 = torch.cat([sOrder0_2, sOrder2_psi_mean], axis = 1)
# =============================================================================
        
        return None

    
m = Scatt_TwoOrder_Ex(8)
x = torch.rand(1,1, 200,28)
y = m(x)
#print(y.shape)

# =============================================================================
# x = torch.randn(batch,1,W1,W2).cuda()
# 
# model_prior = PriorNet()
# model_prior.to(device)
# y_prior = model_prior(x)
# print(y_prior.detach().cpu().shape)
# =============================================================================
