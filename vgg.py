from libcore import *

from torchvision.models import vgg11
from torchsummary import summary

from collections import OrderedDict
from torch.autograd import Variable

from our_classifier import OurLinear

lp = torch.zeros([1, 7, 7])
hp_real = torch.zeros(8, 7, 7)
hp_imag = torch.zeros(8, 7, 7)

lp[0] = torch.tensor(pd.read_csv('filters/filter_lowpass.csv').values).data

for i in range(8):
    hp_real[i] = torch.tensor(pd.read_csv('filters/real_{}.csv'.format(i)).values).data
    hp_imag[i] = torch.tensor(pd.read_csv('filters/imag_{}.csv'.format(i)).values).data
    
    
def test(model, dataset, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in dataset:
        # data, target = data, target
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += loss_fn(output, target).item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum()

        test_loss /= len(dataset)
    print("Test set: Average loss: {:.3f},\
            Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(dataset.dataset),
                100. * float(correct) / float(len(dataset.dataset))))
    

class Vgg_Custome(nn.Module):
    def __init__(self):
        super(Vgg_Custome, self).__init__()
        self.layer1 = nn.Sequential(
                OrderedDict([
                        ('conv', nn.Conv2d(3,64,3, padding=(1,1))),
                        ('relu', nn.ReLU(inplace=True)),
                        ('pool', nn.MaxPool2d(2))
                        ])
                )
        
        self.layer2 = nn.Sequential(
                OrderedDict([
                        ('conv', nn.Conv2d(64,128,3, padding=(1,1))),
                        ('relu', nn.ReLU(inplace=True)),
                        ('pool', nn.MaxPool2d(2))
                        ])
                )
        
        self.layer3 = nn.Sequential(
                OrderedDict([
                        ('conv', nn.Conv2d(128,256,3, padding=(1,1))),
                        ('relu', nn.ReLU(inplace=True))
                        ])
                )
        
        self.layer4 = nn.Sequential(
                OrderedDict([
                        ('conv', nn.Conv2d(256,256,3, padding=(1,1))),
                        ('relu', nn.ReLU(inplace=True)),
                        ('pool', nn.MaxPool2d(2))
                        ])
                )
        self.avg = nn.AdaptiveAvgPool2d((2,2))
        self.classifer = OurLinear(1024, 10)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)        
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return out


# =============================================================================
# class MCore(nn.Module):
#     def __init__(self, chIn, chOut, kernel_size, S):
#         super(MCore, self).__init__()
#         #self.eps = nn.Parameter(torch.randn(1))
#         #self.eps1 = nn.Parameter(torch.randn(1))
#         #self.eps2 = nn.Parameter(torch.randn(1))
#         
#         self.lp = nn.Conv2d(chIn, chIn, kernel_size=kernel_size, stride=S, padding=kernel_size//2) 
#         self.real = nn.Conv2d(chIn, chOut - chIn, kernel_size=kernel_size, stride=S, padding=kernel_size//2)
#         self.imag = nn.Conv2d(chIn, chOut - chIn, kernel_size=kernel_size, stride=S, padding=kernel_size//2)
#         
#     def forward(self, x):
#         lp = self.lp(x)
#         hp = torch.sqrt(self.real(x)**2 + self.imag(x)**2)
#         return torch.cat([lp, hp], axis=1)
# =============================================================================

# =============================================================================
# # 80.5% with 3 layers
# class MCore(nn.Module):
#     def __init__(self, chIn, chOut, kernel_size, S):
#         super(MCore, self).__init__()
#         self.eps = nn.Parameter(torch.randn(1))
#         self.real = nn.Conv2d(chIn, chOut, kernel_size=kernel_size, stride=S, padding=kernel_size//2)
#         self.imag = nn.Conv2d(chIn, chOut, kernel_size=kernel_size, stride=S, padding=kernel_size//2)
#         
#     def forward(self, x):
#         return torch.sqrt(self.real(x)**2 + self.imag(x)**2)
# =============================================================================
    
class MCore(nn.Module):
    def __init__(self, chIn, chOut, kernel_size, S):
        super(MCore, self).__init__()        
        self.chIn = chIn
        self.lp = nn.Conv2d(chIn, 1, kernel_size=7, stride=S, padding=3, bias=False)
        self.eps = nn.Parameter(torch.randn(1))
        
        self.real = nn.Conv2d(chIn, chOut-1, kernel_size=kernel_size, stride=S, padding=kernel_size//2, bias=False)
        self.imag = nn.Conv2d(chIn, chOut-1, kernel_size=kernel_size, stride=S, padding=kernel_size//2, bias=False)        
        
    def forward(self, x):
        self.lp.weight.data = lp.repeat(1,self.chIn,1,1).cuda()
        lpi = self.lp(x) * self.eps
        hp = torch.sqrt(self.real(x)**2 + self.imag(x)**2) * (1-self.eps)
        return torch.cat([lpi,hp], axis=1)

# =============================================================================
# #Khong hieu qua
# class WConv(nn.Module):
#     def __init__(self, kernel_size, S):
#         super(WConv, self).__init__()
#         self.kernel_size = kernel_size
#         self.real = nn.Parameter(torch.randn(kernel_size))
#         self.imag = nn.Parameter(torch.randn(kernel_size))
#         
#         self.conv = nn.Conv2d(1,4, kernel_size=kernel_size, stride=(S,S), padding=(kernel_size//2,kernel_size//2), bias=False)
#     
#     def forward(self, x):
#         
#         filters = torch.stack([
#                 self.real[None] * self.real[:,None],
#                 self.real[None] * self.imag[:,None],
#                 self.imag[None] * self.real[:,None],
#                 self.imag[None] * self.imag[:,None]
#                 ], dim=0)
#     
#         self.conv.weight.data = filters[:, None].data
#         B,C,W,H = x.shape
#         out = x.view(-1,1,W,H)
#         out = self.conv(out)
#         out = out.view(B, -1, out.size(2), out.size(3))    
#         return out
# =============================================================================
    
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.convA = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(3, 64, 3, stride=(1,1), padding=(1,1))),
                ('bn', nn.BatchNorm2d(64)),
                ('relu', nn.ReLU(inplace=True)),
                #('pool', nn.MaxPool2d(2))
                ])
                )
    
        self.convC = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(64, 128, 3, stride=(1,1), padding=(1,1))),
                ('bn', nn.BatchNorm2d(128)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(2))
                ])
                )
    
        self.convE = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(128, 256, 3, stride=(1,1), padding=(1,1))),
                ('bn', nn.BatchNorm2d(256)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(2))
                ])
                )
    
        self.convF = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(256, 256, 3, stride=(1,1), padding=(1,1))),
                ('bn', nn.BatchNorm2d(256)),
                ('relu', nn.ReLU(inplace=True)),
                #('pool', nn.MaxPool2d(2))
                ])
                )
    
        for m in self.modules():            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.layerB = MCore(64, 64, 7, 1)
        self.layerD = MCore(128, 128, 7, 1)
        
        #self.layer4 = MCore(256, 256, 7, 2)
        self.avg = nn.AdaptiveAvgPool2d((4,4))
        self.classifier = OurLinear(4096,10)
        
        
            
    def forward(self, x):
        out = self.convA(x)
        out = self.layerB(out)
        out = self.convC(out)
        out = self.layerD(out)
        out = self.convE(out)
        out = self.convF(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
# =============================================================================
# x = torch.randn(200,3,32,32)
# model_c = ComplexCNN()
# y = model_c(x)
# #print(y.shape)
# =============================================================================




if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    batch_size = 64
    loss_fn = nn.CrossEntropyLoss()
    #clf = Vgg_Custome() # 81%
    clf = ComplexCNN() # 79%
    if use_cuda:
        clf = clf.cuda()
    optimizer = optim.Adam(clf.parameters(), lr=1e-2)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25], gamma=0.2)

    data_train = datasets.CIFAR10(root='root/', download=True, train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    train_load = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
            shuffle=True)

    data_test = datasets.CIFAR10(root='root/', download=True, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    train_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size,
            shuffle=True, )

    epoch = 1

    while epoch < 31:
        for idx, (data, target) in enumerate(train_load):
            #data, target = Variable(data), Variable(target)
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            out = clf(data)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                print('[Epoch {} | {}/{}]: {:.4f}'.format(epoch,
                    idx, len(train_load),
                    loss))
        epoch += 1
        scheduler.step()
        test(clf, train_test, epoch)
        
