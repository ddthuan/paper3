# https://github.com/adobe/antialiased-cnns/blob/master/models_lpf/vgg.py

from libcore import *

lp = torch.zeros([1, 7, 7])
hp_real = torch.zeros(8, 7, 7)
hp_imag = torch.zeros(8, 7, 7)

lp[0] = torch.tensor(pd.read_csv('filters/filter_lowpass.csv').values).data

for i in range(8):
    hp_real[i] = torch.tensor(pd.read_csv('filters/real_{}.csv'.format(i)).values).data
    hp_imag[i] = torch.tensor(pd.read_csv('filters/imag_{}.csv'.format(i)).values).data

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

img_name = 'trump'
imgg = Image.open('{}.jpg'.format(img_name))

imgg_np = np.array(imgg)
imgg_tensor = Fv.to_tensor(imgg)[0:3, :, :][None]

x = imgg_tensor.detach()
B, C, N1, N2 = x.shape
print('N1: {}, N2: {}'.format(N1, N2))

print('lp: ', lp.shape)
lp_tensor = lp.repeat(3,1,1,1)
print('lp_tesor: ', lp_tensor.shape)

class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

        
class PoolLayer(nn.Module):
    def __init__(self, step):
        super(PoolLayer, self).__init__()
        self.conv = nn.Conv2d(3,3,7, stride=(1,1), padding=(3,3), groups=3, bias=False)
        self.pool = nn.MaxPool2d((2,2), stride=step)
        self.conv.weight.data = lp_tensor
        #print('out: ', self.conv.weight.shape)
        #print('lp: ', lp_tensor.shape)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        return out

model = PoolLayer(2)
y_r = model(x)
y_regular = model(y_r)
torchvision.utils.save_image(y_regular[0].detach(), "pool_regular.png")

class PoolLayer_Antialias(nn.Module):
    def __init__(self):
        super(PoolLayer_Antialias, self).__init__()
        self.conv = nn.Conv2d(3,3,7, stride=(1,1), padding=(3,3), groups=3, bias=False)
        self.pool = Downsample(channels=3)
        self.conv.weight.data = lp_tensor        
        
    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        return out

pool_custom = PoolLayer_Antialias()
y_c = pool_custom(x)
y_custom = pool_custom(y_c)
torchvision.utils.save_image(y_custom[0].detach(), "pool_custom.png")
    












    