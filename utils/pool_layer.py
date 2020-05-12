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
lp_tensor = lp.repeat(1,3,1,1)
print('lp_tesor: ', lp_tensor.shape)


class PoolLayer(nn.Module):
    def __init__(self, step):
        super(PoolLayer, self).__init__()
        self.conv = nn.Conv2d(3,1,7, stride=(1,1), padding=(3,3))
        self.pool = nn.MaxPool2d((2,2), stride=step)
        self.conv.weight.data = lp_tensor
        print(self.conv.weight)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        return out

for i in range(2):
    st = i+1
    model = PoolLayer(st)
    y = model(x)
    torchvision.utils.save_image(y[0].detach(), "pool_{}.jpg".format(st))
    
    #print(y.shape)
        