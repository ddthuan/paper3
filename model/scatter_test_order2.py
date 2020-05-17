from scatter_ex_fix import *

from torch.nn import functional as F
from torchvision.transforms import functional as Fv
from PIL import Image
import pandas as pd
import torchvision

from matplotlib import pyplot as plt

img = Image.open('trump.jpg')
img_tensor = Fv.to_tensor(img)
x = img_tensor[None]

lp = torch.zeros([1, 7, 7])
hp_real = torch.zeros(8, 7, 7)
hp_imag = torch.zeros(8, 7, 7)

lp[0] = torch.tensor(pd.read_csv('filters/filter_lowpass.csv').values).data

for i in range(8):
    hp_real[i] = torch.tensor(pd.read_csv('filters/real_{}.csv'.format(i)).values).data
    hp_imag[i] = torch.tensor(pd.read_csv('filters/imag_{}.csv'.format(i)).values).data

C = 3
    
phi = lp[:, None].repeat(C,1,1,1)
psi_real = hp_real[:, None].repeat(C,1,1,1)
psi_imag = hp_imag[:, None].repeat(C,1,1,1)

# =============================================================================
# print(phi.shape)
# print(psi_real.shape)
# print(psi_imag.shape)
# =============================================================================

model = Scatt_TwoOrder(8,3,2)
model.phi.weight.data = phi
model.psi_real.weight.data = psi_real
model.psi_imag.weight.data = psi_imag
print("=====================================")

y = model(x)
for i in range(243):
    name = "trump/2/{}.png".format(i)
    torchvision.utils.save_image(y[0,i:i+1], name)
    plt.imshow(y[0,i].detach().cpu().numpy())    
    plt.show()
