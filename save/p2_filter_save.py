from libscat import *
from p2_model import PriorNet_Complex

model = PriorNet_Complex()
model.to("cuda:0")
model_name = model.__class__.__name__
dsName = 'random'
PATH = './model/{}_{}.pth'.format(model_name, dsName)

model.load_state_dict(torch.load(PATH))
model.eval()

path_lowpass = 'filters/lowpass_{}.pt'.format(dsName)
path_real = 'filters/real_{}.pt'.format(dsName)
path_imagine = 'filters/imagine_{}.pt'.format(dsName)

lowpass = model.lowpass.weight.detach().cpu().data
real = model.real.weight.detach().cpu().data
imagine = model.imag.weight.detach().cpu().data

torch.save(lowpass, path_lowpass)
torch.save(real, path_real)
torch.save(imagine, path_imagine)

