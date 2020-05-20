from libcore import *
import pandas as pd

torch.set_printoptions(precision=5)

def save_filters(ts_value, ts_path):
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []

    for i in range(7):
        col1.append(ts_value[i, 0].numpy())
        col2.append(ts_value[i, 1].numpy())
        col3.append(ts_value[i, 2].numpy())
        col4.append(ts_value[i, 3].numpy())
        col5.append(ts_value[i, 4].numpy())
        col6.append(ts_value[i, 5].numpy())
        col7.append(ts_value[i, 6].numpy())

    df=pd.DataFrame()
    df['col1'] = col1
    df['col2'] = col2
    df['col3'] = col3
    df['col4'] = col4
    df['col5'] = col5
    df['col6'] = col6
    df['col7'] = col7
    df.to_csv(ts_path, index=False)

dsName = ['random', 'cifar10', 'tinyimage']
n_digits = 5

for ds in dsName:
    
    path_lowpass = 'filters/lowpass_{}.pt'.format(ds)
    path_real = 'filters/real_{}.pt'.format(ds)
    path_imagine = 'filters/imagine_{}.pt'.format(ds)
    
    lowpass = torch.load(path_lowpass)
    real = torch.load(path_real)
    imagine = torch.load(path_imagine)
    
    # rounded filters with 5 dicimal number
    lowpass_rounded = (lowpass[0,0] * 10**n_digits).round() / (10**n_digits)
    real_rounded = (real * 10**n_digits).round() / (10**n_digits)
    imagine_rounded = (imagine * 10**n_digits).round() / (10**n_digits)
    
    # export lowpass to CSV file
    save_filters(lowpass_rounded, 'filters/filter_lowpass_{}.csv'.format(ds))
    
    # export real and imaginary part to CSV file
    for k in range(8):
        save_filters(real_rounded[k,0], 'filters/real_{}_{}.csv'.format(ds, k))
        save_filters(imagine_rounded[k,0], 'filters/imag_{}_{}.csv'.format(ds, k))
    
    
