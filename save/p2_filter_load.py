from libcore import *
import pandas as pd

torch.set_printoptions(precision=4)

path_lowpass = 'filters/lowpass.pt'
path_real = 'filters/real.pt'
path_imagine = 'filters/imagine.pt'

lowpass = torch.load(path_lowpass)
real = torch.load(path_real)
imagine = torch.load(path_imagine)

# =============================================================================
# print(lowpass.shape)
# print(real.shape)
# print(imagine.shape)
# 
# imshow_actual_size(lowpass[0,0].numpy(), 'filters/lowpass.png')
# for i in range(8):
#     imshow_actual_size(real[i, 0].numpy(), 'filters/real_{}'.format(i))
#     imshow_actual_size(imagine[i, 0].numpy(), 'filters/imag_{}'.format(i))
# =============================================================================


# rounded filters with 3 dicimal number
n_digits = 4
lowpass_rounded = (lowpass[0,0] * 10**n_digits).round() / (10**n_digits)
real_rounded = (real * 10**n_digits).round() / (10**n_digits)
imagine_rounded = (imagine * 10**n_digits).round() / (10**n_digits)
print(lowpass_rounded)

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
    
# =============================================================================
# for k in range(8):
#     save_filters(real_rounded[k,0], 'filters/real_{}.csv'.format(k))
#     save_filters(imagine_rounded[k,0], 'filters/imag_{}.csv'.format(k))
# =============================================================================
    
    
# =============================================================================
# path_filter_lowpass = 'filters/filter_lowpass.csv'
# 
# col1 = []
# col2 = []
# col3 = []
# col4 = []
# col5 = []
# col6 = []
# col7 = []
# for i in range(7):
#     col1.append(lowpass_rounded[i, 0].numpy())
#     col2.append(lowpass_rounded[i, 1].numpy())
#     col3.append(lowpass_rounded[i, 2].numpy())
#     col4.append(lowpass_rounded[i, 3].numpy())
#     col5.append(lowpass_rounded[i, 4].numpy())
#     col6.append(lowpass_rounded[i, 5].numpy())
#     col7.append(lowpass_rounded[i, 6].numpy())
#     
# df=pd.DataFrame()
# df['col1'] = col1
# df['col2'] = col2
# df['col3'] = col3
# df['col4'] = col4
# df['col5'] = col5
# df['col6'] = col6
# df['col7'] = col7
# df.to_csv(path_filter_lowpass,index=False)
# =============================================================================
