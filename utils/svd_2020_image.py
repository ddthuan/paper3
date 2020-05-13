from libcore import *
import random

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

k = 10
n_comp = 3
n_random = random.sample(range(1,k), n_comp)
print(n_random)


U_img, S_img, V_img = torch.svd_lowrank(x, q=k)
Vt_img = torch.transpose(V_img,2,3)

img_res = torch.zeros(B,C,N1,N2)

for i in range(k):
    sigmai = S_img[:,:,i]
    #comp = S_img[:,:,i] * (U_img[:,:,:,i:i+1] @ Vt_img[:,:,i:i+1,:])
    comp = (U_img[:,:,:,i:i+1] @ Vt_img[:,:,i:i+1,:])
    for j in range(3):
        comp[:,j] = sigmai[:,j] * comp[:,j]
                
    img_res += comp
    
    #if i%10 == 0:
    torchvision.utils.save_image(comp[0].detach(), "comp/comp_{}.jpg".format(i))    
    torchvision.utils.save_image(img_res[0].detach(), "comp/rest_{}.jpg".format(i))    
    
    anh = Fv.to_pil_image(comp[0].detach())
    anh_res = Fv.to_pil_image(img_res[0].detach())
    
    plt.imshow(anh)
    plt.show()
    plt.imshow(anh_res)
    plt.show()