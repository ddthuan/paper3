<h1>Paper:</h1>
<h2>Build the noise, shift, deformation stable filter group based on the reverse engineering with scattering transforms</h2>
<h2><a href="https://docs.google.com/presentation/d/e/2PACX-1vQx2u1m8VT05y0rpVv1l0akFa_2HxtGkU6O54JayVceKY0TC8MTlp_xe7f2WgfwDZP5nAeVzmeeSt2H/pub?start=false&loop=false&delayms=3000">slide</a></h2>
<br>
<br>

<h1>1. Motivation</h1>
<img src = "https://github.com/ddthuan/paper3/blob/master/image/results/motivation.png" /><br><br>

<img src="https://github.com/ddthuan/paper3/blob/master/image/results/solution.png" /><br>

<h1>2. Method</h1>
<br>
<img src="https://github.com/ddthuan/paper3/blob/master/image/results/propose_diagram.png" /><br><br>

<h2>Complex Modulus CNN architecture</h2>
<img src="https://github.com/ddthuan/paper3/blob/master/image/results/propose_model.png" /><br><br>


<h1 style="color:green">3. Results</h1><br>
<h2>Regular CNN Model and Complex Modulus CNN Model </h2>
<img src="https://github.com/ddthuan/paper3/blob/master/image/results/vsModel.png" /></a><br>

<h2>Size of Filter</h2>
<img src = "https://github.com/ddthuan/paper3/blob/master/image/results/kernelSize.png" /></a><br>
<br>

<h2>The filter group is stable to noise, shift, deformation of signal</h2><br>
<h2>Random Dataset (Order 1), size [:,1,512,128]</h2>
<h3>Phi</h3>
<img src="https://github.com/ddthuan/paper3/blob/master/image/random_phi.png" />
<h3>Psi: real part</h3>
<img src="https://github.com/ddthuan/paper3/blob/master/image/random_psi_real.png" />
<h3>Psi: imaginary part</h3>
<img src="https://github.com/ddthuan/paper3/blob/master/image/random_psi_imag.png" /></br>

<h2>CIFAR10 Dataset (Order 1), size [:,1,32,32]</h2>
<a href="https://github.com/ddthuan/paper3/blob/master/test/csv/order1_cifar10_phi.csv"><img src="https://github.com/ddthuan/paper3/blob/master/image/cifar_phi.png" /></a></br>
<img src="https://github.com/ddthuan/paper3/blob/master/image/cifar_psi_real.png" /></br>
<img src="https://github.com/ddthuan/paper3/blob/master/image/cifar_psi_imag.png" /></br>

<h2>Tiny Imagenet (Order 1), size [:,1:2,64,64]</h2>
<a href="https://github.com/ddthuan/paper3/blob/master/test/csv/order1_imagenet_phi.csv"><img src="https://github.com/ddthuan/paper3/blob/master/image/restnet_phi.png" /></a></br>
<img src="https://github.com/ddthuan/paper3/blob/master/image/restnet_psi_real.png" /></br>
<img src="https://github.com/ddthuan/paper3/blob/master/image/restnet_psi_imag.png" /></br>

<h2>Tiny Imagenet (Order 2, Smooth), size [:,1:2,64,64]</h2>
<a href="https://github.com/ddthuan/paper3/blob/master/test/csv/order2_imagenet_smooth_phi.csv"><img src="https://github.com/ddthuan/paper3/blob/master/image/order2/imagenet_phi.png" /></a></br>
<img src="https://github.com/ddthuan/paper3/blob/master/image/order2/imagenet_psi_real.png" /></br>
<img src="https://github.com/ddthuan/paper3/blob/master/image/order2/imagenet_psi_imag.png" /></br>

<h2>Tiny Imagenet (Order 2 -NoneSmooth), size [:,1:2,64,64]</h2>
<a href="https://github.com/ddthuan/paper3/blob/master/test/csv/order2_imagenet_nonesmooth_phi.csv"><img src="https://github.com/ddthuan/paper3/blob/master/image/order2/imagenetNone_phi.png" /></a></br>
<img src="https://github.com/ddthuan/paper3/blob/master/image/order2/imagenetNone_psi_real.png" /></br>
<img src="https://github.com/ddthuan/paper3/blob/master/image/order2/imagenetNone_psi_imag.png" /></br>


<h2>Measure Stability of Noise, Shift, Deformation</h2>
<img src="https://github.com/ddthuan/paper3/blob/master/image/results/stability.png" /></br>

<h2>Speed</h2>
<img src = "https://github.com/ddthuan/paper3/blob/master/image/results/speed.png" /><br>
