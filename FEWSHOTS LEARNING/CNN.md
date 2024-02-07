2 computer vision problems:
- Image classification
- Object detection
challenges in computer vision is that the inputs can become really big


Convolution operation are the fundamental building block of the CNN:
- Edge detection example
- 
![[Pasted image 20240116212854.png]]
<mark style="background: #BBFABBA6;">in vertical edge detection the bright pixels are on the left and the darker pixels are on the right with this we could be able to detect the vertical edge</mark>


# Difference between positive and negative edges:
- the difference between light to dark versus dark to light edge transitions.
- ![[Pasted image 20240117105530.png]]
- the negative 30 shows that the transition is from dark to light rather than light to dark transition
# horizontal edge detection
two edges are discussed brighter and the darker edge:
brighter means positive number after the convolution layer
![[Pasted image 20240117110527.png]]
<mark style="background: #FFF3A3A6;">in this -30 depicts strong darker region </mark>
<mark style="background: #ABF7F7A6;">and the 30 depicts strong lighter region </mark>
<mark style="background: #D2B3FFA6;">and the 10 depicts the transition of the pixels in that region</mark>

![[Pasted image 20240117111404.png]]

sobel filter :
gives little bit more weight to the center pixel

# <mark style="background: #ABF7F7A6;"> padding</mark>

- every time when u do the convolution operator the output shrinks to avoid that we use padding
- to give more importance to the corner pixels we use this 
- why we don't want the image to shrink is that the size will be very small in the output after shrinking
- # valid and same convolutions
- valid ->no padding
- same->pad so that the output size is same as the input size
- ![[Pasted image 20240117112215.png]]
- f is usually odd
- when the f is even then we use asymmentric padding
# Strided convolutions
- ![[Pasted image 20240117112942.png]]
# convolutions over a volumes:
On RGB images
![[Pasted image 20240117114604.png]]
![[Pasted image 20240117114725.png]]
![[Pasted image 20240117114848.png]]

# Applying ReLU Activation:

- After the convolution operation, we apply the ReLU activation function to the output feature map.
- The ReLU function is defined as `ReLU(x) = max(0, x)`. It retains positive values as they are and sets negative values to zero.
- This step introduces non-linearity. Without it, no matter how many convolution layers we stack, the overall operation of the network would remain linear, limiting its ability to capture complex patterns
- ![[Pasted image 20240117131734.png]]
- ![[Pasted image 20240117152016.png]]
- ![[Pasted image 20240119001424.png]]
- ![[Pasted image 20240119001455.png]]

# transposed convolution layer
- ![[Pasted image 20240119003154.png]]
- Look for a dataset medical image segmentation for few shot learning scenario
- run this code for few shot learning
- learn more about cnn
- iou 



How vectorisation can be used for calculating the logistic regression:
![[Pasted image 20240128060103.png]]
vectorisation that is used for calculating the backward propagation



# broadcasting is used for running python code faster
- ![[Pasted image 20240128062905.png]]
- ![[Pasted image 20240128064229.png]]
- Inorder to make sure of the dimesnion of the vector we use `assert (a.shape==(1,2))`
- 