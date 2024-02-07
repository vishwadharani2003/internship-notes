
![[2003.08462v2.pdf]]

<mark style="background: #FFB8EBA6;">Localization </mark>in the context of the highlighted text refers to the process of identifying the exact location of something within an image. Unlike classification tasks where the output is a single class label for the whole image, localization assigns a class label to each individual pixel. This means that the network's output isn't just "<mark style="background: #ADCCFFA6;">what" is in the image, but also "where"</mark> it is precisely. In biomedical image processing, this is crucial for tasks like identifying the boundaries of a tumor or other structures within medical scans, where the exact shape and size are important for diagnosis and treatment planning.



![[Pasted image 20231223175243.png]]


![[Pasted image 20231225163950.png]]


![[Pasted image 20231225164132.png]]


# U-Net is defined for a semantic segmenation

-> y they came with U-net is biomedical segmentation

=>overlap tile strategy for seamless segmenatation


The overlap-tile strategy referred to in the U-Net paper is a technique for the seamless <mark style="background: #D2B3FFA6;">segmentation of arbitrarily large images</mark>, which may be too large to fit into the GPU memory if processed whole. Here's what the strategy entails:

1. **Segmentation of Larger Images:**
    
    - The U-Net is designed to segment large images by <mark style="background: #ABF7F7A6;">processing smaller tiles</mark> of the image rather than the entire image at once.
2. **Overlap of Tiles:**
    
    - When these tiles are processed, they are overlapped such that the prediction for a particular area in the image (yellow area in the paper's figure) requires data from a larger context (blue area in the figure).
    - <mark style="background: #ABF7F7A6;">This means that each tile is segmented with some of its surrounding context, not just the tile alone.</mark>
3. **Extrapolation of Missing Input:**
    
    - For tiles on the border of the image, where there might not be enough surrounding context, the <mark style="background: #FFF3A3A6;">missing input data is extrapolated by mirroring the image data</mark>.
4. **Learning from Expanded Context:**
    
    - The network learns to use this additional context to assemble a more precise output through successive convolutional layers.
5. **Feature Channels in Upsampling:**
    
    - An important modification in U-Net is that during the upsampling (or decoding) part of the network, <mark style="background: #D2B3FFA6;">a large number of feature channels are used</mark>, allowing the network to propagate context information back to higher resolution layers.
6. **Symmetry and U-Shape:**
    
    - The expansive path of the U-Net is designed to be more or less symmetric to the contracting path, contributing to its characteristic U-shape.
7. **Use of Valid Convolution:**
    
    - The network architecture does not include any fully connected layers and relies only on the valid part of each convolution. This means the segmentation map is constructed only from the pixels for which the full context is available in the input image.
8. **Seamless Tiling for Segmentation:**
    
    - By applying this strategy, U-Net can segment very large images seamlessly, as the overlap ensures that the transitions between tiles are smooth and continuous, with no visible seams or discontinuities in the segmented output

- Y the unet output class has 2 classes because it need seperate foreground and background
     - ![[Pasted image 20231226000959.png]]
     - ![[Pasted image 20231226001057.png]]
     - The image describes a 3x3 convolution operation followed by a Rectified Linear Unit (ReLU) activation, which are common components in Convolutional Neural Networks (CNNs).

Here's a detailed explanation of each part:

1. **Input Feature Map (a):**
    
    - This is the input data to the convolution layer. The example shows an input feature map with 5 channels (depth of 5). You can think of this as a 3D volume where the height and width are spatial dimensions and the channels represent different features or filters applied from the previous layer.
2. **3x3 Convolution:**
    
    - The convolution operation involves a filter or kernel (in this case, 3x3 in size) that slides over the input feature map. For each position, it performs an element-wise multiplication with the part of the input feature map it covers.
    - The filter has weights (w), which are learned during the training process. In this example, one such weight is indicated as wi,j,k,lwi,j,k,l​, where i,ji,j are the spatial indices of the kernel, kk is the input channel, and ll is the specific filter out of the 45 mentioned.
3. **Bias (c):**
    
    - After the convolution operation, a bias term (c) is added. This is also a learnable parameter and is added to the sum of the weighted inputs to introduce an additional degree of freedom to the neural network model.
4. **Output Feature Map (b):**
    
    - The result of the convolution and bias addition for each position of the filter is a single value. When the filter slides across the entire input feature map, an entire 2D feature map is produced (this is just one channel of the output feature map).
5. **ReLU Activation:**
    
    - The ReLU function is applied to each value in the output feature map. ReLU is defined as ReLU(x)=max⁡(0,x)ReLU(x)=max(0,x). This means if the input is positive, the output is the same as the input; if the input is negative, the output is 0. It introduces non-linearity into the model, allowing it to learn more complex functions.
6. **Convolution Formula:**
    
    - The formula shown defines the calculation for one value in one channel of the output feature map (b). It sums over all positions i,ji,j in the 3x3 filter, and over all input feature maps kk of the input aa, applying the corresponding weight ww and adding the bias cc. The ReLU function is then applied to this sum.

**Example**: Let's say you have an input feature map of size 5x5x5 (width x height x channels). You apply a 3x3 convolution with 45 filters, meaning you will have 45 different 3x3 kernels each creating its own 2D feature map. For simplicity, let's focus on a single filter (one of the 45):

- You place the filter over the top-left corner of the input feature map so that it covers a 3x3x5 portion of the input.
- You multiply each of the elements of this 3x3x5 block with the corresponding weight of the filter and sum all these products together.
- You add the bias term for this filter to the sum.
- You apply the ReLU activation function to the result, which gives you one value of the new feature map.
- You slide the filter over to the next position and repeat the process until you've covered the entire input feature map, resulting in a 2D feature map of the output (for that single filter).
- Since there are 45 filters, you will end up with an output feature map of dimensions height x width x 45 (assuming no padding and stride of 1).

This process is fundamental to CNNs as it allows the network to learn filters that activate when they see some specific type of feature at some spatial position in the input.

ChatGPT can make mistakes. Consider checking important information.



# when to stop down sampling

In a U-Net architecture, the transition from downsampling to upsampling is typically predefined and is part of the network's structure. However, the decision on when to stop downsampling and start upsampling involves considering several factors:

1. **Input Image Size**:
    
    - The original size of your input images will often dictate the number of downsampling steps. You want to <mark style="background: #ADCCFFA6;">downsample until you reach a feature map that is small enough to efficiently capture the high-level context without losing too much spatial resolution.</mark>
2. **Network Depth**:
    
    - The depth of the network is a design choice. Deeper networks can capture more complex features but also require more computational resources. The number of downsampling (and corresponding upsampling) steps typically determines the depth.
3. **Receptive Field**:
    
    - Each downsampling step increases the<mark style="background: #ABF7F7A6;"> receptive field of the network</mark>, <mark style="background: #FFB8EBA6;">meaning the area of the input image that each pixel in the feature map represents</mark>. You want a sufficiently large receptive field to capture relevant patterns and context.
4. **Computational Constraints**:
    
    - More downsampling steps mean more parameters and potentially deeper networks, which require <mark style="background: #ADCCFFA6;">more memory and computational powe</mark>r. The limitations of your hardware might also influence the maximum depth of your network.
5. **Model Performance**:
    
    - Empirically, you might find that adding more downsampling steps does not improve or might even degrade the performance on your validation set. This could be a sign that you need to balance between the depth of the network and the ability to preserve spatial details.
6. **Data Characteristics**:
    
    - For certain datasets or segmentation tasks, finer details might be necessary, and <mark style="background: #BBFABBA6;">too much downsampling could lead to loss of such details</mark>. In such cases, fewer downsampling steps may be preferable.

In practice, for a U-Net, you typically define the architecture to downsample until you reach a bottleneck layer, which has a small spatial dimension but a large number of feature channels. <mark style="background: #ADCCFFA6;">This is usually a power of 2 that is a fraction of the input image size, like 32x32 or 64x64 for an input image of 512x512 pixels.</mark> After reaching this bottleneck, you start the upsampling process to gradually recover the spatial dimensions while reducing the feature channels, often using transposed convolutions or upscaling operations.

In a standard U-Net implementation, the number of downsampling and upsampling steps is symmetric. If you have 4 downsampling steps (each reducing the spatial dimensions by half), you would also have 4 upsampling steps (each doubling the spatial dimensions) to return to the original image size.



The provided text describes the training procedure for a neural network, specifically focusing on how the input data is used, the structure of the network's output, and the loss function utilized to train the network. Let’s break down each part:

1. **Training Data and Process**:
    
    - The network is trained using pairs of input images and their corresponding segmentation maps.
    - Stochastic gradient descent (SGD) is the optimization algorithm used, which is implemented in Caffe, a deep learning framework.
    - Due to unpadded convolutions in the network, the output image (segmentation map) is smaller than the input image by a constant border width. Unpadded convolutions mean that the edges of the input are not padded with zeros, leading to a reduction in size after each convolution.
2. **Optimization Memory Management**:
    
    - To efficiently use GPU memory and minimize overhead, they prefer to use large input tiles and a large batch size.
    - However, to cope with the large input tiles, they reduce the batch size to a single image, meaning that each batch in the training process consists of only one image and its segmentation map.
3. **High Momentum**:
    
    - A high momentum term of 0.99 is used in SGD. Momentum helps accelerate SGD in the relevant direction and dampens oscillations. It does this by incorporating a fraction of the update from the previous step to the current step. A high momentum means that updates are heavily influenced by past gradients.
4. **Energy Function (Loss Function)**:
    
    - The loss function is a combination of a pixel-wise soft-max function and the cross-entropy loss.
    - Soft-max function is applied to the final feature map for each pixel, transforming the raw scores (logits) from the network into probabilities for each class. The soft-max function is defined for each class kk and pixel position xx by the formula provided, which normalizes the exponentiated logits by the sum of exponentiated logits for all classes.
5. **Cross-Entropy Loss**:
    
    - Cross-entropy loss is used to measure the difference between the predicted probabilities (from the soft-max function) and the true distribution (actual labels in the segmentation map).
    - It penalizes the deviation of the predicted probability pℓ(x)(x)pℓ(x)​(x) for the true class ℓ(x)ℓ(x) from 1. The ideal prediction would have a probability of 1 for the correct class and 0 for all others.
    - The cross-entropy loss is weighted by a weight function w(x)w(x) for each pixel to give different importance to different pixels, which is not further explained in the provided text but is typical in segmentation tasks to handle class imbalance or to emphasize certain areas like boundaries between classes.
6. **Equation (1)**:
    
    - The overall loss EE for the network is the sum of the weighted cross-entropy loss across all pixels in the training set.
    - This loss is what the training process aims to minimize by adjusting the network's weights through backpropagation.

In summary, the network is trained image by image, using a combination of soft-max for probability prediction and cross-entropy for loss calculation, with a high momentum to integrate past gradient information. The network's output size is smaller than the input due to unpadded convolutions, and the loss is minimized during training to achieve the best segmentation performance.




The text provides detailed information about the training strategy for a neural network model, specifically focusing on how it handles the segmentation of biomedical images, such as HeLa cells. Here's the breakdown in points:

1. **Ground Truth Segmentation**:
    
    - The raw input images (a) are recorded with differential interference contrast (DIC) microscopy.
    - The ground truth segmentation (b) is overlaid on the raw image, with different colors indicating different individual cells.
2. **Generated Segmentation Mask**:
    
    - The neural network generates a segmentation mask (c), where white represents the foreground (cells) and black represents the background.
3. **Pixel-wise Loss Weighting**:
    
    - A pixel-wise loss weight map (d) is created to emphasize learning the borders between touching cells.
4. **Weight Map Function**:
    
    - The function w(x)w(x) assigns a weight to each pixel, giving more importance to certain pixels during training. This helps balance the frequency of pixel classes and emphasizes learning the separation borders between cells.
5. **Pre-computation of the Weight Map**:
    
    - For each ground truth segmentation, a weight map is pre-computed to help the network learn the separation borders which are computed using morphological operations.
6. **Formula for Weight Map**:
    
    - The weight map is calculated using the formula w(x)=wc(x)+w0⋅exp⁡(−(d1(x)+d2(x))22σ2)w(x)=wc(x)+w0​⋅exp(−2σ2(d1​(x)+d2​(x))2​).
    - Here, wc(x)wc(x) balances class frequencies, d1(x)d1​(x) is the distance to the nearest cell border, and d2(x)d2​(x) is the distance to the second nearest cell border.
    - The parameters w0w0​ and σσ are set in the experiments to specific values that influence the weight given to separation borders.
7. **Parameters for Weight Map**:
    
    - In the experiments, they set w0=10w0​=10 and σσ to approximately 5 pixels. This means the weight map gives ten times more importance to the pixels close to the cell borders within a range defined by σσ.
8. **Importance of Weight Initialization**:
    
    - Proper weight initialization is crucial for deep networks to prevent parts of the network from having excessive activations while others do not contribute at all.
9. **Initialization Strategy**:
    
    - Initial weights are drawn from a Gaussian distribution with a standard deviation of 2/N2/N

1. - ​, where NN is the number of incoming nodes to a neuron.
    - For example, with a 3x3 convolution and 64 feature channels in the previous layer, NN would be 9×64=5769×64=576. This ensures that the feature maps have approximately unit variance, which is important for network stability and performance.

In essence, this approach aims to improve the segmentation performance of the network by focusing on learning the challenging parts of the images—namely, the borders between touching cells—using a sophisticated weighting strategy in the loss function and a careful initialization of the network weights.


![[Pasted image 20231226131510.png]]


# outputs of CNN
![[Pasted image 20240121103213.png]]


# dropout layers
when we have more number of weights and biases -> that could cause over fitting ->for multi layer neural network
# Two ways to solve the over fitting problem

- Regularization
- Dropout
# Dropout layers
<span style="color:#00b0f0">The key idea behind dropout is to prevent the network from becoming too reliant on any single neuron (or a specific arrangement of neurons)</span>, promoting redundancy and forcing the network to learn more robust features that generalize better to unseen data. By "dropping out" random neurons


When you use dropout in a neural network,<span style="color:#00b0f0"> it does not directly reduce the number of trainable parameters</span>; all the weights remain, and their total number is unchanged. However, it affects the training process by randomly setting a fraction of the neurons' outputs to zero during training. Let's dive deeper into how increasing dropout rates affect the training process, particularly for deeper layers:

**Learning Complex Features**: In deeper layers of a neural network, the features being learned are generally more complex and abstract compared to the simpler, more general features in earlier layers. <span style="color:#e100ff">By increasing the dropout rate (e.g., from 0.1 to 0.3), you're increasing the probability that any given neuron will be "dropped" or temporarily removed from the network during a training iteration.</span> 
**Regularization Effect**: A higher dropout rate means that with each forward pass during training,<span style="color:#16e924"> a larger number of neurons are turned off.</span> This forces the network to not rely too heavily on any single neuron (or a specific configuration of neurons) and to spread out the "importance" across many neurons. This can lead to a more robust model that generalizes better but can also slow down the learning process or lead to a model that is underfit if the dropout rate is too high.


# Dropout layers Example
![[Pasted image 20240131162426.png]]


# Kernel_intializer=he_normal
### Function: `np.random.normal`

This function generates random numbers from a normal (Gaussian) distribution. The normal distribution is a probability distribution that is symmetric about the mean, <span style="color:#16e924">showing that data near the mean are more frequent in occurrence than data far from the mean.</span>
- **Calculate Standard Deviation for `he_normal`**:
    
    - The `he_normal` initializer sets the standard deviation (`stddev`) of the distribution to `sqrt(2 / fan_in)`.
    - With `fan_in = 9`, `stddev = sqrt(2 / 9)`.
    - **Generate Random Numbers**:
    
    - Random numbers are generated from a normal (Gaussian) distribution with a mean of 0 and the calculated standard deviation. This is where the "random" aspect comes into play.
    - In practice, this is done using a random number generator that can produce values from a normal distribution. Libraries like NumPy in Python have functions for this, such as `numpy.random.normal`.
    - **Apply Truncation (if needed)**:
    
    - The term "he_normal" implies a normal distribution, but in practice, some implementations might truncate extreme values to prevent very large weights. This step is optional and depends on the specific implementation in the library you're using.
```import numpy as np

# Parameters
fan_in = 9
stddev = np.sqrt(2 / fan_in)

# Generate weights for a 3x3 filter
weights = np.random.normal(0, stddev, (3, 3))

print(weights)

```

# IOU calculation
```python
# Ground truth mask
y_true = np.array([[0, 1, 1],
                   [0, 0, 1],
                   [1, 1, 0]])

# Predicted mask
y_pred = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]])
```
```python
intersection = np.logical_and(y_true, y_pred)
# Resulting intersection mask
# [[0, 1, 0],
#  [0, 0, 1],
#  [1, 0, 0]]
```
```python
union = np.logical_or(y_true, y_pred)
# Resulting union mask
# [[0, 1, 1],
#  [0, 0, 1],
#  [1, 1, 0]]
```
```python
iou_score = np.sum(intersection) / np.sum(union)
# IoU = (1 + 1 + 1) / (2 + 2 + 2) = 3 / 6 = 0.5
```

what paper u r implementing