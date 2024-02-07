- [ ] apeer.com for taking a file and giving the segemnted images
gradient and metric
- forward and backward pass
   - [[differernce between f and b pass]]
- metric
    - prototypical learning
- few shots learning how does it  work
   - <mark style="background: #FFB8EBA6;">Most few-shot classification methods are _metric-based_. It works in two phases : </mark>
   1) they use a CNN to project both support and query images into a feature space, and
   2) they classify query images by comparing them to support images. 
- <mark style="background: #ABF7F7A6;">what is Segmentation</mark>
    - In medical image processing, segmentation plays a critical role in helping doctors and researchers analyze medical images such as X-rays, MRI scans, or CT scans. There are several types of segmentation methods used, each with its own approach to dividing an image into meaningful parts. Here's a simple explanation of the main types:

1. **Thresholding**: This is like using a <mark style="background: #BBFABBA6;">filter to separate light and dark areas in a photo</mark>. In medical images, this method sets a specific value (threshold), and every part of the image that is lighter or darker than this value is marked. For example, it can help highlight bones in an X-ray by filtering out the softer tissues.
    
2. **Region-Based Segmentation**: Imagine if you had a puzzle where each piece was a different area of a picture. Region-based segmentation works like that. It starts with a <mark style="background: #FF5582A6;">part of the image and keeps adding nearby parts that are similar</mark> until it has a complete 'puzzle piece' or region. This is useful for identifying specific areas like a tumor in an MRI scan.
    
3. **Edge Detection**: This method works like tracing the outlines in a coloring book. The computer looks for <mark style="background: #ADCCFFA6;">the edges or lines that separate different parts of the image</mark>. For instance, it can outline the edge of an organ to help doctors see its shape and size more clearly.
    
4. **Clustering**: Clustering is like <mark style="background: #D2B3FFA6;">sorting candies by color</mark>. The computer groups together parts of the image that look similar. This method doesn’t need specific rules about what to look for; it automatically finds patterns in the image, like different tissues in a scan.
    
5. **Model-Based Segmentation**: This is like using a cookie cutter to make cookies in certain shapes. In model-based segmentation, <mark style="background: #FFF3A3A6;">the computer uses a predefined model (like the typical shape of a heart) to find and segment parts of the image</mark>. This is especially helpful when looking for organs that usually have a standard shape.
    
6. **Atlas-Based Segmentation**: Imagine having a map that shows you where everything should be in a city. Atlas-based segmentation uses a similar 'map' or 'atlas' of the human body to guide the segmentation. <mark style="background: #FFB86CA6;">It matches parts of the medical image with this atlas to identify different structures.</mark>
    

### How Segmentation is Done in Medical Imaging:

1. **Image Acquisition**: First, a medical image is taken using technologies like MRI, CT, or X-rays.
    
2. **Preprocessing**: The image might be cleaned up to remove noise or enhance contrast, making it easier to segment.
    
3. **Segmentation**: One or more of the segmentation methods are applied to the image to identify and isolate different structures like organs, bones, or anomalies.
    
4. **Post-processing**: Sometimes, the segmented areas are refined to improve accuracy, like smoothing the edges or removing small, irrelevant spots.
    
5. **Analysis and Interpretation**: Finally, doctors or medical professionals analyze the segmented image for diagnosis, treatment planning, or research.
    

Segmentation in medical imaging helps in precise measurement, analysis, and visualization of different anatomical structures and is a vital tool in modern medicine.


 =>
- how do humans do segmentation 
- k means clustering






=> How humans do segmenation:
- proximity principle
- similarity principle
- Common fate




=>Similarity principle

![[Pasted image 20231223064205.png]]

    Similar objects are grouped together



=>Common Fate

![[Pasted image 20231223064338.png]]

objects with similar motion or change in appearance are grouped together


=>Common Region/connectivity

![[Pasted image 20231223064511.png]]



=>Continuity Principle


![[Pasted image 20231223064707.png]]


=>Symmentry principle

![[Pasted image 20231223065101.png]]

parallel and symmentrical feaetures are grouped together


=>Illusory Contours
![[Pasted image 20231223142823.png]]


illusory ot subject contours are perceived




=>Segementation strategies are highly subjective

- Segementation is intuitive for us .
- very hard to translate these intuitions to an algorithms

     - Top-down Segmen






=>Semantic segemntation


![[Pasted image 20231223161442.png]]




why callbacks are used :
they used to perform actions at various stages of the training process
<mark style="background: #FFF3A3A6;">Monitoring training:</mark>
callbacks can be used to monitor the performance of the model during training .for instance ,by measuring metrics such as loss and accuracy after each epoch
<mark style="background: #BBFABBA6;">Early stopping:</mark>
to prevent overfitting,a callback can halt the training process ear;y if there is no improvement in the model's performance on a validation set for a defined number of epoches
early stopping is used to find how many epoches to be used to train a model 



2018 data science bowl
we used to count the nuclei and then segment this
