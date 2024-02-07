
![[Pasted image 20240121211658.png]]

# why use machine learning or deep learning
-  For a complex problem can u think of all the rules?
- If u can build a <span style="color:#00b0f0">s</span><span style="color:#00b0f0">imple rule based</span> system that doesn't require ML , do that ....
# why deep learning is good for

- <span style="color:#2d0fc2">where the problem has long list of rules</span> ->  where the traditional approach fails and the machine learning or deep learning might be useful
- <span style="color:#00b0f0">Continually changing environment</span> -> Deep learning can adapt to new scenarios
- <span style="color:#00b0f0">Insights with large collection of datase</span><span style="color:#00b0f0">t</span>-> can u image handcrafting rules for 101 different dishes
# why deep learning is not good for
- <span style="color:#00b0f0">when u need explainability </span>-> the patterns learned by deep learning models are typically uninterpreted by humans
- <span style="color:#ffff00">when the traditional approach is better option</span> ->you can accomplish what u need with a simple rule based system.
- <span style="color:#16e924">when errors are unacceptable</span> -> Since the output of the DL model aren't always unpredictable
- <span style="color:#16e924">When you don't have that much of data</span> -> DL models require fairly large amount of data to produce great results
# ML vs DL
- where to use the ML ->over the<span style="color:#bf32d2"> structured data </span>Ml  algorithms has to be applied
- DL -> it is used for <span style="color:#16e924">unstructured data</span> 
![[Pasted image 20240121223033.png]]
# what are neural networks
![[Pasted image 20240121223521.png]]
![[Pasted image 20240121223804.png]]
<span style="color:#00b0f0">Each layer is usually a combination of linear and /or non linear function.</span> 

# what is deeplearning actually used for
- ![[Pasted image 20240121224454.png]]
- ![[Pasted image 20240121230355.png]]
# PyTorch
 -  what is pyTorch
 - most popular deep learning framework
 - write fast DL  code in python
 - able to access pre-built deep learning models
 - whole stack: 
	 - preprocessing data,
	 - model data
	 - deploy data in ut cloud or application
- originally used in house of facebook,meta
- TPU-> what is tensors
- Tensors are any representation of numbers
- ![[Pasted image 20240121232133.png]]
- ![[Pasted image 20240121233830.png]]
- ![[Pasted image 20240121234158.png]]
- ![[Pasted image 20240121235851.png]]
- ![[Pasted image 20240122000006.png]]
- Tensors datatypes is one of the 3 big errors you 'll run into with pytorch and deep learning:
	- tensors not rgt datatype
	- tensors not rgt shape
	- tensors not on the rgt device-> if u trying to do operation for 2 tensors at that time one tensor live on the gpu and the other lives on the cpu at that time hard for us to do the operation so that time it wont be working out
	- 