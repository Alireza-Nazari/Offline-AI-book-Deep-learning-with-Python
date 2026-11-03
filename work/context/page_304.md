# Generation bundle for page_304  (PDF page 304, printed page 280)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.5 Putting it together: A mini Xception-like model
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_304.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_304.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p304_e03] (figure_caption) (continued)
[p304_e04] (sidebar) As a result, you would expect depthwise separable convolutions to be dramatically faster, right? Hold on. This would be true if you were writing simple CUDA or C++ implementations of these algorithms—in fact, you do see a meaningful speedup when running on CPU, where the underlying implementation is parallelized C++. But in practice, you’re probably using a GPU, and what you’re executing on it is far from a “simple” CUDA implementation: it’s a cuDNN kernel, a piece of code that has been extraordinarily optimized, down to each machine instruction. It certainly makes sense to spend a lot of effort optimizing this code, since cuDNN convolutions on NVIDIA hardware are responsible for many exaflops of computation every day. But a side effect of this extreme micro-optimization is that alternative approaches have little chance to compete on performance—even approaches that have significant intrinsic advantages, like depthwise separable convolutions. Despite repeated requests to NVIDIA, depthwise separable convolutions have not benefited from nearly the same level of software and hardware optimization as regular convolutions, and as a result they remain only about as fast as regular convolutions, even though they’re using quadratically fewer parameters and floating-point operations. Note, though, that using depthwise separable convolutions remains a good idea even if it does not result in a speedup: their lower parameter count means that you are less at risk of overfitting, and their assumption that channels should be uncorrelated leads to faster model convergence and more robust representations. What is a slight inconvenience in this case can become an impassable wall in other situations: because the entire hardware and software ecosystem of deep learning has been micro-optimized for a very specific set of algorithms (in particular, ConvNets trained via backpropagation), there’s an extremely high cost to steering away from the beaten path. If you were to experiment with alternative algorithms, such as gradient-free optimization or spiking neural networks, the first few parallel C++ or CUDA implementations you’d come up with would be orders of magnitude slower than a good old ConvNet—no matter how clever and efficient your ideas were. Convincing other researchers to adopt your method would be a tough sell, even if it were just plain better. You could say that modern deep learning is the product of a co-evolution process between hardware, software, and algorithms: the availability of NVIDIA GPUs and CUDA led to the early success of backpropagation-trained ConvNets, which led NVIDIA to optimize its hardware and software for these algorithms, which in turn led to consolidation of the research community behind these methods. At this point, figuring out a different path would require a multiyear reengineering of the entire ecosystem.
[p304_e05] (heading_section) 9.5
[p304_e06] (heading_section) Putting it together: A mini Xception-like model
[p304_e07] (body) As a reminder, here are the ConvNet architecture principles you’ve learned so far:
[p304_e08] (bullet) Your model should be organized into repeated blocks of layers, usually made of multiple convolution layers and a max pooling layer.
[p304_e09] (bullet) The number of filters in your layers should increase as the size of the spatial feature maps decreases.

## Figures
- Figure None: (continued)  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 303) text, for continuity
Figure 9.4 Depthwise separable convolution: a depthwise convolution followed by a pointwise convolution

neural networks, it serves as a useful prior that helps the model make more efficient use of its training data. A model with stronger priors about the structure of the information it will have to process is a better model—as long as the priors are accurate.

Depthwise separable convolution requires significantly fewer parameters and involves fewer computations compared to regular convolution, while having comparable representational power. They result in smaller models that converge faster and are less prone to overfitting. These advantages become especially important when you’re training small models from scratch on limited data. When it comes to larger-scale models, depthwise separable convolutions are the basis of the Xception architecture, a high-performing ConvNet that comes packaged with Keras. You can read more about the theoretical grounding for depthwise separable convolutions and Xception in the paper “Xception: Deep Learning with Depthwise Separable Convolutions.”3

The co-evolution of hardware, software, and algorithms

Consider a regular convolution operation with a 3 x 3 window, 64 input channels, and 64 output channels. It uses 3 × 3 × 64 × 64 = 36,864 trainable parameters, and when you apply it to an image, it runs a number of floating-point operations that is proportional to this parameter count. Meanwhile, consider an equivalent depthwise separable convolution: it only involves 3 × 3 × 64 + 64 × 64 = 4,672 trainable parameters and proportionally fewer floating-point operations. This efficiency improvement only increases as the number of filters or the size of the convolution windows gets larger.

3 François Chollet, “Xception: Deep Learning with Dep …

## Next page (PDF 305) text, for continuity
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.307 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.294 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.256 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.292 — 9 ConvNet architecture patterns › 
- PDF p.295 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.12 — contents › 
- PDF p.635 — index › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.302 (printed 278) — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions › Depthwise separable convolutions
What if we told you that there’s a layer you can use as a drop-in replacement for `Conv2D` that will make your model smaller (fewer trainable weight parameters), leaner (fewer floating-point operations), and cause it to perform a few percentage points better on its task? That is precisely what the depthwise separable convolution layer does (`Separable` `Conv2D` in Keras). This layer performs a spatial convolution on each channel of its input, independently, before mixing output channels via a pointwise convolution (a 1 × 1 convolution), as shown in figure 9.4.

This is equivalent to separating the learning of spatial features and the learning of channel-wise features. In much the same way that convolution relies on the assumption that the patterns in images are not tied to specific locations, depthwise separable convolution relies on the assumption that spatial locations in intermediate activations are highly correlated, but different channels are highly independent. Because this assumption is generally true for the image representations learned by deep

### PDF p.12 (printed xii) — contents ›  › 
8.2

Training a ConvNet from scratch on a small dataset 241

The relevance of deep learning for small-data problems 242 Downloading the data 242 ■ Building your model 245 Data preprocessing 247 ■ Using data augmentation 252

8.3

Using a pretrained model 256

Feature extraction with a pretrained model 256 ■ Fine-tuning a pretrained model 264

9

ConvNet architecture patterns 268 9.1

Modularity, hierarchy, and reuse 269 9.2 Residual connections 272 9.3 Batch normalization 276 9.4 Depthwise separable convolutions 278 9.5 Putting it together: A mini Xception-like model 280 9.6 Beyond convolution: Vision Transformers 282 10 Interpreting what ConvNets learn 284 10.1 Visualizing intermediate activations 285 10.2 Visualizing ConvNet filters 291

Gradient ascent in TensorFlow 294 ■ Gradient ascent in PyTorch 295 ■ Gradient ascent in JAX 295 ■ The filter visualization loop 296

10.3

Visualizing heatmaps of class activation 299

Getting the gradient of the top class: TensorFlow version 302 Getting the gradient of the top class: PyTorch version 302 Getting the gradient of the top class: JAX version 303 Displaying the class activation heatmap 304

10.4 Visualizing the latent space of a ConvN

### PDF p.85 (printed 61) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.1 A brief history of deep learning frameworks › A brief history of deep learning frameworks
It took a long time for the field to develop robust solutions for all three problems and package those solutions in a reusable form. Since its inception in the 1960s and until the 2000s, autodifferentiation had no practical applications in machine learning—folks who worked with neural networks simply wrote their own gradient logic by hand, usually in a language like C++. Meanwhile, GPU programming was all but impossible. Things started to slowly change in the late 2000s. First, Python and its ecosystem were slowly rising in popularity in the scientific community, gaining traction over MATLAB and C++. Second, NVIDIA released CUDA in 2006, unlocking the possibility of building neural networks that could run on consumer GPUs. The initial focus on CUDA was on physics simulation rather than machine learning, but that didn’t stop machine learning researchers from starting to implement CUDA-based neural networks from 2009 onward. They were typically one-off implementations that ran on a single GPU without any autodifferentiation. The first framework to enable autodifferentiation and GPU computation to train deep learning models was Theano, circa 2009. Theano is the conceptual ancestor of 

### PDF p.398 (printed 374) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › On RNN runtime performance
Recurrent models with very few parameters, like the ones in this chapter, tend to be significantly faster on a multicore CPU than on GPU because they only involve small matrix multiplications, and the chain of multiplications is not well parallelizable due to the presence of a `for` loop. But larger RNNs can greatly benefit from a GPU runtime. When using a Keras `LSTM` or `GRU` layer on GPU with default keyword arguments, your layer will be using a cuDNN kernel, a highly optimized, low-level, NVIDIA-provided implementation of the underlying algorithm (we’ve mentioned those in the previous chapter). As usual, cuDNN kernels are a mixed blessing: they’re fast, but inflexible—if you try doing anything not supported by the default kernel, you will suffer a dramatic

### PDF p.623 (printed 599) — 20 Conclusions › 20.1 Key concepts in review › 
Moore’s law applied to storage media. Today, state-of-the-art language models are trained on a large fraction of the entire internet.

The availability of fast, highly parallel computation hardware at a low price, especially the GPUs produced by NVIDIA—first gaming GPUs and then chips designed from the ground up for deep learning. Early on, NVIDIA CEO Jensen Huang took note of the deep learning boom and decided to bet the company’s future on it, which paid off in a big way.

A complex stack of software layers that makes this computational power available to humans: the CUDA language, frameworks like TensorFlow, JAX, and PyTorch that do automatic differentiation, and Keras, which makes deep learning accessible to most people.

In the future, deep learning will not be used only by specialists such as researchers, graduate students, and engineers with an academic profile; it will be a tool in the toolbox of every developer, much like web technology today. Everyone needs to build intelligent apps: just as every business today needs a website, every product will need to intelligently make sense of user-generated data. Bringing about this future will require us to build tools that make d

### PDF p.399 (printed 375) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
slowdown, which more or less forces you stick to what NVIDIA happens to provide. For instance, recurrent dropout isn’t supported by the LSTM and GRU cuDNN kernels, so adding it to your layers forces the runtime to fall back to the regular TensorFlow implementation, which is generally two to five times slower on GPU (even though its computational cost is the same). As a way to speed up your RNN layer when you can’t use cuDNN, you can try unrolling it. Unrolling a `for` loop consists of removing the loop and simply in-lining its content N times. In the case of the `for` loop of an RNN, unrolling can help TensorFlow optimize the underlying computation graph. However, it will also considerably increase the memory consumption of your RNN—as such, it’s only viable for relatively small sequences (around 100 steps or fewer). Also, you can only do this if the number of timesteps in the data is known in advance by the model (that is, if you pass a `shape` without any `None` entries to your initial `Input()`). It works like this:

### PDF p.54 (printed 30) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Likewise, when running JAX/TensorFlow/PyTorch code on a GPU, element-wise operations are executed via fully vectorized CUDA implementations that can best utilize the highly parallel GPU chip architecture.

### PDF p.635 (printed 611) — index ›  › 
ConvNet filters 285

achievements of 12 architecture patterns 268, 272–276, 280–282 advantages of 597 convolution operation 235–239 AI winters 14 depthwise separable convolutions 278–280 artificial intelligence, machine learning,

and 2–3, 599 interpreting what they learn 284 depth of model 7 max-pooling operation 239–241 enabling technologies 598 modularity, hierarchy, and reuse 269 for image generation 509–518 training from scratch on small dataset 241 frameworks 61–62 visualizing filters 291–299 generative AI 11 visualizing heatmaps of class activation 299–306 guided program search 606 visualizing intermediate activations 285–291 image classification for small-data problems 242 visualizing latent space of 306 integrating modules into hybrid systems 590 ViTs (Vision Transformers) 282 Keras 206–214 convolutional base 257 limitations of 564–570, 605 convolution kernel 237 network architectures 600–605 convolution operation 235–239, 550 object detection, training YOLO model 342 border effects and padding 238 overview of 8, 597 strides 239 properties of 10 Core Knowledge priors 581 short-term hype 13–14 cosine distance 462 staying up to date 607–608 cost function 9 text classificatio
