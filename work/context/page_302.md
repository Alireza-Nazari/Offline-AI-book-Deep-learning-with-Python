# Generation bundle for page_302  (PDF page 302, printed page 278)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.4 Depthwise separable convolutions
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_302.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_302.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p302_e03] (code)
```
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.BatchNormalization()(x)
```
[p302_e04] (body) you would actually do the following:
[p302_e05] (code)
```
x = layers.Conv2D(32, 3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
```
[p302_e06] (body) The intuitive reason why is that batch normalization will center your inputs on zero, while your ReLU activation uses zero as a pivot for keeping or dropping activated channels: doing normalization before the activation maximizes the utilization of the ReLU. That said, this ordering best practice is not exactly critical, so if you do convolution-activation-batch normalization, your model will still train, and you won’t necessarily see worse results.
[p302_e07] (body) NOTE Batch normalization has many quirks. One of the main ones relates to fine-tuning: when fine-tuning a model that includes `BatchNormalization` layers, I recommend leaving these layers frozen (set their `trainable` attribute to `False`). Otherwise, they will keep updating their internal mean and variance, which can interfere with the very small updates applied to the surrounding `Conv2D` layers.
[p302_e08] (body) Now, let’s take a look at the last architecture pattern in our series: depthwise separable convolutions.
[p302_e09] (heading_section) 9.4
[p302_e10] (heading_section) Depthwise separable convolutions
[p302_e11] (body) What if we told you that there’s a layer you can use as a drop-in replacement for `Conv2D` that will make your model smaller (fewer trainable weight parameters), leaner (fewer floating-point operations), and cause it to perform a few percentage points better on its task? That is precisely what the depthwise separable convolution layer does (`Separable` `Conv2D` in Keras). This layer performs a spatial convolution on each channel of its input, independently, before mixing output channels via a pointwise convolution (a 1 × 1 convolution), as shown in figure 9.4.
[p302_e12] (body) This is equivalent to separating the learning of spatial features and the learning of channel-wise features. In much the same way that convolution relies on the assumption that the patterns in images are not tied to specific locations, depthwise separable convolution relies on the assumption that spatial locations in intermediate activations are highly correlated, but different channels are highly independent. Because this assumption is generally true for the image representations learned by deep

## Code annotations printed next to listings (callouts, in order)
- Note the lack of
- activation here.
- We place the activation after
- the BatchNormalization layer.

## Equation candidates (formulas written inline in the text)
- [p302_e03] x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.BatchNormalization()(x)
- [p302_e05] x = layers.Conv2D(32, 3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 301) text, for continuity
Batch normalization does just that. It’s a type of layer (`BatchNormalization` in Keras) introduced in 2015 by Ioffe and Szegedy;2 it can adaptively normalize data even as the mean and variance change over time during training. During training, it uses the mean and variance of the current batch of data to normalize samples, and during inference (when a big enough batch of representative data may not be available), it uses an exponential moving average of the batchwise mean and variance of the data seen during training. Although Ioffe and Szegedy’s original paper suggested that batch normalization operates by “reducing internal covariate shift,” no one really knows for sure why batch normalization helps. There are various hypotheses but no certitudes. You’ll find that this is true of many things in deep learning—deep learning is not an exact science but a set of ever-changing, empirically derived engineering best practices, woven together by unreliable narratives. You will sometimes feel like the book you have in hand tells you how to do something but doesn’t quite satisfactorily say why it works: that’s because we know the how but we don’t know the why. Whenever a reliable explanation is available, we make sure to mention it. Batch normalization isn’t one of those cases. In practice, the main effect of batch normalization appears to be that it helps with gradient propagation—much like residual connections—and thus allows for deeper networks. Some very deep networks can only be trained if they include multiple `Batch` `Normalization` layers. For instance, batch normalization is used liberally in many of the advanced ConvNet architectures that come packaged with Keras, such as ResNet50, EfficientNet, and Xception. The `BatchNormalization` layer can be used after any layer …

## Next page (PDF 303) text, for continuity
Figure 9.4 Depthwise separable convolution: a depthwise convolution followed by a pointwise convolution

neural networks, it serves as a useful prior that helps the model make more efficient use of its training data. A model with stronger priors about the structure of the information it will have to process is a better model—as long as the priors are accurate.

Depthwise separable convolution requires significantly fewer parameters and involves fewer computations compared to regular convolution, while having comparable representational power. They result in smaller models that converge faster and are less prone to overfitting. These advantages become especially important when you’re training small models from scratch on limited data. When it comes to larger-scale models, depthwise separable convolutions are the basis of the Xception architecture, a high-performing ConvNet that comes packaged with Keras. You can read more about the theoretical grounding for depthwise separable convolutions and Xception in the paper “Xception: Deep Learning with Depthwise Separable Convolutions.”3

The co-evolution of hardware, software, and algorithms

Consider a regular convolution operation with a 3 x 3 window, 64 input channels, and 64 output channels. It uses 3 × 3 × 64 × 64 = 36,864 trainable parameters, and when you apply it to an image, it runs a number of floating-point operations that is proportional to this parameter count. Meanwhile, consider an equivalent depthwise separable convolution: it only involves 3 × 3 × 64 + 64 × 64 = 4,672 trainable parameters and proportionally fewer floating-point operations. This efficiency improvement only increases as the number of filters or the size of the convolution windows gets larger.

3 François Chollet, “Xception: Deep Learning with Dep …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.304 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.297 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.292 — 9 ConvNet architecture patterns › 
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.300 — 9 ConvNet architecture patterns › 9.3 Batch normalization
- PDF p.307 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.304 (printed 280) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
As a result, you would expect depthwise separable convolutions to be dramatically faster, right? Hold on. This would be true if you were writing simple CUDA or C++ implementations of these algorithms—in fact, you do see a meaningful speedup when running on CPU, where the underlying implementation is parallelized C++. But in practice, you’re probably using a GPU, and what you’re executing on it is far from a “simple” CUDA implementation: it’s a cuDNN kernel, a piece of code that has been extraordinarily optimized, down to each machine instruction. It certainly makes sense to spend a lot of effort optimizing this code, since cuDNN convolutions on NVIDIA hardware are responsible for many exaflops of computation every day. But a side effect of this extreme micro-optimization is that alternative approaches have little chance to compete on performance—even approaches that have significant intrinsic advantages, like depthwise separable convolutions. Despite repeated requests to NVIDIA, depthwise separable convolutions have not benefited from nearly the same level of software and hardware optimization as regular convolutions, and as a result they remain only about as fast as regular convol

### PDF p.626 (printed 602) — 20 Conclusions › 20.1 Key concepts in review › ConvNets
Convolution layers look at spatially local patterns by applying the same geometric transformation to different spatial locations (patches) in an input tensor. This results in representations that are translation invariant, making convolution layers highly data efficient and modular. This idea is applicable to spaces of any dimensionality: 1D (continuous sequences), 2D (images), 3D (volumes), and so on. You can use the `Conv1D` layer to process sequences, the `Conv2D` layer to process images, and the `Conv3D` layer to process volumes. As a leaner, more efficient alternative to convolution layers, you can also use depthwise separable convolution layers, such as `SeparableConv2D`.

ConvNets, or convolutional networks, consist of stacks of convolution and max-pooling layers. The pooling layers let you spatially downsample the data, which is required to keep feature maps to a reasonable size as the number of features grows and to allow subsequent convolution layers to “see” a greater spatial extent of the inputs. ConvNets are often ended with either a `Flatten` operation or a global pooling layer, turning spatial feature maps into vectors, followed by `Dense` layers to achieve classific

### PDF p.305 (printed 281) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
stored in a non-trainable weight, and at each batch, your layer would increment the counter by one. Among Keras built-in layers, the only layer that features non-trainable weights is the `BatchNormalization` layer, which we will introduce in chapter 9. The `BatchNormalization` layer needs non-trainable weights to track information about the mean and standard deviation of the data that passes through it, so as to perform an online approximation of feature normalization (a concept you’ve learned about in chapters 4 and 6).

### PDF p.289 (printed 265) — 8 Image classification › 8.3 Using a pretrained model › 
Note that you should not unfreeze “batch normalization” layers (`BatchNormalization`). Batch normalization and its effect on fine-tuning is explained in the next chapter.

You already completed the first three steps when doing feature extraction. Let’s proceed with step 4: you’ll unfreeze your `conv_base`.

### PDF p.292 (printed 268) — 9 ConvNet architecture patterns ›  › This chapter covers
The modularity-hierarchy-reuse formula for model

architecture

An overview of standard best practices for

building ConvNets: residual connections,

batch normalization, and depthwise separable

convolutions

Ongoing design trends for computer vision

models

A model’s “architecture” is the sum of the choices that went into creating it: which layers to use, how to configure them, in what arrangement to connect them. These choices define the hypothesis space of your model: the space of possible functions that gradient descent can search over, parameterized by the model’s weights. Like feature engineering, a good hypothesis space encodes prior knowledge that you have about the problem at hand and its solution. For instance, using convolution layers means that you know in advance that the relevant patterns present in your input

### PDF p.469 (printed 445) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
You’ll note that the normalization layers we’re using here aren’t `BatchNormalization` layers like those we’ve used in image models. That’s because `BatchNormalization` doesn’t work well for sequence data. Instead, we’re using the `LayerNormalization` layer, which normalizes each sequence independently from other sequences in the batch— like this, in NumPy-like pseudocode:

### PDF p.300 (printed 276) — 9 ConvNet architecture patterns › 9.3 Batch normalization › Batch normalization
Previous examples you saw in this book normalized data before feeding it into models. But data normalization may be a concern after every transformation performed by the network: even if the data entering a `Dense` or `Conv2D` network has a 0 mean and unit variance, there’s no reason to expect a priori that this will be the case for the data coming out. Could normalizing intermediate activations help?
