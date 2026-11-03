# Generation bundle for page_303  (PDF page 303, printed page 279)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.4 Depthwise separable convolutions
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_303.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_303.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p303_e03] (figure_caption) Figure 9.4 Depthwise separable convolution: a depthwise convolution followed by a pointwise convolution
[p303_e04] (body) neural networks, it serves as a useful prior that helps the model make more efficient use of its training data. A model with stronger priors about the structure of the information it will have to process is a better model—as long as the priors are accurate.
[p303_e05] (body) Depthwise separable convolution requires significantly fewer parameters and involves fewer computations compared to regular convolution, while having comparable representational power. They result in smaller models that converge faster and are less prone to overfitting. These advantages become especially important when you’re training small models from scratch on limited data. When it comes to larger-scale models, depthwise separable convolutions are the basis of the Xception architecture, a high-performing ConvNet that comes packaged with Keras. You can read more about the theoretical grounding for depthwise separable convolutions and Xception in the paper “Xception: Deep Learning with Depthwise Separable Convolutions.”3
[p303_e06] (heading_subsection) The co-evolution of hardware, software, and algorithms
[p303_e07] (sidebar) Consider a regular convolution operation with a 3 x 3 window, 64 input channels, and 64 output channels. It uses 3 × 3 × 64 × 64 = 36,864 trainable parameters, and when you apply it to an image, it runs a number of floating-point operations that is proportional to this parameter count. Meanwhile, consider an equivalent depthwise separable convolution: it only involves 3 × 3 × 64 + 64 × 64 = 4,672 trainable parameters and proportionally fewer floating-point operations. This efficiency improvement only increases as the number of filters or the size of the convolution windows gets larger.
[p303_e08] (footnote) 3 François Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions,” Conference on Computer Vision and Pattern Recognition (2017), https://arxiv.org/abs/1610.02357.

## Code annotations printed next to listings (callouts, in order)
- Depthwise convolution:
- independent spatial
- convs per channel

## Figures
- Figure 9.4: Depthwise separable convolution: a depthwise convolution followed by a pointwise convolution  | labels inside the figure: 1 × 1 conv, (pointwise conv), Concatenate, Split channels, 3 × 3 conv, 3 × 3 conv, 3 × 3 conv, 3 × 3 conv

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 302) text, for continuity
you would actually do the following:

The intuitive reason why is that batch normalization will center your inputs on zero, while your ReLU activation uses zero as a pivot for keeping or dropping activated channels: doing normalization before the activation maximizes the utilization of the ReLU. That said, this ordering best practice is not exactly critical, so if you do convolution-activation-batch normalization, your model will still train, and you won’t necessarily see worse results.

NOTE Batch normalization has many quirks. One of the main ones relates to fine-tuning: when fine-tuning a model that includes `BatchNormalization` layers, I recommend leaving these layers frozen (set their `trainable` attribute to `False`). Otherwise, they will keep updating their internal mean and variance, which can interfere with the very small updates applied to the surrounding `Conv2D` layers.

Now, let’s take a look at the last architecture pattern in our series: depthwise separable convolutions.

9.4

Depthwise separable convolutions

What if we told you that there’s a layer you can use as a drop-in replacement for `Conv2D` that will make your model smaller (fewer trainable weight parameters), leaner (fewer floating-point operations), and cause it to perform a few percentage points better on its task? That is precisely what the depthwise separable convolution layer does (`Separable` `Conv2D` in Keras). This layer performs a spatial convolution on each channel of its input, independently, before mixing output channels via a pointwise convolution (a 1 × 1 convolution), as shown in figure 9.4.

This is equivalent to separating the learning of spatial features and the learning of channel-wise features. In much the same way that convolution relies on the assumption that the patterns i …

## Next page (PDF 304) text, for continuity
(continued)

As a result, you would expect depthwise separable convolutions to be dramatically faster, right? Hold on. This would be true if you were writing simple CUDA or C++ implementations of these algorithms—in fact, you do see a meaningful speedup when running on CPU, where the underlying implementation is parallelized C++. But in practice, you’re probably using a GPU, and what you’re executing on it is far from a “simple” CUDA implementation: it’s a cuDNN kernel, a piece of code that has been extraordinarily optimized, down to each machine instruction. It certainly makes sense to spend a lot of effort optimizing this code, since cuDNN convolutions on NVIDIA hardware are responsible for many exaflops of computation every day. But a side effect of this extreme micro-optimization is that alternative approaches have little chance to compete on performance—even approaches that have significant intrinsic advantages, like depthwise separable convolutions. Despite repeated requests to NVIDIA, depthwise separable convolutions have not benefited from nearly the same level of software and hardware optimization as regular convolutions, and as a result they remain only about as fast as regular convolutions, even though they’re using quadratically fewer parameters and floating-point operations. Note, though, that using depthwise separable convolutions remains a good idea even if it does not result in a speedup: their lower parameter count means that you are less at risk of overfitting, and their assumption that channels should be uncorrelated leads to faster model convergence and more robust representations. What is a slight inconvenience in this case can become an impassable wall in other situations: because the entire hardware and software ecosystem of deep learning has been …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.307 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.306 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.260 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.265 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.12 — contents › 
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.263 — 8 Image classification › 8.1 Introduction to convnets

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.305 (printed 281) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

### PDF p.626 (printed 602) — 20 Conclusions › 20.1 Key concepts in review › ConvNets
Convolution layers look at spatially local patterns by applying the same geometric transformation to different spatial locations (patches) in an input tensor. This results in representations that are translation invariant, making convolution layers highly data efficient and modular. This idea is applicable to spaces of any dimensionality: 1D (continuous sequences), 2D (images), 3D (volumes), and so on. You can use the `Conv1D` layer to process sequences, the `Conv2D` layer to process images, and the `Conv3D` layer to process volumes. As a leaner, more efficient alternative to convolution layers, you can also use depthwise separable convolution layers, such as `SeparableConv2D`.

ConvNets, or convolutional networks, consist of stacks of convolution and max-pooling layers. The pooling layers let you spatially downsample the data, which is required to keep feature maps to a reasonable size as the number of features grows and to allow subsequent convolution layers to “see” a greater spatial extent of the inputs. ConvNets are often ended with either a `Flatten` operation or a global pooling layer, turning spatial feature maps into vectors, followed by `Dense` layers to achieve classific

### PDF p.260 (printed 236) — 8 Image classification › 8.1 Introduction to convnets › 
Convolutions are defined by two key parameters:

Size of the patches extracted from the inputs—These are typically 3 × 3 or 5 × 5. In the example, they were 3 × 3, which is a common choice.

Depth of the output feature map—The number of filters computed by the convolution. The example started with a depth of 32 and ended with a depth of 64.

In Keras `Conv2D` layers, these parameters are the first arguments passed to the layer:

### PDF p.635 (printed 611) — index ›  › 
ConvNet filters 285

achievements of 12 architecture patterns 268, 272–276, 280–282 advantages of 597 convolution operation 235–239 AI winters 14 depthwise separable convolutions 278–280 artificial intelligence, machine learning,

and 2–3, 599 interpreting what they learn 284 depth of model 7 max-pooling operation 239–241 enabling technologies 598 modularity, hierarchy, and reuse 269 for image generation 509–518 training from scratch on small dataset 241 frameworks 61–62 visualizing filters 291–299 generative AI 11 visualizing heatmaps of class activation 299–306 guided program search 606 visualizing intermediate activations 285–291 image classification for small-data problems 242 visualizing latent space of 306 integrating modules into hybrid systems 590 ViTs (Vision Transformers) 282 Keras 206–214 convolutional base 257 limitations of 564–570, 605 convolution kernel 237 network architectures 600–605 convolution operation 235–239, 550 object detection, training YOLO model 342 border effects and padding 238 overview of 8, 597 strides 239 properties of 10 Core Knowledge priors 581 short-term hype 13–14 cosine distance 462 staying up to date 607–608 cost function 9 text classificatio

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
the initial input. The high-level patterns learned by the ConvNet will still be very small with regard to the initial input, which may not be enough to learn to classify digits (try recognizing a digit by only looking at it through windows that are 7 × 7 pixels!). We need the features from the last convolution layer to contain information about the totality of the input.

The final feature map has dimensions 22 × 22. That’s huge—when you take the average of each 22 × 22 feature map, you are going to be destroying a lot of information compared to when your feature maps were only 3 × 3.

In short, the reason to use downsampling is to reduce the size of the feature maps, making the information they contain increasingly less spatially distributed and increasingly contained in the channels, while also inducing spatial-filter hierarchies by making successive convolution layers “look” at increasingly large windows (in terms of the fraction of the original input image they cover).

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Note that max pooling isn’t the only way you can achieve such downsampling. As you already know, you can also use strides in the prior convolution layer. And you can use average pooling instead of max pooling, where each local input patch is transformed by taking the average value of each channel over the patch, rather than the max. But max pooling tends to work better than these alternative solutions. In a nutshell, the reason is that features tend to encode the spatial presence of some pattern or concept over the different tiles of the feature map (hence the term feature map), and it’s more informative to look at the maximal presence of different features than at their average presence. So the most reasonable subsampling strategy is to first produce dense maps of features (via unstrided convolutions) and then look at the maximal activation of the features over small patches, rather than looking at sparser windows of the inputs (via strided convolutions) or averaging input patches, which could cause you to miss or dilute feature-presence information. At this point, you should understand the basics of ConvNets—feature maps, convolution, and max pooling—and you know how to build a s

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
