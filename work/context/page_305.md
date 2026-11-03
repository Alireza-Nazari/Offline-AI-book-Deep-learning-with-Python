# Generation bundle for page_305  (PDF page 305, printed page 281)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.5 Putting it together: A mini Xception-like model
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_305.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_305.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p305_e03] (bullet) Deep and narrow is better than broad and shallow.
[p305_e04] (bullet) Introducing residual connections around blocks of layers helps you train deeper networks.
[p305_e05] (bullet) It can be beneficial to introduce batch normalization layers after your convolution layers.
[p305_e06] (bullet) It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.
[p305_e07] (body) Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:
[p305_e08] (code)
```
import keras
inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)
for size in [32, 64, 128, 256, 512]:
    residual = x
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    residual = layers.Conv2D(
        size, 1, strides=2, padding="same", use_bias=False
    )(residual)
    x = layers.add([x, residual])
x = layers.GlobalAveragePooling2D()(x)
```

## Code annotations printed next to listings (callouts, in order)
- We apply a series of convolutional blocks with increasing
- feature depth. Each block consists of two batch-normalized
- depthwise separable convolution layers and a max pooling
- layer, with a residual connection around the entire block.
- The assumption that underlies separable convolution, “Feature
- channels are largely independent,” does not hold for RGB images!
- Red, green, and blue color channels are actually highly correlated in
- natural images. As such, the first layer in our model is a regular
- Conv2D layer. We’ll start using SeparableConv2D afterward.
- Don’t forget
- input rescaling!
- In the original model, we used a Flatten
- layer before the Dense layer. Here, we go
- with a GlobalAveragePooling2D layer.

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 304) text, for continuity
(continued)

As a result, you would expect depthwise separable convolutions to be dramatically faster, right? Hold on. This would be true if you were writing simple CUDA or C++ implementations of these algorithms—in fact, you do see a meaningful speedup when running on CPU, where the underlying implementation is parallelized C++. But in practice, you’re probably using a GPU, and what you’re executing on it is far from a “simple” CUDA implementation: it’s a cuDNN kernel, a piece of code that has been extraordinarily optimized, down to each machine instruction. It certainly makes sense to spend a lot of effort optimizing this code, since cuDNN convolutions on NVIDIA hardware are responsible for many exaflops of computation every day. But a side effect of this extreme micro-optimization is that alternative approaches have little chance to compete on performance—even approaches that have significant intrinsic advantages, like depthwise separable convolutions. Despite repeated requests to NVIDIA, depthwise separable convolutions have not benefited from nearly the same level of software and hardware optimization as regular convolutions, and as a result they remain only about as fast as regular convolutions, even though they’re using quadratically fewer parameters and floating-point operations. Note, though, that using depthwise separable convolutions remains a good idea even if it does not result in a speedup: their lower parameter count means that you are less at risk of overfitting, and their assumption that channels should be uncorrelated leads to faster model convergence and more robust representations. What is a slight inconvenience in this case can become an impassable wall in other situations: because the entire hardware and software ecosystem of deep learning has been …

## Next page (PDF 306) text, for continuity
This ConvNet has a trainable parameter count of 721,857, significantly lower than the 1,569,089 trainable parameters of the model from the previous chapter, yet it achieves better results. Figure 9.5 shows the training and validation curves.

Figure 9.5 Training and validation metrics with a Xception-like architecture

You’ll find that our new model achieves a test accuracy of 90.8%—compared to 83.9% for the previous model. As you can see, following architecture best practices does have an immediate, sizeable effect on model performance!

At this point, if you want to further improve performance, you should start systematically tuning the hyperparameters of your architecture—a topic we cover in detail in chapter 18. We haven’t gone through this step here, so the configuration of the previous model is purely from the best practices we outlined, plus, when it comes to gauging model size, a small amount of intuition.

9.6

Beyond convolution: Vision Transformers

While ConvNets have been dominating the field of computer vision since the mid-2010s, they’ve been recently competing with an alternative architecture: Vision Transformers (or ViTs for short). It may well be that ViTs will end up replacing ConvNets in the long term—though, for now, ConvNets remain your best option in most cases.

You don’t yet know what Transformers are because we’ll cover them in chapter 15. In short, the Transformer architecture was developed to process text—it’s fundamentally

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.298 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.301 — 9 ConvNet architecture patterns › 9.3 Batch normalization
- PDF p.299 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.297 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.256 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.295 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.295 (printed 271) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › 
Figure 9.2 The “entry flow” of the Xception architecture: note the repeated layer blocks and the gradually shrinking and deepening feature maps, going from 299 x 299 x 3 to 19 x 19 x 728.

Deeper hierarchies are intrinsically good because they encourage feature reuse and, therefore, abstraction. In general, a deep stack of narrow layers performs better than a shallow stack of large layers. However, there’s a limit to how deep you can stack layers: the problem of vanishing gradients. This leads us to our first essential model architecture pattern: residual connections.

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › 
When building a very deep ConvNet, it’s common to add batch normalization layers as well as residual connections—two architecture patterns that help gradient information flow smoothly through the network.

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

### PDF p.303 (printed 279) — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions › 
Figure 9.4 Depthwise separable convolution: a depthwise convolution followed by a pointwise convolution

neural networks, it serves as a useful prior that helps the model make more efficient use of its training data. A model with stronger priors about the structure of the information it will have to process is a better model—as long as the priors are accurate.

Depthwise separable convolution requires significantly fewer parameters and involves fewer computations compared to regular convolution, while having comparable representational power. They result in smaller models that converge faster and are less prone to overfitting. These advantages become especially important when you’re training small models from scratch on limited data. When it comes to larger-scale models, depthwise separable convolutions are the basis of the Xception architecture, a high-performing ConvNet that comes packaged with Keras. You can read more about the theoretical grounding for depthwise separable convolutions and Xception in the paper “Xception: Deep Learning with Depthwise Separable Convolutions.”3

### PDF p.316 (printed 292) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
You’ll notice that the `SeparableConv2D` layers here are all named something like `block6_sepconv1`, `block7_sepconv2`, etc.—Xception is structured into blocks, each containing several convolutional layers.

Now let’s create a second model that returns the output of a specific layer—a “feature extractor” model. Because our model is a Functional API model, it is inspectable: you can query the `output` of one of its layers and reuse it in a new model. No need to copy the entire Xception code.

Listing 10.9 A feature extractor model returning a specific output

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

In this case, because the ImageNet class set contains multiple dog and cat classes, it’s likely to be beneficial to reuse the information contained in the densely connected layers of the original model. But we’ll choose not to, so we can cover the more general case where the class set of the new problem doesn’t overlap the class set of the original model. Let’s put this in practice by using the convolutional base of our pretrained model to extract interesting features from cat and dog images and then train a dogs-versus-cats classifier on top of these features. We will use the KerasHub library to create all pretrained models used in this book. KerasHub contains Keras implementations of popular pretrained model architectures paired with pretrained weights that can be downloaded to your machine. It contains a number of ConvNets like Xception, ResNet, EfficientNet and MobileNet, as well as larger, generative models we will use in the later chapters of this book. Let’s try using it to instantiate the Xception model trained on the ImageNet dataset.

NOTE KerasHub comes as a

### PDF p.302 (printed 278) — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions › 
The intuitive reason why is that batch normalization will center your inputs on zero, while your ReLU activation uses zero as a pivot for keeping or dropping activated channels: doing normalization before the activation maximizes the utilization of the ReLU. That said, this ordering best practice is not exactly critical, so if you do convolution-activation-batch normalization, your model will still train, and you won’t necessarily see worse results.

NOTE Batch normalization has many quirks. One of the main ones relates to fine-tuning: when fine-tuning a model that includes `BatchNormalization` layers, I recommend leaving these layers frozen (set their `trainable` attribute to `False`). Otherwise, they will keep updating their internal mean and variance, which can interfere with the very small updates applied to the surrounding `Conv2D` layers.

Now, let’s take a look at the last architecture pattern in our series: depthwise separable convolutions.

### PDF p.301 (printed 277) — 9 ConvNet architecture patterns › 9.3 Batch normalization › 
Batch normalization does just that. It’s a type of layer (`BatchNormalization` in Keras) introduced in 2015 by Ioffe and Szegedy;2 it can adaptively normalize data even as the mean and variance change over time during training. During training, it uses the mean and variance of the current batch of data to normalize samples, and during inference (when a big enough batch of representative data may not be available), it uses an exponential moving average of the batchwise mean and variance of the data seen during training. Although Ioffe and Szegedy’s original paper suggested that batch normalization operates by “reducing internal covariate shift,” no one really knows for sure why batch normalization helps. There are various hypotheses but no certitudes. You’ll find that this is true of many things in deep learning—deep learning is not an exact science but a set of ever-changing, empirically derived engineering best practices, woven together by unreliable narratives. You will sometimes feel like the book you have in hand tells you how to do something but doesn’t quite satisfactorily say why it works: that’s because we know the how but we don’t know the why. Whenever a reliable explanat
