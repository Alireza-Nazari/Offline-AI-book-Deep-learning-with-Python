# Generation bundle for page_301  (PDF page 301, printed page 277)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.3 Batch normalization
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_301.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_301.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p301_e03] (body) Batch normalization does just that. It’s a type of layer (`BatchNormalization` in Keras) introduced in 2015 by Ioffe and Szegedy;2 it can adaptively normalize data even as the mean and variance change over time during training. During training, it uses the mean and variance of the current batch of data to normalize samples, and during inference (when a big enough batch of representative data may not be available), it uses an exponential moving average of the batchwise mean and variance of the data seen during training. Although Ioffe and Szegedy’s original paper suggested that batch normalization operates by “reducing internal covariate shift,” no one really knows for sure why batch normalization helps. There are various hypotheses but no certitudes. You’ll find that this is true of many things in deep learning—deep learning is not an exact science but a set of ever-changing, empirically derived engineering best practices, woven together by unreliable narratives. You will sometimes feel like the book you have in hand tells you how to do something but doesn’t quite satisfactorily say why it works: that’s because we know the how but we don’t know the why. Whenever a reliable explanation is available, we make sure to mention it. Batch normalization isn’t one of those cases. In practice, the main effect of batch normalization appears to be that it helps with gradient propagation—much like residual connections—and thus allows for deeper networks. Some very deep networks can only be trained if they include multiple `Batch` `Normalization` layers. For instance, batch normalization is used liberally in many of the advanced ConvNet architectures that come packaged with Keras, such as ResNet50, EfficientNet, and Xception. The `BatchNormalization` layer can be used after any layer—`Dense`, `Conv2D`, and so on:
[p301_e04] (code)
```
x = ...
x = layers.Conv2D(32, 3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
```
[p301_e05] (body) NOTE Both `Dense` and `Conv2D` involve a “bias vector,” a learned variable whose purpose is to make the layer affine rather than purely linear. For instance, `Conv2D` returns, schematically, `y = conv(x, kernel) + bias`, and `Dense` returns `y = dot(x, kernel) + bias`. Because the normalization step will take care of centering the layer’s output on zero, the bias vector is no longer needed when using `BatchNormalization`, and the layer can be created without it via the option `use_bias=False`. This makes the layer slightly leaner.
[p301_e06] (body) Importantly, I generally recommend placing the previous layer’s activation after the batch normalization layer (although this is still a subject of debate). So instead of doing
[p301_e07] (footnote) 2 Sergey Ioffe and Christian Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Inter
[p301_e08] (body) nal Covariate Shift,” Proceedings of the 32nd International Conference on Machine Learning (2015), https://arxiv.org/
[p301_e09] (footnote) abs/1502.03167.

## Code annotations printed next to listings (callouts, in order)
- Because the output of the
- Conv2D layer gets
- normalized, the layer doesn’t
- need its own bias vector.

## Equation candidates (formulas written inline in the text)
- [p301_e04] x = ...
x = layers.Conv2D(32, 3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
- [p301_e05] For instance, `Conv2D` returns, schematically, `y = conv(x, kernel) + bias`, and `Dense` returns `y = dot(x, kernel) + bias`.
- [p301_e05] Because the normalization step will take care of centering the layer’s output on zero, the bias vector is no longer needed when using `BatchNormalization`, and the layer can be created without it via the option `use_bias=False`.

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 300) text, for continuity
With residual connections, you can build networks of arbitrary depth, without having to worry about vanishing gradients. Now, let’s move on to the next essential ConvNet architecture pattern: batch normalization.

9.3

Batch normalization

Normalization in machine learning is a broad category of methods that seek to make different samples seen by a machine learning model more similar to each other, which helps the model learn and generalize well to new data. The most common form of data normalization is one you’ve seen several times in this book already: centering the data on zero by subtracting the mean from the data and giving the data a unit standard deviation by dividing the data by its standard deviation. In effect, this makes the assumption that the data follows a normal (or Gaussian) distribution and makes sure this distribution is centered and scaled to unit variance:

Previous examples you saw in this book normalized data before feeding it into models. But data normalization may be a concern after every transformation performed by the network: even if the data entering a `Dense` or `Conv2D` network has a 0 mean and unit variance, there’s no reason to expect a priori that this will be the case for the data coming out. Could normalizing intermediate activations help?

## Next page (PDF 302) text, for continuity
you would actually do the following:

The intuitive reason why is that batch normalization will center your inputs on zero, while your ReLU activation uses zero as a pivot for keeping or dropping activated channels: doing normalization before the activation maximizes the utilization of the ReLU. That said, this ordering best practice is not exactly critical, so if you do convolution-activation-batch normalization, your model will still train, and you won’t necessarily see worse results.

NOTE Batch normalization has many quirks. One of the main ones relates to fine-tuning: when fine-tuning a model that includes `BatchNormalization` layers, I recommend leaving these layers frozen (set their `trainable` attribute to `False`). Otherwise, they will keep updating their internal mean and variance, which can interfere with the very small updates applied to the surrounding `Conv2D` layers.

Now, let’s take a look at the last architecture pattern in our series: depthwise separable convolutions.

9.4

Depthwise separable convolutions

What if we told you that there’s a layer you can use as a drop-in replacement for `Conv2D` that will make your model smaller (fewer trainable weight parameters), leaner (fewer floating-point operations), and cause it to perform a few percentage points better on its task? That is precisely what the depthwise separable convolution layer does (`Separable` `Conv2D` in Keras). This layer performs a spatial convolution on each channel of its input, independently, before mixing output channels via a pointwise convolution (a 1 × 1 convolution), as shown in figure 9.4.

This is equivalent to separating the learning of spatial features and the learning of channel-wise features. In much the same way that convolution relies on the assumption that the patterns i …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.469 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.307 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.289 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.293 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.292 — 9 ConvNet architecture patterns › 
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.469 (printed 445) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
You’ll note that the normalization layers we’re using here aren’t `BatchNormalization` layers like those we’ve used in image models. That’s because `BatchNormalization` doesn’t work well for sequence data. Instead, we’re using the `LayerNormalization` layer, which normalizes each sequence independently from other sequences in the batch— like this, in NumPy-like pseudocode:

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
stored in a non-trainable weight, and at each batch, your layer would increment the counter by one. Among Keras built-in layers, the only layer that features non-trainable weights is the `BatchNormalization` layer, which we will introduce in chapter 9. The `BatchNormalization` layer needs non-trainable weights to track information about the mean and standard deviation of the data that passes through it, so as to perform an online approximation of feature normalization (a concept you’ve learned about in chapters 4 and 6).

### PDF p.289 (printed 265) — 8 Image classification › 8.3 Using a pretrained model › 
Note that you should not unfreeze “batch normalization” layers (`BatchNormalization`). Batch normalization and its effect on fine-tuning is explained in the next chapter.

You already completed the first three steps when doing feature extraction. Let’s proceed with step 4: you’ll unfreeze your `conv_base`.

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › 
When building a very deep ConvNet, it’s common to add batch normalization layers as well as residual connections—two architecture patterns that help gradient information flow smoothly through the network.

### PDF p.305 (printed 281) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

### PDF p.204 (printed 180) — 6 The universal workflow of machine learning › 6.2 Developing a model › 
converging. To make learning easier for your network, your data should have the following characteristics:

Take small values—Typically, most values should be in the 0–1 range.

Be homogeneous—That is, all features should take values in roughly the same range.

Additionally, the following stricter normalization practice is common and can help, although it isn’t always necessary (for example, you didn’t do this in the digit-classification example):

Normalize each feature independently to have a mean of 0.

Normalize each feature independently to have a standard deviation of 1.

This is easy to do with NumPy arrays:

### PDF p.49 (printed 25) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › The notion of data batches
In general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll come across in deep learning will be the samples axis. In the MNIST example, “samples” are images of digits.

In addition, deep learning models don’t process an entire dataset at once; rather, they break the data into small “batches,” or groups of samples with a fixed size. Concretely, here’s one batch of our MNIST digits, with a batch size of 128:

### PDF p.50 (printed 26) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › 
When considering such a batch tensor, the first axis (axis 0) is called the batch axis (or batch dimension). You’ll frequently encounter this term when using Keras and other deep learning libraries.
