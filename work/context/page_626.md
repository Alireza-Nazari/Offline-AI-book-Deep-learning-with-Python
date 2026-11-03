# Generation bundle for page_626  (PDF page 626, printed page 602)

Page kind: content
Chapter: 20 Conclusions
Section: 20.1 Key concepts in review
Subsection: 20.1.6 Key network architectures
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_626.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_626.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p626_e03] (code)
```
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```
[p626_e04] (body) To perform regression toward a vector of continuous values, end your stack of layers with a `Dense` layer with a number of units equal to the number of values you’re trying to predict (often a single one, such as the price of a house) and no activation. Various losses can be used for regression—most commonly `mean_squared_error` (MSE):
[p626_e05] (code)
```
inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_values)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="mse")
```
[p626_e06] (heading_subsubsection) ConvNets
[p626_e07] (body) Convolution layers look at spatially local patterns by applying the same geometric transformation to different spatial locations (patches) in an input tensor. This results in representations that are translation invariant, making convolution layers highly data efficient and modular. This idea is applicable to spaces of any dimensionality: 1D (continuous sequences), 2D (images), 3D (volumes), and so on. You can use the `Conv1D` layer to process sequences, the `Conv2D` layer to process images, and the `Conv3D` layer to process volumes. As a leaner, more efficient alternative to convolution layers, you can also use depthwise separable convolution layers, such as `SeparableConv2D`.
[p626_e08] (body) ConvNets, or convolutional networks, consist of stacks of convolution and max-pooling layers. The pooling layers let you spatially downsample the data, which is required to keep feature maps to a reasonable size as the number of features grows and to allow subsequent convolution layers to “see” a greater spatial extent of the inputs. ConvNets are often ended with either a `Flatten` operation or a global pooling layer, turning spatial feature maps into vectors, followed by `Dense` layers to achieve classification or regression. Here’s a typical image-classification network (categorical classification, in this case) using `SeparableConv2D` layers:
[p626_e09] (code)
```
inputs = keras.Input(shape=(height, width, channels))
x = layers.SeparableConv2D(32, 3, activation="relu")(inputs)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.SeparableConv2D(128, 3, activation="relu")(x)
```

## Chapter outline
Chapter: 20 Conclusions (PDF pages 619-632)
  - 20.1 Key concepts in review (p.619)
      - 20.1.1 Various approaches to artificial intelligence (p.620)
      - 20.1.2 What makes deep learning special within the field of machine learning (p.620)
      - 20.1.3 How to think about deep learning (p.621)
      - 20.1.4 Key enabling technologies (p.622)
      - 20.1.5 The universal machine learning workflow (p.623)
      - 20.1.6 Key network architectures (p.624)
  - 20.2 Limitations of deep learning (p.629)
  - 20.3 What might lie ahead (p.630)
  - 20.4 Staying up to date in a fast-moving field (p.631)
      - 20.4.1 Practice on real-world problems using Kaggle (p.631)
      - 20.4.2 Read about the latest developments on arXiv (p.631)
      - 20.4.3 Explore the Keras ecosystem (p.632)
  - 20.5 Final words (p.632)

## Previous page (PDF 625) text, for continuity
assume no specific structure in the input features: they’re called densely connected because the units of a `Dense` layer are connected to every other unit. The layer attempts to map relationships between any two input features; this is unlike a 2D convolution layer, for instance, which only looks at local relationships.

Densely connected networks are most commonly used for categorical data (for example, where the input features are lists of attributes), such as the Boston Housing Price dataset used in chapter 4. They’re also used as the final classification or regression stage of most networks. For instance, the ConvNets covered in chapter 8 typically end with one or two `Dense` layers, and so do the recurrent networks in chapter 13. Remember, to perform binary classification, end your stack of layers with a `Dense` layer with a single unit and a `sigmoid` activation, and use `binary_crossentropy` as the loss. Your targets should be either 0 or 1:

To perform single-label categorical classification (where each sample has exactly one class, no more), end your stack of layers with a `Dense` layer with a number of units equal to the number of classes and a `softmax` activation. If your targets are one-hot encoded, use `categorical_crossentropy` as the loss; if they’re integers, use `sparse_categorical_ ` `crossentropy`:

To perform multilabel categorical classification (where each sample can have several classes), end your stack of layers with a `Dense` layer with a number of units equal to the number of classes and a `sigmoid` activation, and use `binary_crossentropy` as the loss. Your targets should be k-hot encoded:

## Next page (PDF 627) text, for continuity
When building a very deep ConvNet, it’s common to add batch normalization layers as well as residual connections—two architecture patterns that help gradient information flow smoothly through the network.

Transformers

A Transformer looks at a set of vectors (such as word vectors) and uses neural attention to transform each vector into a representation that is aware of the context provided by the other vectors in the set. When the set in question is an ordered sequence, you can also use positional encoding to create Transformers that can take into account both global context and word order, capable of processing long text paragraphs much more effectively than RNNs or 1D ConvNets.

Transformers can be used for any set-processing or sequence-processing task, including text classification, but they excel especially at sequence-to-sequence learning, such as translating paragraphs in a source language into a target language. A sequence-to-sequence Transformer is made of two parts:

A `TransformerEncoder` that turns an input vector sequence into a context-aware, order-aware output vector sequence

A `TransformerDecoder` that takes the output of the `TransformerEncoder`, as well as a target sequence, and predicts what should come next in the target sequence

If you’re only processing a single sequence (or set) of vectors, you’d only use the `TransformerEncoder`.

Following is a sequence-to-sequence Transformer for mapping a source sequence to a target sequence (this setup could be used for machine translation or question-answering, for instance):

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.628 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.629 — 20 Conclusions › 20.2 Limitations of deep learning
- PDF p.270 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.133 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.302 (printed 278) — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions › Depthwise separable convolutions
What if we told you that there’s a layer you can use as a drop-in replacement for `Conv2D` that will make your model smaller (fewer trainable weight parameters), leaner (fewer floating-point operations), and cause it to perform a few percentage points better on its task? That is precisely what the depthwise separable convolution layer does (`Separable` `Conv2D` in Keras). This layer performs a spatial convolution on each channel of its input, independently, before mixing output channels via a pointwise convolution (a 1 × 1 convolution), as shown in figure 9.4.

This is equivalent to separating the learning of spatial features and the learning of channel-wise features. In much the same way that convolution relies on the assumption that the patterns in images are not tied to specific locations, depthwise separable convolution relies on the assumption that spatial locations in intermediate activations are highly correlated, but different channels are highly independent. Because this assumption is generally true for the image representations learned by deep

### PDF p.259 (printed 235) — 8 Image classification › 8.1 Introduction to convnets › 
for a MNIST digit, patterns involving all pixels), whereas convolution layers learn local patterns (see figure 8.1): in the case of images, patterns found in small 2D windows of the inputs. In the previous example, these windows were all 3 × 3. This key characteristic gives ConvNets two interesting properties:

The patterns they learn are translation invariant. After learning a certain pattern in the lower-right corner of a picture, a ConvNet can recognize it anywhere: for example, in the upper-left corner. A densely connected model would have to learn the pattern anew if it appeared at a new location. This makes ConvNets data efficient when pro

Figure 8.1 Images can be broken into

cessing images—because the visual world

local patterns such as edges, textures,

is fundamentally translation invariant. They

and so on.

need fewer training samples to learn representations that have generalization power.

They can learn spatial hierarchies of patterns (see figure 8.2). A first convolution layer will learn small local patterns such as edges, a second convolution layer will learn larger patterns made of the features of the first layers, and so on. This allows ConvNets to efficiently 

### PDF p.386 (printed 362) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.4 Let’s try a 1D convolutional model
Speaking of using the right architecture priors: since our input sequences feature daily cycles, perhaps a convolutional model could work? A temporal ConvNet could reuse the same representations across different days, much like a spatial ConvNet can reuse the same representations across different locations in an image.

You already know about the `Conv2D` and `SeparableConv2D` layers, which see their inputs through small windows that swipe across 2D grids. There are also 1D and even 3D versions of these layers: `Conv1D`, `SeparableConv1D`, and `Conv3D`.2 The `Conv1D` layer relies on 1D windows that slide across input sequences, and the `Conv3D` layer relies on cubic windows that slide across input volumes. You can thus build 1D ConvNets, strictly analogous to 2D ConvNets. They’re a great fit for any sequence data that follows the translation invariance assumption (meaning that if you slide a window over the sequence, the content of the window should follow the same properties independently of the location of the window). Let’s try one on our temperature forecasting problem. We’ll pick an initial window length of 24, so that we look at 24 hours of data at a time (one cycle). As we d

### PDF p.269 (printed 245) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Building your model
We will reuse the same general model structure you saw in the first example: the ConvNet will be a stack of alternated `Conv2D` (with `relu` activation) and `MaxPooling2D` layers.

But because we’re dealing with bigger images and a more complex problem, we’ll make our model larger, accordingly: it will have two more `Conv2D MaxPooling2D` stages. This serves both to augment the capacity of the model and to further reduce the size of the feature maps so they aren’t overly large when we reach the pooling layer. Here, because we start from inputs of size 180 × 180 pixels (a somewhat arbitrary choice), we end up with feature maps of size 7 × 7 just before the `GlobalAveragePooling2D` layer. NOTE The depth of the feature maps progressively increases in the model (from 32 to 512), whereas the size of the feature maps decreases (from 180 × 180 to 7 × 7). This is a pattern you’ll see in almost all ConvNets.

Because we’re looking at a binary classification problem, we’ll end the model with a single unit (a `Dense` layer of size 1) and a `sigmoid` activation. This unit will encode the probability that the model is looking at one class or the other.

One last small difference: we will start t

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Note that max pooling isn’t the only way you can achieve such downsampling. As you already know, you can also use strides in the prior convolution layer. And you can use average pooling instead of max pooling, where each local input patch is transformed by taking the average value of each channel over the patch, rather than the max. But max pooling tends to work better than these alternative solutions. In a nutshell, the reason is that features tend to encode the spatial presence of some pattern or concept over the different tiles of the feature map (hence the term feature map), and it’s more informative to look at the maximal presence of different features than at their average presence. So the most reasonable subsampling strategy is to first produce dense maps of features (via unstrided convolutions) and then look at the maximal activation of the features over small patches, rather than looking at sparser windows of the inputs (via strided convolutions) or averaging input patches, which could cause you to miss or dilute feature-presence information. At this point, you should understand the basics of ConvNets—feature maps, convolution, and max pooling—and you know how to build a s

### PDF p.297 (printed 273) — 9 ConvNet architecture patterns › 9.2 Residual connections › 
Note that adding the input back to the output of a block implies that the output should have the same shape as the input. This is not the case if your block includes convolutional layers with an increased number of filters or a max pooling layer. In such cases, use a 1 × 1 `Conv2D` layer with no activation to linearly project the residual to the desired output shape. You’d typically use `padding="same"` in the convolution layers in your target block to avoid spatial downsampling due to padding, and you’d use strides in the residual projection to match any downsampling caused by a max pooling layer.

Listing 9.2 The target block changing the number of output filters

### PDF p.117 (printed 93) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Different types of layers are appropriate for different tensor formats and different types of data processing. For instance, simple vector data, stored in 2D tensors of shape `(samples,` `features)`, is often processed by densely connected layers, also called fully connected or dense layers (the `Dense` class in Keras). Sequence data, stored in 3D tensors of shape `(samples,` `timesteps,` `features)`, is typically processed by recurrent layers, such as an LSTM layer, or 1D convolution layers (`Conv1D`). Image data, stored in rank-4 tensors, is usually processed by 2D convolution layers (`Conv2D`). You can think of layers as the LEGO bricks of deep learning, a metaphor that is made explicit by Keras. Building deep learning models in Keras is done by clipping together compatible layers to form useful data transformation pipelines.

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
the initial input. The high-level patterns learned by the ConvNet will still be very small with regard to the initial input, which may not be enough to learn to classify digits (try recognizing a digit by only looking at it through windows that are 7 × 7 pixels!). We need the features from the last convolution layer to contain information about the totality of the input.

The final feature map has dimensions 22 × 22. That’s huge—when you take the average of each 22 × 22 feature map, you are going to be destroying a lot of information compared to when your feature maps were only 3 × 3.

In short, the reason to use downsampling is to reduce the size of the feature maps, making the information they contain increasingly less spatially distributed and increasingly contained in the channels, while also inducing spatial-filter hierarchies by making successive convolution layers “look” at increasingly large windows (in terms of the fraction of the original input image they cover).
