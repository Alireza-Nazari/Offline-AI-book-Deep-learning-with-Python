# Generation bundle for page_298  (PDF page 298, printed page 274)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.2 Residual connections
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_298.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_298.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p298_e03] (code)
```
x = layers.Conv2D(32, 3, activation="relu")(inputs)
residual = x
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
residual = layers.Conv2D(64, 1)(residual)
x = layers.add([x, residual])
```
[p298_e04] (listing_caption) Listing 9.3 The target block including a max pooling layer
[p298_e05] (code)
```
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
residual = x
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
residual = layers.Conv2D(64, 1, strides=2)(residual)
x = layers.add([x, residual])
```
[p298_e06] (body) To make these ideas more concrete, here’s an example of a simple ConvNet structured into a series of blocks, each made of two convolution layers and one optional max pooling layer, with a residual connection around each block:
[p298_e07] (code)
```
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Rescaling(1.0 / 255)(inputs)
def residual_block(x, filters, pooling=False):
    residual = x
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
```

## Code annotations printed next to listings (callouts, in order)
- Sets aside the residual
- This is the layer around which
- we create a residual connection:
- Now the block output and the residual
- it increases the number of
- have the same shape and can be added.
- output filers from 32 to 64. We
- use padding="same" to avoid
- The residual only had 32 filters, so we use a
- downsampling due to padding.
- 1 x 1 Conv2D to project it to the correct shape.
- This is the block of two layers around which we create a
- residual connection: it includes a 2 x 2 max pooling layer. We
- use padding="same" in both the convolution layer and the
- max pooling layer to avoid downsampling due to padding.
- Sets aside the residual
- We use strides=2
- in the residual
- projection to
- match the
- Now the block output and the residual
- downsampling
- have the same shape and can be added.
- created by the max
- pooling layer.
- If we use max pooling, we add a strided convolution
- to project the residual to the expected shape.
- Utility function to apply a
- convolutional block with a
- residual connection, with an
- option to add max pooling

## Listings
- Listing 9.3: The target block including a max pooling layer (code elements p298_e05)

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 297) text, for continuity
The fix is simple: just force each function in the chain to be nondestructive—to retain a noiseless version of the information contained in the previous input. The easiest way to implement this is called a residual connection. It’s dead easy: just add the input of a layer or block of layers back to its output (see figure 9.3). The residual connection acts as an information shortcut around destructive or noisy blocks (such as blocks that contain ReLU activations or dropout layers), enabling error gradient information from early

Figure 9.3 A residual connection

layers to propagate noiselessly through a deep net

around a processing block

work. This technique was introduced in 2015 with the ResNet family of models (developed by He et al. at Microsoft).1

In practice, you’d implement a residual connection like the following listing.

Listing 9.1 A residual connection in pseudocode

Note that adding the input back to the output of a block implies that the output should have the same shape as the input. This is not the case if your block includes convolutional layers with an increased number of filters or a max pooling layer. In such cases, use a 1 × 1 `Conv2D` layer with no activation to linearly project the residual to the desired output shape. You’d typically use `padding="same"` in the convolution layers in your target block to avoid spatial downsampling due to padding, and you’d use strides in the residual projection to match any downsampling caused by a max pooling layer.

Listing 9.2 The target block changing the number of output filters

1 Kaiming He et al., “Deep Residual Learning for Image Recognition,” Conference on Computer Vision and Pattern Recognition (2015), https://arxiv.org/abs/1512.03385.

## Next page (PDF 299) text, for continuity
Let’s take a look at the model summary:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.256 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.304 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.270 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.263 (printed 239) — 8 Image classification › 8.1 Introduction to convnets › The max-pooling operation
In the ConvNet example, you may have noticed that the size of the feature maps is halved after every `MaxPooling2D` layer. For instance, before the first `MaxPooling2D` layers, the feature map is 26 × 26, but the max-pooling operation halves it to 13 × 13. That’s the role of max pooling: to aggressively downsample feature maps, much like strided convolutions.

### PDF p.264 (printed 240) — 8 Image classification › 8.1 Introduction to convnets › 
Max pooling consists of extracting windows from the input feature maps and outputting the max value of each channel. It’s conceptually similar to convolution, except that instead of transforming local patches via a learned linear transformation (the convolution kernel), they’re transformed via a hardcoded `max` tensor operation. A big difference from convolution is that max pooling is usually done with 2 × 2 windows and stride 2 to downsample the feature maps by a factor of 2. On the other hand, convolution is typically done with 3 × 3 windows and no stride (stride 1). Why downsample feature maps this way? Why not remove the max-pooling layers and keep fairly large feature maps all the way up? Let’s look at this option. Our model would then look like this.

Listing 8.5 An incorrectly structured ConvNet missing its max-pooling layers

### PDF p.304 (printed 280) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › Putting it together: A mini Xception-like model
As a reminder, here are the ConvNet architecture principles you’ve learned so far:

Your model should be organized into repeated blocks of layers, usually made of multiple convolution layers and a max pooling layer.

The number of filters in your layers should increase as the size of the spatial feature maps decreases.

### PDF p.305 (printed 281) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

### PDF p.264 (printed 240) — 8 Image classification › 8.1 Introduction to convnets › 
Listing 8.5 An incorrectly structured ConvNet missing its max-pooling layers
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model_no_max_pool = keras.Model(inputs=inputs, outputs=outputs)

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Note that max pooling isn’t the only way you can achieve such downsampling. As you already know, you can also use strides in the prior convolution layer. And you can use average pooling instead of max pooling, where each local input patch is transformed by taking the average value of each channel over the patch, rather than the max. But max pooling tends to work better than these alternative solutions. In a nutshell, the reason is that features tend to encode the spatial presence of some pattern or concept over the different tiles of the feature map (hence the term feature map), and it’s more informative to look at the maximal presence of different features than at their average presence. So the most reasonable subsampling strategy is to first produce dense maps of features (via unstrided convolutions) and then look at the maximal activation of the features over small patches, rather than looking at sparser windows of the inputs (via strided convolutions) or averaging input patches, which could cause you to miss or dilute feature-presence information. At this point, you should understand the basics of ConvNets—feature maps, convolution, and max pooling—and you know how to build a s

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › 
When building a very deep ConvNet, it’s common to add batch normalization layers as well as residual connections—two architecture patterns that help gradient information flow smoothly through the network.

### PDF p.339 (printed 315) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
One important difference between the first half of this model and the classification models you’ve seen before is the way we do downsampling: in the classification ConvNets from chapter 8, we used `MaxPooling2D` layers to downsample feature maps. Here, we downsample by adding strides to every other convolution layer (if you don’t remember the details of how convolution strides work, see chapter 8, section 8.1.1). We do this because, in the case of image segmentation, we care a lot about the spatial location of information in the image since we need to produce per-pixel target masks as output of the model. When you do 2 × 2 max pooling, you are completely destroying location information within each pooling window: you return one scalar value per window, with zero knowledge of which of the four locations in the windows the value came from. So, while max pooling layers perform well for classification tasks, they would hurt us quite a bit for a segmentation task. Meanwhile, strided convolutions do a better job at downsampling feature maps while retaining location information. Throughout this book, you’ll notice that we tend to use strides instead of max pooling in any model that cares 
