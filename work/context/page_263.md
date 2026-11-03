# Generation bundle for page_263  (PDF page 263, printed page 239)

Page kind: content
Chapter: 8 Image classification
Section: 8.1 Introduction to convnets
Subsection: 8.1.2 The max-pooling operation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_263.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_263.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p263_e03] (body) In `Conv2D` layers, padding is configurable via the `padding` argument, which takes two values: `"valid"`, which means no padding (only valid window locations will be used); and `"same"`, which means “pad in such a way as to have an output with the same width and height as the input.” The `padding` argument defaults to `"valid"`.
[p263_e04] (heading_subsubsection) Understanding convolution strides
[p263_e05] (body) The other factor that can influence output size is the notion of strides. The description of convolution so far has assumed that the center tiles of the convolution windows are all contiguous. But the distance between two successive windows is a parameter of the convolution, called its stride, which defaults to 1. It’s possible to have strided convolutions: convolutions with a stride higher than 1. In figure 8.7, you can see the patches extracted by a 3 × 3 convolution with stride 2 over a 5 × 5 input (without padding)
[p263_e06] (figure_caption) Figure 8.7 3 × 3 convolution patches with 2 × 2 strides
[p263_e07] (body) Using stride 2 means the width and height of the feature map are downsampled by a factor of 2 (in addition to any changes induced by border effects). Strided convolutions are rarely used in classification models, but they come in handy for some types of models, as you will find out in the next chapter.
[p263_e08] (body) In classification models, instead of strides, we tend to use the max-pooling operation to downsample feature maps—which you saw in action in our first ConvNet example. Let’s look at it in more depth.
[p263_e09] (heading_subsection) 8.1.2
[p263_e10] (heading_subsection) The max-pooling operation
[p263_e11] (body) In the ConvNet example, you may have noticed that the size of the feature maps is halved after every `MaxPooling2D` layer. For instance, before the first `MaxPooling2D` layers, the feature map is 26 × 26, but the max-pooling operation halves it to 13 × 13. That’s the role of max pooling: to aggressively downsample feature maps, much like strided convolutions.

## Figures
- Figure 8.7: 3 × 3 convolution patches with 2 × 2 strides  | labels inside the figure: 1, 1, 2, 3, 4, 2, 3, 4

## Chapter outline
Chapter: 8 Image classification (PDF pages 255-291)
  - 8.1 Introduction to convnets (p.256)
      - 8.1.1 The convolution operation (p.258)
      - 8.1.2 The max-pooling operation (p.263)
  - 8.2 Training a convnet from scratch on a small dataset (p.265)
      - 8.2.1 The relevance of deep learning for small-data problems (p.266)
      - 8.2.2 Downloading the data (p.266)
      - 8.2.3 Building your model (p.269)
      - 8.2.4 Data preprocessing (p.271)
      - 8.2.5 Using data augmentation (p.276)
  - 8.3 Using a pretrained model (p.280)
      - 8.3.1 Feature extraction with a pretrained model (p.280)
      - 8.3.2 Fine-tuning a pretrained model (p.288)

## Previous page (PDF 262) text, for continuity
Understanding border effects and padding

Consider a 5 × 5 feature map (25 tiles total). There are only 9 tiles around which you can center a 3 × 3 window, forming a 3 × 3 grid (see figure 8.5). Hence, the output feature map will be 3 × 3. It shrinks a little: by exactly two tiles alongside each dimension, in this case. You can see this border effect in action in the earlier example: you start with 28 × 28 inputs, which become 26 × 26 after the first convolution layer.

Figure 8.5 Valid locations of 3 × 3 patches in a 5 × 5 input feature map

If you want to get an output feature map with the same spatial dimensions as the input, you can use padding. Padding consists of adding an appropriate number of rows and columns on each side of the input feature map so as to make it possible to fit centered convolution windows around every input tile. For a 3 × 3 window, you add one column on the right, one column on the left, one row at the top, and one row at the bottom. For a 5 × 5 window, you add two rows (see figure 8.6).

Figure 8.6 Padding a 5 × 5 input to be able to extract 25 3 × 3 patches

## Next page (PDF 264) text, for continuity
Max pooling consists of extracting windows from the input feature maps and outputting the max value of each channel. It’s conceptually similar to convolution, except that instead of transforming local patches via a learned linear transformation (the convolution kernel), they’re transformed via a hardcoded `max` tensor operation. A big difference from convolution is that max pooling is usually done with 2 × 2 windows and stride 2 to downsample the feature maps by a factor of 2. On the other hand, convolution is typically done with 3 × 3 windows and no stride (stride 1). Why downsample feature maps this way? Why not remove the max-pooling layers and keep fairly large feature maps all the way up? Let’s look at this option. Our model would then look like this.

Listing 8.5 An incorrectly structured ConvNet missing its max-pooling layers

Here’s a summary of the model:

What’s wrong with this setup? Two things:

It isn’t conducive to learning a spatial hierarchy of features. The 3 × 3 windows in the third layer will only contain information coming from 7 × 7 windows in

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.339 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.265 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.260 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.261 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.269 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.297 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.257 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.339 (printed 315) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
One important difference between the first half of this model and the classification models you’ve seen before is the way we do downsampling: in the classification ConvNets from chapter 8, we used `MaxPooling2D` layers to downsample feature maps. Here, we downsample by adding strides to every other convolution layer (if you don’t remember the details of how convolution strides work, see chapter 8, section 8.1.1). We do this because, in the case of image segmentation, we care a lot about the spatial location of information in the image since we need to produce per-pixel target masks as output of the model. When you do 2 × 2 max pooling, you are completely destroying location information within each pooling window: you return one scalar value per window, with zero knowledge of which of the four locations in the windows the value came from. So, while max pooling layers perform well for classification tasks, they would hurt us quite a bit for a segmentation task. Meanwhile, strided convolutions do a better job at downsampling feature maps while retaining location information. Throughout this book, you’ll notice that we tend to use strides instead of max pooling in any model that cares 

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Note that max pooling isn’t the only way you can achieve such downsampling. As you already know, you can also use strides in the prior convolution layer. And you can use average pooling instead of max pooling, where each local input patch is transformed by taking the average value of each channel over the patch, rather than the max. But max pooling tends to work better than these alternative solutions. In a nutshell, the reason is that features tend to encode the spatial presence of some pattern or concept over the different tiles of the feature map (hence the term feature map), and it’s more informative to look at the maximal presence of different features than at their average presence. So the most reasonable subsampling strategy is to first produce dense maps of features (via unstrided convolutions) and then look at the maximal activation of the features over small patches, rather than looking at sparser windows of the inputs (via strided convolutions) or averaging input patches, which could cause you to miss or dilute feature-presence information. At this point, you should understand the basics of ConvNets—feature maps, convolution, and max pooling—and you know how to build a s

### PDF p.261 (printed 237) — 8 Image classification › 8.1 Introduction to convnets › 
done via a tensor product with a learned weight matrix, called the convolution kernel— the same kernel is reused across every patch. All of these vectors (one per patch) are then spatially reassembled into a 3D output map of shape `(height,` `width,` `output_` `depth)`. Every spatial location in the output feature map corresponds to the same location in the input feature map (for example, the lower-right corner of the output contains information about the lower-right corner of the input). For instance, with 3 × 3 windows, the vector `output[i, j, :]` comes from the 3D patch `input[i-1:i+1, ` `j-1:j+1, :]`. The full process is detailed in figure 8.4.

Figure 8.4 How convolution works

Note that the output width and height may differ from the input width and height. They may differ for two reasons:

Border effects, which can be countered by padding the input feature map

The use of strides, which we’ll define in a second

Let’s take a deeper look at these notions.

### PDF p.297 (printed 273) — 9 ConvNet architecture patterns › 9.2 Residual connections › 
Note that adding the input back to the output of a block implies that the output should have the same shape as the input. This is not the case if your block includes convolutional layers with an increased number of filters or a max pooling layer. In such cases, use a 1 × 1 `Conv2D` layer with no activation to linearly project the residual to the desired output shape. You’d typically use `padding="same"` in the convolution layers in your target block to avoid spatial downsampling due to padding, and you’d use strides in the residual projection to match any downsampling caused by a max pooling layer.

Listing 9.2 The target block changing the number of output filters

### PDF p.260 (printed 236) — 8 Image classification › 8.1 Introduction to convnets › 
Convolutions are defined by two key parameters:

Size of the patches extracted from the inputs—These are typically 3 × 3 or 5 × 5. In the example, they were 3 × 3, which is a common choice.

Depth of the output feature map—The number of filters computed by the convolution. The example started with a depth of 32 and ended with a depth of 64.

In Keras `Conv2D` layers, these parameters are the first arguments passed to the layer:

### PDF p.626 (printed 602) — 20 Conclusions › 20.1 Key concepts in review › ConvNets
Convolution layers look at spatially local patterns by applying the same geometric transformation to different spatial locations (patches) in an input tensor. This results in representations that are translation invariant, making convolution layers highly data efficient and modular. This idea is applicable to spaces of any dimensionality: 1D (continuous sequences), 2D (images), 3D (volumes), and so on. You can use the `Conv1D` layer to process sequences, the `Conv2D` layer to process images, and the `Conv3D` layer to process volumes. As a leaner, more efficient alternative to convolution layers, you can also use depthwise separable convolution layers, such as `SeparableConv2D`.

ConvNets, or convolutional networks, consist of stacks of convolution and max-pooling layers. The pooling layers let you spatially downsample the data, which is required to keep feature maps to a reasonable size as the number of features grows and to allow subsequent convolution layers to “see” a greater spatial extent of the inputs. ConvNets are often ended with either a `Flatten` operation or a global pooling layer, turning spatial feature maps into vectors, followed by `Dense` layers to achieve classific

### PDF p.298 (printed 274) — 9 ConvNet architecture patterns › 9.2 Residual connections › 
Listing 9.3 The target block including a max pooling layer
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
residual = x
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
residual = layers.Conv2D(64, 1, strides=2)(residual)
x = layers.add([x, residual])

### PDF p.269 (printed 245) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Building your model
We will reuse the same general model structure you saw in the first example: the ConvNet will be a stack of alternated `Conv2D` (with `relu` activation) and `MaxPooling2D` layers.

But because we’re dealing with bigger images and a more complex problem, we’ll make our model larger, accordingly: it will have two more `Conv2D MaxPooling2D` stages. This serves both to augment the capacity of the model and to further reduce the size of the feature maps so they aren’t overly large when we reach the pooling layer. Here, because we start from inputs of size 180 × 180 pixels (a somewhat arbitrary choice), we end up with feature maps of size 7 × 7 just before the `GlobalAveragePooling2D` layer. NOTE The depth of the feature maps progressively increases in the model (from 32 to 512), whereas the size of the feature maps decreases (from 180 × 180 to 7 × 7). This is a pattern you’ll see in almost all ConvNets.

Because we’re looking at a binary classification problem, we’ll end the model with a single unit (a `Dense` layer of size 1) and a `sigmoid` activation. This unit will encode the probability that the model is looking at one class or the other.

One last small difference: we will start t
