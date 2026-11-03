# Generation bundle for page_260  (PDF page 260, printed page 236)

Page kind: content
Chapter: 8 Image classification
Section: 8.1 Introduction to convnets
Subsection: 8.1.1 The convolution operation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_260.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_260.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p260_e03] (body) Convolutions operate over rank-3 tensors, called feature maps, with two spatial axes (height and width) as well as a depth axis (also called the channels axis). For an RGB image, the dimension of the depth axis is 3, because the image has three color channels: red, green, and blue. For a black-and-white picture, like the MNIST digits, the depth is 1 (levels of gray). The convolution operation extracts patches from its input feature map and applies the same transformation to all of these patches, producing an output feature map. This output feature map is still a rank-3 tensor: it has a width and a height. Its depth can be arbitrary because the output depth is a parameter of the layer, and the different channels in that depth axis no longer stand for specific colors as in RGB input; rather, they stand for filters. Filters encode specific aspects of the input data: at a high level, a single filter could encode the concept “presence of a face in the input,” for instance.
[p260_e04] (body) In the MNIST example, the first convolution layer takes a feature map of size `(28, ` `28,` `1)` and outputs a feature map of size `(26,` `26,` `64)`: it computes 64 filters over its input. Each of these 64 output channels contains a 26 × 26 grid of values, which is a response map of the filter over the input, indicating the response of that filter pattern at different locations in the input (see figure 8.3). That is what the term feature map means: every dimension in the depth axis is a feature (or filter), and the rank-2 tensor `output[:, :, n]` is the 2D spatial map of the response of this filter over the input.
[p260_e05] (figure_caption) Figure 8.3 The concept of a response map: a 2D map of the presence of a pattern at different locations in an input
[p260_e06] (body) Convolutions are defined by two key parameters:
[p260_e07] (bullet) Size of the patches extracted from the inputs—These are typically 3 × 3 or 5 × 5. In the example, they were 3 × 3, which is a common choice.
[p260_e08] (bullet) Depth of the output feature map—The number of filters computed by the convolution. The example started with a depth of 32 and ended with a depth of 64.
[p260_e09] (body) In Keras `Conv2D` layers, these parameters are the first arguments passed to the layer:
[p260_e10] (code)
```
Conv2D(output_depth, (window_height, window_width)).
```
[p260_e11] (body) A convolution works by sliding these windows of size 3 × 3 or 5 × 5 over the 3D input feature map, stopping at every possible location, and extracting the 3D patch of surrounding features of shape (`(window_height,` `window_width,` `input_depth)`). Each such 3D patch is then transformed into a 1D vector of shape `(output_depth,)`, which is

## Figures
- Figure 8.3: The concept of a response map: a 2D map of the presence of a pattern at different locations in an input  | labels inside the figure: different locations, Original input, Single filter

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

## Previous page (PDF 259) text, for continuity
for a MNIST digit, patterns involving all pixels), whereas convolution layers learn local patterns (see figure 8.1): in the case of images, patterns found in small 2D windows of the inputs. In the previous example, these windows were all 3 × 3. This key characteristic gives ConvNets two interesting properties:

The patterns they learn are translation invariant. After learning a certain pattern in the lower-right corner of a picture, a ConvNet can recognize it anywhere: for example, in the upper-left corner. A densely connected model would have to learn the pattern anew if it appeared at a new location. This makes ConvNets data efficient when pro

Figure 8.1 Images can be broken into

cessing images—because the visual world

local patterns such as edges, textures,

is fundamentally translation invariant. They

and so on.

need fewer training samples to learn representations that have generalization power.

They can learn spatial hierarchies of patterns (see figure 8.2). A first convolution layer will learn small local patterns such as edges, a second convolution layer will learn larger patterns made of the features of the first layers, and so on. This allows ConvNets to efficiently learn increasingly complex and abstract visual concepts—

because the visual world is fundamentally spatially hierarchical.

Figure 8.2 The visual world forms a spatial hierarchy of visual modules: elementary lines or textures combine into simple objects such as eyes or ears, which combine into high-level concepts such as “cat.”

## Next page (PDF 261) text, for continuity
done via a tensor product with a learned weight matrix, called the convolution kernel— the same kernel is reused across every patch. All of these vectors (one per patch) are then spatially reassembled into a 3D output map of shape `(height,` `width,` `output_` `depth)`. Every spatial location in the output feature map corresponds to the same location in the input feature map (for example, the lower-right corner of the output contains information about the lower-right corner of the input). For instance, with 3 × 3 windows, the vector `output[i, j, :]` comes from the 3D patch `input[i-1:i+1, ` `j-1:j+1, :]`. The full process is detailed in figure 8.4.

Figure 8.4 How convolution works

Note that the output width and height may differ from the input width and height. They may differ for two reasons:

Border effects, which can be countered by padding the input feature map

The use of strides, which we’ll define in a second

Let’s take a deeper look at these notions.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.263 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.339 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.265 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.262 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.257 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.269 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.262 (printed 238) — 8 Image classification › 8.1 Introduction to convnets › Understanding border effects and padding
Consider a 5 × 5 feature map (25 tiles total). There are only 9 tiles around which you can center a 3 × 3 window, forming a 3 × 3 grid (see figure 8.5). Hence, the output feature map will be 3 × 3. It shrinks a little: by exactly two tiles alongside each dimension, in this case. You can see this border effect in action in the earlier example: you start with 28 × 28 inputs, which become 26 × 26 after the first convolution layer.

Figure 8.5 Valid locations of 3 × 3 patches in a 5 × 5 input feature map

If you want to get an output feature map with the same spatial dimensions as the input, you can use padding. Padding consists of adding an appropriate number of rows and columns on each side of the input feature map so as to make it possible to fit centered convolution windows around every input tile. For a 3 × 3 window, you add one column on the right, one column on the left, one row at the top, and one row at the bottom. For a 5 × 5 window, you add two rows (see figure 8.6).

Figure 8.6 Padding a 5 × 5 input to be able to extract 25 3 × 3 patches

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
the initial input. The high-level patterns learned by the ConvNet will still be very small with regard to the initial input, which may not be enough to learn to classify digits (try recognizing a digit by only looking at it through windows that are 7 × 7 pixels!). We need the features from the last convolution layer to contain information about the totality of the input.

The final feature map has dimensions 22 × 22. That’s huge—when you take the average of each 22 × 22 feature map, you are going to be destroying a lot of information compared to when your feature maps were only 3 × 3.

In short, the reason to use downsampling is to reduce the size of the feature maps, making the information they contain increasingly less spatially distributed and increasingly contained in the channels, while also inducing spatial-filter hierarchies by making successive convolution layers “look” at increasingly large windows (in terms of the fraction of the original input image they cover).

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 10.1 Visualizing intermediate activations
Visualizing intermediate activations consists of displaying the values returned by various convolution and pooling layers in a model, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. You want to visualize feature maps with three dimensions: width, height, and depth (channels). Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image. Let’s start by loading the model that you saved in section 8.2:

### PDF p.626 (printed 602) — 20 Conclusions › 20.1 Key concepts in review › ConvNets
Convolution layers look at spatially local patterns by applying the same geometric transformation to different spatial locations (patches) in an input tensor. This results in representations that are translation invariant, making convolution layers highly data efficient and modular. This idea is applicable to spaces of any dimensionality: 1D (continuous sequences), 2D (images), 3D (volumes), and so on. You can use the `Conv1D` layer to process sequences, the `Conv2D` layer to process images, and the `Conv3D` layer to process volumes. As a leaner, more efficient alternative to convolution layers, you can also use depthwise separable convolution layers, such as `SeparableConv2D`.

ConvNets, or convolutional networks, consist of stacks of convolution and max-pooling layers. The pooling layers let you spatially downsample the data, which is required to keep feature maps to a reasonable size as the number of features grows and to allow subsequent convolution layers to “see” a greater spatial extent of the inputs. ConvNets are often ended with either a `Flatten` operation or a global pooling layer, turning spatial feature maps into vectors, followed by `Dense` layers to achieve classific

### PDF p.258 (printed 234) — 8 Image classification › 8.1 Introduction to convnets › 
After the last `Conv2D` layer, we end up with an output of shape `(3, 3, 256)`—a 3 × 3 feature map of 256 channels. The next step is to feed this output into a densely connected classifier like those you’re already familiar with: a stack of `Dense` layers. These classifiers process vectors, which are 1D, whereas the current output is a rank-3 tensor. To bridge the gap, we flatten the 3D outputs to 1D with a `GlobalAveragePooling2D` layer before adding the `Dense` layers. This layer will take the average of each 3 × 3 feature map in the tensor of shape `(3,` `3,` `256)`, resulting in an output vector of shape `(256,)`. Finally, we’ll do 10-way classification, so our last layer has 10 outputs and a softmax activation.

Now, let’s train the ConvNet on the MNIST digits. We’ll reuse a lot of the code from the MNIST example in chapter 2. Because we’re doing 10-way classification with a softmax output, we’ll use the categorical crossentropy loss, and because our labels are integers, we’ll use the sparse version, `sparse_categorical_crossentropy`.

Listing 8.3 Training the ConvNet on MNIST images

### PDF p.51 (printed 27) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Image data
Images typically have three dimensions: height, width, and color depth. Although grayscale images (like our MNIST digits) have only a single color channel and could thus be stored in rank-2 tensors, by convention image tensors are always rank-3, with a one-dimensional color channel for grayscale images. A batch of 128 grayscale images of size 256 × 256 could thus be stored in a tensor of shape `(128,` `256,` `256, 1)`, and a batch of 128 color images could be stored in a tensor of shape `(128,` `256,` `256,` `3)` (see figure 2.4).

There are two conventions for the shapes of image tensors: the channels-last convention (which is standard in JAX and TensorFlow, as well as most other deep learning tools out there) and the channels-first convention (which is standard in PyTorch). The channels-last convention places the color-depth axis at the end: `(samples,` `height,` `width,` `color_depth)`. Meanwhile, the channels-first convention places the color depth axis right after the batch axis: `(samples,` `color_depth,` `height,` `width)`.

### PDF p.339 (printed 315) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
The second half of the model is a stack of `Conv2DTranspose` layers. What are those? Well, the output of the first half of the model is a feature map of shape `(25,` `25,` `256)`, but we want our final output to predict a class for each pixel, matching the original spatial dimensions. The final model output will have shape `(200, 200, num_classes)`, which is `(200,` `200,` `3)` here. Therefore, we need to apply a kind of inverse of the transformations we’ve applied so far, something that will upsample the feature maps instead of downsampling them. That’s the purpose of the `Conv2DTranspose` layer: you can think of it as a kind of convolution layer that learns to upsample. If you have an input of shape `(100,` `100,` `64)` and you run it through the layer `Conv2D(128,` `3,` `strides=2, ` `padding="same")`, you get an output of shape `(50,` `50,` `128)`. If you run this output through the layer `Conv2DTranspose(64,` `3,` `strides=2,` `padding="same")`, you get back an output of shape `(100,` `100,` `64)`, the same as the original. So after compressing our inputs into feature maps of shape `(25,` `25,` `256)` via a stack of `Conv2D` layers, we can

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Here’s one last visualization technique—one that is useful for understanding which parts of a given image led a ConvNet to its final classification decision. This is helpful for “debugging” the decision process of a ConvNet, particularly in the case of a classification mistake (a problem domain called model interpretability). It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM) visualization, and it consists of producing heatmaps of class activation over input images. A class activation heatmap is a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration. For instance, given an image fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate a heatmap for the class “cat,” indicating how cat-like different parts of the image are, and also a heatmap for the class “dog,” indicating how dog-like parts of the image are. The specific implementation we’ll use is the one described in Selvaraju et al.1 Grad-CAM consists of taking the output feature 
