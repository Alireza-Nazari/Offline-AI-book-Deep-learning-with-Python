# Generation bundle for page_313  (PDF page 313, printed page 289)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.1 Visualizing intermediate activations
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_313.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_313.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p313_e03] (body) This channel appears to encode a diagonal edge detector, but note that your own channels may vary because the specific filters learned by convolution layers aren’t deterministic.
[p313_e04] (body) Now let’s plot a complete visualization of all the activations in the network (see figure 10.3). We’ll extract and plot every channel in each of the layer activations, and we’ll stack the results in one big grid, with channels stacked side by side.
[p313_e05] (listing_caption) Listing 10.6 Visualizing every channel in every intermediate activation
[p313_e06] (code)
```
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros(
        ((size + 1) * n_cols - 1, images_per_row * (size + 1) - 1)
    )
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy()
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[
                col * (size + 1) : (col + 1) * size + col,
                row * (size + 1) : (row + 1) * size + row,
            ] = channel_image
    scale = 1.0 / size
    plt.figure(
        figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0])
    )
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")
```

## Code annotations printed next to listings (callouts, in order)
- Prepares an empty grid for displaying
- all the channels in this activation
- The layer activation has shape
- (1, size, size, n_features).
- Iterates over the activations
- (and the names of the
- corresponding layers)
- This is a single channel (or feature).
- Normalizes channel
- values within the [0, 255]
- range. All-zero channels
- are kept at zero.
- Places the channel
- matrix in the empty
- grid we prepared
- Displays the
- grid for the layer

## Listings
- Listing 10.6: Visualizing every channel in every intermediate activation (code elements p313_e06)

## Chapter outline
Chapter: 10 Interpreting what ConvNets learn (PDF pages 308-331)
  - 10.1 Visualizing intermediate activations (p.309)
  - 10.2 Visualizing ConvNet filters (p.315)
      - 10.2.1 Gradient ascent in TensorFlow (p.318)
      - 10.2.2 Gradient ascent in PyTorch (p.319)
      - 10.2.3 Gradient ascent in JAX (p.319)
      - 10.2.4 The filter visualization loop (p.320)
  - 10.3 Visualizing heatmaps of class activation (p.323)
      - 10.3.1 Getting the gradient of the top class: TensorFlow version (p.326)
      - 10.3.2 Getting the gradient of the top class: PyTorch version (p.326)
      - 10.3.3 Getting the gradient of the top class: JAX version (p.327)
      - 10.3.4 Displaying the class activation heatmap (p.328)
  - 10.4 Visualizing the latent space of a ConvNet (p.330)

## Previous page (PDF 312) text, for continuity
When fed an image input, this model returns the values of the layer activations in the original model, as a list. This is the first time you’ve encountered a multi-output model in this book in practice since you learned about them in chapter 7: until now, the models you’ve seen have had exactly one input and one output. This one has one input and nine outputs—one output per layer activation.

Listing 10.4 Using the model to compute layer activations

For instance, this is the activation of the first convolution layer for the cat image input:

It’s a 178 × 178 feature map with 32 channels. Let’s try plotting the sixth channel of the activation of the first layer of the original model (see figure 10.2).

Listing 10.5 Visualizing the sixth channel

Figure 10.2 Sixth channel of the activation of the first layer on the test cat picture

## Next page (PDF 314) text, for continuity
Figure 10.3 Every channel of every layer activation on the test cat picture

There are a few things to note here:

The first layer acts as a collection of various edge detectors. At that stage, the activations retain almost all of the information present in the initial picture.

As you go higher, the activations become increasingly abstract and less visually interpretable. They begin to encode higher-level concepts such as “cat ear” and “cat eye.” Higher representations carry increasingly less information about the visual contents of the image and increasingly more information related to the class of the image.

The sparsity of the activations increases with the depth of the layer: in the first layer, all filters are activated by the input image, but in the following layers, more and more filters are blank. This means the pattern encoded by the filter isn’t found in the input image.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.321 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.311 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.269 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.328 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.260 — 8 Image classification › 8.1 Introduction to convnets

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 10.1 Visualizing intermediate activations
Visualizing intermediate activations consists of displaying the values returned by various convolution and pooling layers in a model, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. You want to visualize feature maps with three dimensions: width, height, and depth (channels). Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image. Let’s start by loading the model that you saved in section 8.2:

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
wide array of techniques has been developed for visualizing and interpreting these representations. We won’t survey all of them, but we’ll cover three of the most accessible and useful ones:

Visualizing intermediate ConvNet outputs (intermediate activations)—Useful for understanding how successive ConvNet layers transform their input, and for getting a first idea of the meaning of individual ConvNet filters

Visualizing ConvNets filters—Useful for understanding precisely what visual pattern or concept each filter in a ConvNet is receptive to

Visualizing heatmaps of class activation in an image—Useful for understanding which parts of an image were identified as belonging to a given class, thus allowing you to localize objects in images

Visualizing the way ConvNets learn semantically organized manifolds of images—Useful for detecting outliers and getting a sense for what images are seen by your model as semantically ambiguous, thus allowing you to clean and improve your training dataset as well as refine your evaluation dataset

For the first method—activation visualization—you’ll use the small ConvNet that you trained from scratch on the dogs-versus-cats classification problem in

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Here’s one last visualization technique—one that is useful for understanding which parts of a given image led a ConvNet to its final classification decision. This is helpful for “debugging” the decision process of a ConvNet, particularly in the case of a classification mistake (a problem domain called model interpretability). It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM) visualization, and it consists of producing heatmaps of class activation over input images. A class activation heatmap is a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration. For instance, given an image fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate a heatmap for the class “cat,” indicating how cat-like different parts of the image are, and also a heatmap for the class “dog,” indicating how dog-like parts of the image are. The specific implementation we’ll use is the one described in Selvaraju et al.1 Grad-CAM consists of taking the output feature 

### PDF p.302 (printed 278) — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions › Depthwise separable convolutions
What if we told you that there’s a layer you can use as a drop-in replacement for `Conv2D` that will make your model smaller (fewer trainable weight parameters), leaner (fewer floating-point operations), and cause it to perform a few percentage points better on its task? That is precisely what the depthwise separable convolution layer does (`Separable` `Conv2D` in Keras). This layer performs a spatial convolution on each channel of its input, independently, before mixing output channels via a pointwise convolution (a 1 × 1 convolution), as shown in figure 9.4.

This is equivalent to separating the learning of spatial features and the learning of channel-wise features. In much the same way that convolution relies on the assumption that the patterns in images are not tied to specific locations, depthwise separable convolution relies on the assumption that spatial locations in intermediate activations are highly correlated, but different channels are highly independent. Because this assumption is generally true for the image representations learned by deep

### PDF p.260 (printed 236) — 8 Image classification › 8.1 Introduction to convnets › 
Convolutions operate over rank-3 tensors, called feature maps, with two spatial axes (height and width) as well as a depth axis (also called the channels axis). For an RGB image, the dimension of the depth axis is 3, because the image has three color channels: red, green, and blue. For a black-and-white picture, like the MNIST digits, the depth is 1 (levels of gray). The convolution operation extracts patches from its input feature map and applies the same transformation to all of these patches, producing an output feature map. This output feature map is still a rank-3 tensor: it has a width and a height. Its depth can be arbitrary because the output depth is a parameter of the layer, and the different channels in that depth axis no longer stand for specific colors as in RGB input; rather, they stand for filters. Filters encode specific aspects of the input data: at a high level, a single filter could encode the concept “presence of a face in the input,” for instance.

In the MNIST example, the first convolution layer takes a feature map of size `(28, ` `28,` `1)` and outputs a feature map of size `(26,` `26,` `64)`: it computes 64 filters over its input. Each of these 64 output ch

### PDF p.311 (printed 287) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
Figure 10.1 The test cat picture

To extract the feature maps you want to look at, you’ll create a Keras model that takes batches of images as input and outputs the activations of all convolution and pooling layers.

Listing 10.3 Instantiating a model that returns layer activations

### PDF p.322 (printed 298) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
Figure 10.6 Some filter patterns for layers `block2_sepconv1`, `block4_sepconv1`, and `block8_sepconv1`

These filter visualizations tell you a lot about how ConvNet layers see the world: each layer in a ConvNet learns a collection of filters such that their inputs can be expressed as a combination of the filters. This is similar to how the Fourier transform decomposes signals onto a bank of cosine functions. The filters in these ConvNet filter banks get increasingly complex and refined as you go higher in the model:

The filters from the first layers in the model encode simple directional edges and colors (or colored edges, in some cases).

The filters from layers a bit further up the stack, such as `block4_sepconv1`, encode simple textures made from combinations of edges and colors.

### PDF p.321 (printed 297) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
Figure 10.5 Pattern that the second channel in layer `block3_` `sepconv1` responds to maximally

It seems that filter 2 in layer `block3_sepconv1` is responsive to a horizontal lines pattern, somewhat water-like or fur-like.

Now the fun part: you can start visualizing every filter in the layer—and even every filter in every layer in the model (see figure 10.6).

Listing 10.16 Generating a grid of all filter response patterns
