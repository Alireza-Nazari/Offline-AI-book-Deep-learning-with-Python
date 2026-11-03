# Generation bundle for page_320  (PDF page 320, printed page 296)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.2 Visualizing ConvNet filters
Subsection: 10.2.4 The filter visualization loop
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_320.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_320.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p320_e03] (heading_subsection) 10.2.4 The filter visualization loop
[p320_e04] (body) Now you have all the pieces. Let’s put them together into a Python function that takes a filter index as input and returns a tensor representing the pattern that maximizes the activation of the specified filter in our target layer.
[p320_e05] (listing_caption) Listing 10.14 Function to generate filter visualizations
[p320_e06] (code)
```
img_width = 200
img_height = 200
def generate_filter_pattern(filter_index):
    iterations = 30
    learning_rate = 10.0
    image = keras.random.uniform(
        minval=0.4, maxval=0.6, shape=(1, img_width, img_height, 3)
    )
    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0]
```
[p320_e07] (body) The resulting image tensor is a floating-point array of shape `(200, 200, 3)`, with values that may not be integers within `[0, 255]`. Hence, you need to post-process this tensor to turn it into a displayable image. You do so with the following straightforward utility function.
[p320_e08] (listing_caption) Listing 10.15 Utility function to convert a tensor into a valid image
[p320_e09] (code)
```
def deprocess_image(image):
    image -= ops.mean(image)
    image /= ops.std(image)
    image *= 64
    image += 128
    image = ops.clip(image, 0, 255)
    image = image[25:-25, 25:-25, :]
    image = ops.cast(image, dtype="uint8")
    return ops.convert_to_numpy(image)
```
[p320_e10] (body) Let’s try it (see figure 10.5):
[p320_e11] (code)
```
>>> plt.axis("off")
>>> plt.imshow(deprocess_image(generate_filter_pattern(filter_index=2)))
```

## Code annotations printed next to listings (callouts, in order)
- Initialize an image tensor with random values.
- (The Xception model expects input values in the [0, 1]
- range, so here we pick a range centered on 0.5.)
- The number of gradient
- ascent steps to apply
- The amplitude of a single step
- Repeatedly updates the values of the image
- tensor to maximize our loss function
- Normalizes image values
- within the [0, 255] range
- Center crop to avoid
- border artifacts

## Listings
- Listing 10.14: Function to generate filter visualizations (code elements p320_e06)
- Listing 10.15: Utility function to convert a tensor into a valid image (code elements p320_e09)

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

## Previous page (PDF 319) text, for continuity
10.2.2 Gradient ascent in PyTorch

In the case of PyTorch, we use `loss.backward()` and `image.grad` to obtain the gradients of the loss with respect to the input image, like this.

Listing 10.12 Loss maximization via stochastic gradient ascent: PyTorch

No need to reset the gradients since the image tensor is recreated at each iteration.

10.2.3 Gradient ascent in JAX

In the case of JAX, we use `jax.grad()` to obtain a function that returns the gradients of the loss with respect to the input image.

Listing 10.13 Loss maximization via stochastic gradient ascent: JAX

## Next page (PDF 321) text, for continuity
Figure 10.5 Pattern that the second channel in layer `block3_` `sepconv1` responds to maximally

It seems that filter 2 in layer `block3_sepconv1` is responsive to a horizontal lines pattern, somewhat water-like or fur-like.

Now the fun part: you can start visualizing every filter in the layer—and even every filter in every layer in the model (see figure 10.6).

Listing 10.16 Generating a grid of all filter response patterns

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.317 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.315 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.318 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.310 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.331 — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet
- PDF p.325 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.315 (printed 291) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2 Visualizing ConvNet filters
Another easy way to inspect the filters learned by ConvNets is to display the visual pattern that each filter is meant to respond to. This can be done with gradient ascent in input space, applying gradient descent to the value of the input image of a ConvNet so as to maximize the response of a specific filter, starting from a blank input image. The resulting input image will be one that the chosen filter is maximally responsive to.

Let’s try this with the filters of the Xception model. The process is simple: we’ll build a loss function that maximizes the value of a given filter in a given convolution layer, and then we’ll use stochastic gradient descent to adjust the values of the input image so as to maximize this activation value. This will be your second example of a low-level

### PDF p.317 (printed 293) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
Let’s use our feature extractor model to define a function that returns a scalar value quantifying how much a given input image “activates” a given filter in the layer. This is the loss function that we’ll maximize during the gradient ascent process:

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
wide array of techniques has been developed for visualizing and interpreting these representations. We won’t survey all of them, but we’ll cover three of the most accessible and useful ones:

Visualizing intermediate ConvNet outputs (intermediate activations)—Useful for understanding how successive ConvNet layers transform their input, and for getting a first idea of the meaning of individual ConvNet filters

Visualizing ConvNets filters—Useful for understanding precisely what visual pattern or concept each filter in a ConvNet is receptive to

Visualizing heatmaps of class activation in an image—Useful for understanding which parts of an image were identified as belonging to a given class, thus allowing you to localize objects in images

Visualizing the way ConvNets learn semantically organized manifolds of images—Useful for detecting outliers and getting a sense for what images are seen by your model as semantically ambiguous, thus allowing you to clean and improve your training dataset as well as refine your evaluation dataset

For the first method—activation visualization—you’ll use the small ConvNet that you trained from scratch on the dogs-versus-cats classification problem in

### PDF p.260 (printed 236) — 8 Image classification › 8.1 Introduction to convnets › 
Convolutions operate over rank-3 tensors, called feature maps, with two spatial axes (height and width) as well as a depth axis (also called the channels axis). For an RGB image, the dimension of the depth axis is 3, because the image has three color channels: red, green, and blue. For a black-and-white picture, like the MNIST digits, the depth is 1 (levels of gray). The convolution operation extracts patches from its input feature map and applies the same transformation to all of these patches, producing an output feature map. This output feature map is still a rank-3 tensor: it has a width and a height. Its depth can be arbitrary because the output depth is a parameter of the layer, and the different channels in that depth axis no longer stand for specific colors as in RGB input; rather, they stand for filters. Filters encode specific aspects of the input data: at a high level, a single filter could encode the concept “presence of a face in the input,” for instance.

In the MNIST example, the first convolution layer takes a feature map of size `(28, ` `28,` `1)` and outputs a feature map of size `(26,` `26,` `64)`: it computes 64 filters over its input. Each of these 64 output ch

### PDF p.322 (printed 298) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
Figure 10.6 Some filter patterns for layers `block2_sepconv1`, `block4_sepconv1`, and `block8_sepconv1`

These filter visualizations tell you a lot about how ConvNet layers see the world: each layer in a ConvNet learns a collection of filters such that their inputs can be expressed as a combination of the filters. This is similar to how the Fourier transform decomposes signals onto a bank of cosine functions. The filters in these ConvNet filter banks get increasingly complex and refined as you go higher in the model:

The filters from the first layers in the model encode simple directional edges and colors (or colored edges, in some cases).

The filters from layers a bit further up the stack, such as `block4_sepconv1`, encode simple textures made from combinations of edges and colors.

### PDF p.331 (printed 307) — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet › Summary
ConvNets process images by applying a set of learned filters. Filters from earlier layers detect edges and basic textures, while filters from later layers detect increasingly abstract concepts.

You can visualize both the pattern that a filter detects and a filter’s response map across an image.

You can use the Grad-CAM technique to visualize what area(s) in an image were responsible for a classifier’s decision.

Together, these techniques make ConvNets highly interpretable.

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 10.1 Visualizing intermediate activations
Visualizing intermediate activations consists of displaying the values returned by various convolution and pooling layers in a model, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. You want to visualize feature maps with three dimensions: width, height, and depth (channels). Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image. Let’s start by loading the model that you saved in section 8.2:

### PDF p.535 (printed 511) — 17 Image generation › 17.1 Deep learning for image generation › 
latent space of representations, `z_mean` and `z_log_variance`. 2 You randomly sample a point `z` from the latent normal distribution that’s assumed to generate the input image, via `z = z_mean + exp(z_log_variance) * epsilon`, where `epsilon` is a random tensor of small values. 3 A decoder module maps this point in the latent space back to the original input image. Because `epsilon` is random, the process ensures that every point that’s close to the latent location where you encoded `input_img` (`z-mean`) can be decoded to something
