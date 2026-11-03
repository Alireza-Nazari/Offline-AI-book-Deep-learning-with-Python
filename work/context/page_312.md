# Generation bundle for page_312  (PDF page 312, printed page 288)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.1 Visualizing intermediate activations
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_312.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_312.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p312_e03] (body) When fed an image input, this model returns the values of the layer activations in the original model, as a list. This is the first time you’ve encountered a multi-output model in this book in practice since you learned about them in chapter 7: until now, the models you’ve seen have had exactly one input and one output. This one has one input and nine outputs—one output per layer activation.
[p312_e04] (listing_caption) Listing 10.4 Using the model to compute layer activations
[p312_e05] (code)
```
activations = activation_model.predict(img_tensor)
```
[p312_e06] (body) For instance, this is the activation of the first convolution layer for the cat image input:
[p312_e07] (code)
```
>>> first_layer_activation = activations[0]
>>> print(first_layer_activation.shape)
(1, 178, 178, 32)
```
[p312_e08] (body) It’s a 178 × 178 feature map with 32 channels. Let’s try plotting the sixth channel of the activation of the first layer of the original model (see figure 10.2).
[p312_e09] (listing_caption) Listing 10.5 Visualizing the sixth channel
[p312_e10] (code)
```
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 5], cmap="viridis")
```
[p312_e11] (figure_caption) Figure 10.2 Sixth channel of the activation of the first layer on the test cat picture

## Code annotations printed next to listings (callouts, in order)
- Returns a list of nine
- NumPy arrays—one
- array per layer
- activation

## Listings
- Listing 10.4: Using the model to compute layer activations (code elements p312_e05)
- Listing 10.5: Visualizing the sixth channel (code elements p312_e10)

## Figures
- Figure 10.2: Sixth channel of the activation of the first layer on the test cat picture  | labels inside the figure: (none - image)

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

## Previous page (PDF 311) text, for continuity
Let’s display the picture (see figure 10.1).

Listing 10.2 Displaying the test picture

Figure 10.1 The test cat picture

To extract the feature maps you want to look at, you’ll create a Keras model that takes batches of images as input and outputs the activations of all convolution and pooling layers.

Listing 10.3 Instantiating a model that returns layer activations

## Next page (PDF 313) text, for continuity
This channel appears to encode a diagonal edge detector, but note that your own channels may vary because the specific filters learned by convolution layers aren’t deterministic.

Now let’s plot a complete visualization of all the activations in the network (see figure 10.3). We’ll extract and plot every channel in each of the layer activations, and we’ll stack the results in one big grid, with channels stacked side by side.

Listing 10.6 Visualizing every channel in every intermediate activation

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.325 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.314 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.326 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.315 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 10.1 Visualizing intermediate activations
Visualizing intermediate activations consists of displaying the values returned by various convolution and pooling layers in a model, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. You want to visualize feature maps with three dimensions: width, height, and depth (channels). Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image. Let’s start by loading the model that you saved in section 8.2:

### PDF p.314 (printed 290) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
Figure 10.3 Every channel of every layer activation on the test cat picture

There are a few things to note here:

The first layer acts as a collection of various edge detectors. At that stage, the activations retain almost all of the information present in the initial picture.

As you go higher, the activations become increasingly abstract and less visually interpretable. They begin to encode higher-level concepts such as “cat ear” and “cat eye.” Higher representations carry increasingly less information about the visual contents of the image and increasingly more information related to the class of the image.

The sparsity of the activations increases with the depth of the layer: in the first layer, all filters are activated by the input image, but in the following layers, more and more filters are blank. This means the pattern encoded by the filter isn’t found in the input image.

### PDF p.325 (printed 301) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 
First, we create a model that maps the input image to the activations of the last convolutional layer.

Listing 10.18 Returning the last convolutional output

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Here’s one last visualization technique—one that is useful for understanding which parts of a given image led a ConvNet to its final classification decision. This is helpful for “debugging” the decision process of a ConvNet, particularly in the case of a classification mistake (a problem domain called model interpretability). It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM) visualization, and it consists of producing heatmaps of class activation over input images. A class activation heatmap is a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration. For instance, given an image fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate a heatmap for the class “cat,” indicating how cat-like different parts of the image are, and also a heatmap for the class “dog,” indicating how dog-like parts of the image are. The specific implementation we’ll use is the one described in Selvaraju et al.1 Grad-CAM consists of taking the output feature 

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 10.1 Visualizing intermediate activations
>>> import keras
>>> model = keras.models.load_model(
...     "convnet_from_scratch_with_augmentation.keras"
... )
>>> model.summary()
Model: "functional_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_3 (InputLayer)        │ (None, 180, 180, 3)      │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ rescaling_1 (Rescaling)           │ (None, 180, 180, 3)      │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_11 (Conv2D)                │ (None, 178, 178, 32)     │           896 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤

### PDF p.262 (printed 238) — 8 Image classification › 8.1 Introduction to convnets › Understanding border effects and padding
Consider a 5 × 5 feature map (25 tiles total). There are only 9 tiles around which you can center a 3 × 3 window, forming a 3 × 3 grid (see figure 8.5). Hence, the output feature map will be 3 × 3. It shrinks a little: by exactly two tiles alongside each dimension, in this case. You can see this border effect in action in the earlier example: you start with 28 × 28 inputs, which become 26 × 26 after the first convolution layer.

Figure 8.5 Valid locations of 3 × 3 patches in a 5 × 5 input feature map

If you want to get an output feature map with the same spatial dimensions as the input, you can use padding. Padding consists of adding an appropriate number of rows and columns on each side of the input feature map so as to make it possible to fit centered convolution windows around every input tile. For a 3 × 3 window, you add one column on the right, one column on the left, one row at the top, and one row at the bottom. For a 5 × 5 window, you add two rows (see figure 8.6).

Figure 8.6 Padding a 5 × 5 input to be able to extract 25 3 × 3 patches

### PDF p.339 (printed 315) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
The second half of the model is a stack of `Conv2DTranspose` layers. What are those? Well, the output of the first half of the model is a feature map of shape `(25,` `25,` `256)`, but we want our final output to predict a class for each pixel, matching the original spatial dimensions. The final model output will have shape `(200, 200, num_classes)`, which is `(200,` `200,` `3)` here. Therefore, we need to apply a kind of inverse of the transformations we’ve applied so far, something that will upsample the feature maps instead of downsampling them. That’s the purpose of the `Conv2DTranspose` layer: you can think of it as a kind of convolution layer that learns to upsample. If you have an input of shape `(100,` `100,` `64)` and you run it through the layer `Conv2D(128,` `3,` `strides=2, ` `padding="same")`, you get an output of shape `(50,` `50,` `128)`. If you run this output through the layer `Conv2DTranspose(64,` `3,` `strides=2,` `padding="same")`, you get back an output of shape `(100,` `100,` `64)`, the same as the original. So after compressing our inputs into feature maps of shape `(25,` `25,` `256)` via a stack of `Conv2D` layers, we can

### PDF p.258 (printed 234) — 8 Image classification › 8.1 Introduction to convnets › 
After the last `Conv2D` layer, we end up with an output of shape `(3, 3, 256)`—a 3 × 3 feature map of 256 channels. The next step is to feed this output into a densely connected classifier like those you’re already familiar with: a stack of `Dense` layers. These classifiers process vectors, which are 1D, whereas the current output is a rank-3 tensor. To bridge the gap, we flatten the 3D outputs to 1D with a `GlobalAveragePooling2D` layer before adding the `Dense` layers. This layer will take the average of each 3 × 3 feature map in the tensor of shape `(3,` `3,` `256)`, resulting in an output vector of shape `(256,)`. Finally, we’ll do 10-way classification, so our last layer has 10 outputs and a softmax activation.

Now, let’s train the ConvNet on the MNIST digits. We’ll reuse a lot of the code from the MNIST example in chapter 2. Because we’re doing 10-way classification with a softmax output, we’ll use the categorical crossentropy loss, and because our labels are integers, we’ll use the sparse version, `sparse_categorical_crossentropy`.

Listing 8.3 Training the ConvNet on MNIST images
