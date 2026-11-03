# Generation bundle for page_310  (PDF page 310, printed page 286)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.1 Visualizing intermediate activations
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_310.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_310.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p310_e03] (code)
```
│ max_pooling2d_6 (MaxPooling2D)    │ (None, 89, 89, 32)       │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_12 (Conv2D)                │ (None, 87, 87, 64)       │        18,496 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ max_pooling2d_7 (MaxPooling2D)    │ (None, 43, 43, 64)       │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_13 (Conv2D)                │ (None, 41, 41, 128)      │        73,856 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ max_pooling2d_8 (MaxPooling2D)    │ (None, 20, 20, 128)      │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_14 (Conv2D)                │ (None, 18, 18, 256)      │       295,168 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ max_pooling2d_9 (MaxPooling2D)    │ (None, 9, 9, 256)        │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_15 (Conv2D)                │ (None, 7, 7, 512)        │     1,180,160 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ global_average_pooling2d_3        │ (None, 512)              │             0 │
│ (GlobalAveragePooling2D)          │                          │               │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dropout (Dropout)                 │ (None, 512)              │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_3 (Dense)                   │ (None, 1)                │           513 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 4,707,269 (17.96 MB)
 Trainable params: 1,569,089 (5.99 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 3,138,180 (11.97 MB)
```
[p310_e04] (body) Next, you’ll get an input image—a picture of a cat, not part of the images the network was trained on.
[p310_e05] (listing_caption) Listing 10.1 Preprocessing a single image
[p310_e06] (code)
```
import keras
import numpy as np
img_path = keras.utils.get_file(
    fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg"
)
def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array
img_tensor = get_img_array(img_path, target_size=(180, 180))
```

## Code annotations printed next to listings (callouts, in order)
- Downloads a test image
- Opens the image file and resizes it
- Turns the
- image into a
- float32 NumPy
- array of shape
- (180, 180, 3)
- We add a dimension to transform our
- array into a “batch” of a single sample.
- Its shape is now (1, 180, 180, 3).

## Listings
- Listing 10.1: Preprocessing a single image (code elements p310_e06)

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

## Previous page (PDF 309) text, for continuity
wide array of techniques has been developed for visualizing and interpreting these representations. We won’t survey all of them, but we’ll cover three of the most accessible and useful ones:

Visualizing intermediate ConvNet outputs (intermediate activations)—Useful for understanding how successive ConvNet layers transform their input, and for getting a first idea of the meaning of individual ConvNet filters

Visualizing ConvNets filters—Useful for understanding precisely what visual pattern or concept each filter in a ConvNet is receptive to

Visualizing heatmaps of class activation in an image—Useful for understanding which parts of an image were identified as belonging to a given class, thus allowing you to localize objects in images

Visualizing the way ConvNets learn semantically organized manifolds of images—Useful for detecting outliers and getting a sense for what images are seen by your model as semantically ambiguous, thus allowing you to clean and improve your training dataset as well as refine your evaluation dataset

For the first method—activation visualization—you’ll use the small ConvNet that you trained from scratch on the dogs-versus-cats classification problem in chapter 8. For the next two methods, you’ll use a pretrained Xception model.

10.1 Visualizing intermediate activations

Visualizing intermediate activations consists of displaying the values returned by various convolution and pooling layers in a model, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. You want to visualize feature maps with three dimensions: width, height, and depth (channels). Each channel encodes relatively …

## Next page (PDF 311) text, for continuity
Let’s display the picture (see figure 10.1).

Listing 10.2 Displaying the test picture

Figure 10.1 The test cat picture

To extract the feature maps you want to look at, you’ll create a Keras model that takes batches of images as input and outputs the activations of all convolution and pooling layers.

Listing 10.3 Instantiating a model that returns layer activations

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.257 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.270 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.300 — 9 ConvNet architecture patterns › 9.3 Batch normalization
- PDF p.271 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.299 — 9 ConvNet architecture patterns › 9.2 Residual connections

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.314 (printed 290) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
Figure 10.3 Every channel of every layer activation on the test cat picture

There are a few things to note here:

The first layer acts as a collection of various edge detectors. At that stage, the activations retain almost all of the information present in the initial picture.

As you go higher, the activations become increasingly abstract and less visually interpretable. They begin to encode higher-level concepts such as “cat ear” and “cat eye.” Higher representations carry increasingly less information about the visual contents of the image and increasingly more information related to the class of the image.

The sparsity of the activations increases with the depth of the layer: in the first layer, all filters are activated by the input image, but in the following layers, more and more filters are blank. This means the pattern encoded by the filter isn’t found in the input image.

### PDF p.312 (printed 288) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
For instance, this is the activation of the first convolution layer for the cat image input:

### PDF p.325 (printed 301) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 
First, we create a model that maps the input image to the activations of the last convolutional layer.

Listing 10.18 Returning the last convolutional output

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Listing 10.17 Preprocessing an input image for Xception
img_path = keras.utils.get_file(
    fname="elephant.jpg",
    origin="https://img-datasets.s3.amazonaws.com/elephant.jpg",
)
img = keras.utils.load_img(img_path)
img_array = np.expand_dims(img, axis=0)

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Here’s one last visualization technique—one that is useful for understanding which parts of a given image led a ConvNet to its final classification decision. This is helpful for “debugging” the decision process of a ConvNet, particularly in the case of a classification mistake (a problem domain called model interpretability). It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM) visualization, and it consists of producing heatmaps of class activation over input images. A class activation heatmap is a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration. For instance, given an image fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate a heatmap for the class “cat,” indicating how cat-like different parts of the image are, and also a heatmap for the class “dog,” indicating how dog-like parts of the image are. The specific implementation we’ll use is the one described in Selvaraju et al.1 Grad-CAM consists of taking the output feature 

### PDF p.317 (printed 293) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
To use this model, we can simply call it on some input data, but we should be careful to apply our model-specific image preprocessing so that our images are scaled to the same range as the Xception pretraining data.

Listing 10.10 Using the feature extractor

### PDF p.312 (printed 288) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
Figure 10.2 Sixth channel of the activation of the first layer on the test cat picture

### PDF p.329 (printed 305) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 
Finally, let’s generate an image that superimposes the original image on the heatmap you just obtained (see figure 10.9).

Figure 10.9 African elephant class activation heatmap over the test picture

Listing 10.25 Superimposing the heatmap with the original picture
