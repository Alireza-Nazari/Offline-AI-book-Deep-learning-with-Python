# Generation bundle for page_329  (PDF page 329, printed page 305)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.3 Visualizing heatmaps of class activation
Subsection: 10.3.4 Displaying the class activation heatmap
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_329.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_329.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p329_e03] (body) Finally, let’s generate an image that superimposes the original image on the heatmap you just obtained (see figure 10.9).
[p329_e04] (figure_caption) Figure 10.9 African elephant class activation heatmap over the test picture
[p329_e05] (listing_caption) Listing 10.25 Superimposing the heatmap with the original picture
[p329_e06] (code)
```
import matplotlib.cm as cm
img = keras.utils.load_img(img_path)
img = keras.utils.img_to_array(img)
heatmap = np.uint8(255 * heatmap)
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]
jet_heatmap = keras.utils.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.utils.img_to_array(jet_heatmap)
```

## Code annotations printed next to listings (callouts, in order)
- Loads the original image
- Rescales the heatmap to the range 0–255
- Uses the “jet” colormap to
- recolorize the heatmap
- Creates an image that contains
- the recolorized heatmap

## Listings
- Listing 10.25: Superimposing the heatmap with the original picture (code elements p329_e06)

## Figures
- Figure 10.9: African elephant class activation heatmap over the test picture  | labels inside the figure: (none - image)

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

## Previous page (PDF 328) text, for continuity
10.3.4 Displaying the class activation heatmap

Now, we apply pooling and importance weighting to the gradient tensor to obtain our heatmap of class activation.

Listing 10.23 Gradient pooling and channel importance weighting

For visualization purposes, you’ll also normalize the heatmap between 0 and 1. The result is shown in figure 10.8.

Listing 10.24 Heatmap post-processing

Figure 10.8 Standalone class activation heatmap

## Next page (PDF 330) text, for continuity
This visualization technique answers two important questions:

Why did the network think this image contained an African elephant?

Where is the African elephant located in the picture?

In particular, it’s interesting to note that the ears of the elephant calf are strongly activated: this is probably how the network can tell the difference between African and Indian elephants.

10.4 Visualizing the latent space of a ConvNet

I’ll introduce one last model inspection technique: latent space visualization. All deep learning models work by embedding their inputs in a latent space, a manifold (or curve) where points that are spatially close to each other represent semantically similar inputs. On this manifold, different classes of inputs become nicely separable, which is what enables classification.

Can we visualize in two dimensions what this manifold looks like? The general idea is to pick a certain layer in the model, typically the one right before the classification layer, and record its activations for a set of inputs. The activations for each input represent the coordinates of the corresponding point on the latent manifold. They’re points on a curve. To visualize that curve—the latent manifold—all we need to do is to plot these points. Note that this process can be done with any layer in your model—each layer encodes a different latent space, and these latent spaces get gradually more semantically organized as your go up the model. Remember our analogy about how deep learning models are machines for uncrumpling paper balls? The latent space of each layer is a snapshot of the paper ball at different stages of its uncrumpling. This is why it’s most interesting to target the second-before-last layer, when the paper ball is almost fully flattened into a nice sheet. Now t …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.310 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.311 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.325 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.545 — 17 Image generation › 17.2 Diffusion models
- PDF p.312 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.324 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Here’s one last visualization technique—one that is useful for understanding which parts of a given image led a ConvNet to its final classification decision. This is helpful for “debugging” the decision process of a ConvNet, particularly in the case of a classification mistake (a problem domain called model interpretability). It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM) visualization, and it consists of producing heatmaps of class activation over input images. A class activation heatmap is a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration. For instance, given an image fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate a heatmap for the class “cat,” indicating how cat-like different parts of the image are, and also a heatmap for the class “dog,” indicating how dog-like parts of the image are. The specific implementation we’ll use is the one described in Selvaraju et al.1 Grad-CAM consists of taking the output feature 

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Listing 10.17 Preprocessing an input image for Xception
img_path = keras.utils.get_file(
    fname="elephant.jpg",
    origin="https://img-datasets.s3.amazonaws.com/elephant.jpg",
)
img = keras.utils.load_img(img_path)
img_array = np.expand_dims(img, axis=0)

### PDF p.364 (printed 340) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Let’s visualize our YOLO training data with our box drawing helpers (figure 12.5). We will draw the entire class activation map over our first input image4 and add the confidence score of a box along with its label.

4 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm9.staticflickr.com /8081/8387882360_5b97a233c4_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

### PDF p.311 (printed 287) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
Listing 10.2 Displaying the test picture
import matplotlib.pyplot as plt
plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
wide array of techniques has been developed for visualizing and interpreting these representations. We won’t survey all of them, but we’ll cover three of the most accessible and useful ones:

Visualizing intermediate ConvNet outputs (intermediate activations)—Useful for understanding how successive ConvNet layers transform their input, and for getting a first idea of the meaning of individual ConvNet filters

Visualizing ConvNets filters—Useful for understanding precisely what visual pattern or concept each filter in a ConvNet is receptive to

Visualizing heatmaps of class activation in an image—Useful for understanding which parts of an image were identified as belonging to a given class, thus allowing you to localize objects in images

Visualizing the way ConvNets learn semantically organized manifolds of images—Useful for detecting outliers and getting a sense for what images are seen by your model as semantically ambiguous, thus allowing you to clean and improve your training dataset as well as refine your evaluation dataset

For the first method—activation visualization—you’ll use the small ConvNet that you trained from scratch on the dogs-versus-cats classification problem in

### PDF p.325 (printed 301) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 
To visualize which parts of the image are the most African elephant–like, let’s set up the Grad-CAM process.

You will note that we didn’t need to preprocess our image before calling the task model. That’s because the KerasHub `ImageClassifier` is preprocessing inputs for us as part of `predict()`. Let’s preprocess the image ourselves so we can use the preprocessed inputs directly:

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 10.1 Visualizing intermediate activations
Visualizing intermediate activations consists of displaying the values returned by various convolution and pooling layers in a model, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. You want to visualize feature maps with three dimensions: width, height, and depth (channels). Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image. Let’s start by loading the model that you saved in section 8.2:

### PDF p.311 (printed 287) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
Let’s display the picture (see figure 10.1).

Listing 10.2 Displaying the test picture
