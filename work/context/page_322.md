# Generation bundle for page_322  (PDF page 322, printed page 298)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.2 Visualizing ConvNet filters
Subsection: 10.2.4 The filter visualization loop
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_322.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_322.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p322_e03] (figure_caption) Figure 10.6 Some filter patterns for layers `block2_sepconv1`, `block4_sepconv1`, and `block8_sepconv1`
[p322_e04] (body) These filter visualizations tell you a lot about how ConvNet layers see the world: each layer in a ConvNet learns a collection of filters such that their inputs can be expressed as a combination of the filters. This is similar to how the Fourier transform decomposes signals onto a bank of cosine functions. The filters in these ConvNet filter banks get increasingly complex and refined as you go higher in the model:
[p322_e05] (bullet) The filters from the first layers in the model encode simple directional edges and colors (or colored edges, in some cases).
[p322_e06] (bullet) The filters from layers a bit further up the stack, such as `block4_sepconv1`, encode simple textures made from combinations of edges and colors.

## Figures
- Figure 10.6: Some filter patterns for layers `block2_sepconv1`, `block4_sepconv1`, and `block8_sepconv1`  | labels inside the figure: (none - image)

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

## Previous page (PDF 321) text, for continuity
Figure 10.5 Pattern that the second channel in layer `block3_` `sepconv1` responds to maximally

It seems that filter 2 in layer `block3_sepconv1` is responsive to a horizontal lines pattern, somewhat water-like or fur-like.

Now the fun part: you can start visualizing every filter in the layer—and even every filter in every layer in the model (see figure 10.6).

Listing 10.16 Generating a grid of all filter response patterns

## Next page (PDF 323) text, for continuity
The filters in higher layers begin to resemble textures found in natural images: feathers, eyes, leaves, and so on.

10.3 Visualizing heatmaps of class activation

Here’s one last visualization technique—one that is useful for understanding which parts of a given image led a ConvNet to its final classification decision. This is helpful for “debugging” the decision process of a ConvNet, particularly in the case of a classification mistake (a problem domain called model interpretability). It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM) visualization, and it consists of producing heatmaps of class activation over input images. A class activation heatmap is a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration. For instance, given an image fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate a heatmap for the class “cat,” indicating how cat-like different parts of the image are, and also a heatmap for the class “dog,” indicating how dog-like parts of the image are. The specific implementation we’ll use is the one described in Selvaraju et al.1 Grad-CAM consists of taking the output feature map of a convolution layer, given an input image, and weighting every channel in that feature map by the gradient of the class with respect to the channel. Intuitively, one way to understand this trick is that you’re weighting a spatial map of “how intensely the input image activates different channels” by “how important each channel is with regard to the class,” resulting in a spatial map of “how intensely the input image activates …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.331 — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet
- PDF p.315 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.259 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.313 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.304 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.256 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.294 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.331 (printed 307) — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet › Summary
ConvNets process images by applying a set of learned filters. Filters from earlier layers detect edges and basic textures, while filters from later layers detect increasingly abstract concepts.

You can visualize both the pattern that a filter detects and a filter’s response map across an image.

You can use the Grad-CAM technique to visualize what area(s) in an image were responsible for a classifier’s decision.

Together, these techniques make ConvNets highly interpretable.

### PDF p.315 (printed 291) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2 Visualizing ConvNet filters
Another easy way to inspect the filters learned by ConvNets is to display the visual pattern that each filter is meant to respond to. This can be done with gradient ascent in input space, applying gradient descent to the value of the input image of a ConvNet so as to maximize the response of a specific filter, starting from a blank input image. The resulting input image will be one that the chosen filter is maximally responsive to.

Let’s try this with the filters of the Xception model. The process is simple: we’ll build a loss function that maximizes the value of a given filter in a given convolution layer, and then we’ll use stochastic gradient descent to adjust the values of the input image so as to maximize this activation value. This will be your second example of a low-level

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
wide array of techniques has been developed for visualizing and interpreting these representations. We won’t survey all of them, but we’ll cover three of the most accessible and useful ones:

Visualizing intermediate ConvNet outputs (intermediate activations)—Useful for understanding how successive ConvNet layers transform their input, and for getting a first idea of the meaning of individual ConvNet filters

Visualizing ConvNets filters—Useful for understanding precisely what visual pattern or concept each filter in a ConvNet is receptive to

Visualizing heatmaps of class activation in an image—Useful for understanding which parts of an image were identified as belonging to a given class, thus allowing you to localize objects in images

Visualizing the way ConvNets learn semantically organized manifolds of images—Useful for detecting outliers and getting a sense for what images are seen by your model as semantically ambiguous, thus allowing you to clean and improve your training dataset as well as refine your evaluation dataset

For the first method—activation visualization—you’ll use the small ConvNet that you trained from scratch on the dogs-versus-cats classification problem in

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

### PDF p.320 (printed 296) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2.4 The filter visualization loop
Let’s try it (see figure 10.5):

### PDF p.313 (printed 289) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
This channel appears to encode a diagonal edge detector, but note that your own channels may vary because the specific filters learned by convolution layers aren’t deterministic.

Now let’s plot a complete visualization of all the activations in the network (see figure 10.3). We’ll extract and plot every channel in each of the layer activations, and we’ll stack the results in one big grid, with channels stacked side by side.

Listing 10.6 Visualizing every channel in every intermediate activation

### PDF p.320 (printed 296) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2.4 The filter visualization loop
Now you have all the pieces. Let’s put them together into a Python function that takes a filter index as input and returns a tensor representing the pattern that maximizes the activation of the specified filter in our target layer.

Listing 10.14 Function to generate filter visualizations

### PDF p.304 (printed 280) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › Putting it together: A mini Xception-like model
As a reminder, here are the ConvNet architecture principles you’ve learned so far:

Your model should be organized into repeated blocks of layers, usually made of multiple convolution layers and a max pooling layer.

The number of filters in your layers should increase as the size of the spatial feature maps decreases.
