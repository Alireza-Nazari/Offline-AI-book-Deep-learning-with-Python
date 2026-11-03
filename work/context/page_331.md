# Generation bundle for page_331  (PDF page 331, printed page 307)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.4 Visualizing the latent space of a ConvNet
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_331.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_331.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p331_e03] (body) This task is not as impossible as it may sound. You see, while the points you’re working with may be, say, 2048-dimensional, the curve they represent has a much lower intrinsic dimensionality. That’s precisely why we refer to it as a manifold—a manifold is a lower-dimensional curve inside a higher-dimensional space. For instance, while planet Earth is a three-dimensional object, its surface is actually 2D, and so it possible to project it on a 2D plane. That’s what world maps are all about. In fact, world maps illustrate both the feasibility and the challenges of lower-dimensional projections of higher-dimensional manifolds. All projections distort reality. The widely used Mercator map projection distorts distances, in particular by making areas near the equator look smaller than they actually are, and inversely by making areas near the poles stretch towards infinity. And that’s even though the surface you’re projecting is in fact 2D—whereas deep learning latent manifolds may be much higher-dimensional than that. There’s no projection that won’t destroy at least some information.
[p331_e04] (heading_section) Summary
[p331_e05] (bullet) ConvNets process images by applying a set of learned filters. Filters from earlier layers detect edges and basic textures, while filters from later layers detect increasingly abstract concepts.
[p331_e06] (bullet) You can visualize both the pattern that a filter detects and a filter’s response map across an image.
[p331_e07] (bullet) You can use the Grad-CAM technique to visualize what area(s) in an image were responsible for a classifier’s decision.
[p331_e08] (bullet) Together, these techniques make ConvNets highly interpretable.

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

## Previous page (PDF 330) text, for continuity
This visualization technique answers two important questions:

Why did the network think this image contained an African elephant?

Where is the African elephant located in the picture?

In particular, it’s interesting to note that the ears of the elephant calf are strongly activated: this is probably how the network can tell the difference between African and Indian elephants.

10.4 Visualizing the latent space of a ConvNet

I’ll introduce one last model inspection technique: latent space visualization. All deep learning models work by embedding their inputs in a latent space, a manifold (or curve) where points that are spatially close to each other represent semantically similar inputs. On this manifold, different classes of inputs become nicely separable, which is what enables classification.

Can we visualize in two dimensions what this manifold looks like? The general idea is to pick a certain layer in the model, typically the one right before the classification layer, and record its activations for a set of inputs. The activations for each input represent the coordinates of the corresponding point on the latent manifold. They’re points on a curve. To visualize that curve—the latent manifold—all we need to do is to plot these points. Note that this process can be done with any layer in your model—each layer encodes a different latent space, and these latent spaces get gradually more semantically organized as your go up the model. Remember our analogy about how deep learning models are machines for uncrumpling paper balls? The latent space of each layer is a snapshot of the paper ball at different stages of its uncrumpling. This is why it’s most interesting to target the second-before-last layer, when the paper ball is almost fully flattened into a nice sheet. Now t …

## Next page (PDF 332) text, for continuity
Image segmentation

This chapter covers

The different branches of computer vision: image

classification, image segmentation, and object

detection

Building a segmentation model from scratch

Using the pretrained Segment Anything Model

Chapter 8 gave you a first introduction to deep learning for computer vision via a simple use case: binary image classification. But there’s more to computer vision than image classification! This chapter dives deeper into another essential computer vision application—image segmentation.

11.1 Computer vision tasks

So far, we’ve focused on image classification models: an image goes in, a label comes out. “This image likely contains a cat; this other one likely contains a dog.” But image classification is only one of several possible applications of deep learning in computer vision. In general, there are three essential computer vision tasks you need to know about:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.322 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.308 — 10 Interpreting what ConvNets learn › 
- PDF p.315 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.259 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.170 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.265 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.168 (printed 144) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › The manifold hypothesis
In technical terms, you would say that handwritten digits form a manifold within the space of possible 28 × 28 `uint8` arrays. That’s a big word, but the concept is pretty intuitive. A manifold is a lower-dimensional subspace of some parent space that is locally similar to a linear (Euclidean) space. For instance, a smooth curve in the plane is a 1D manifold within a 2D space because for every point of the curve, you can draw a tangent (the curve can be approximated by a line in every point). A smooth surface within a 3D space is a 2D manifold. And so on.

More generally, the manifold hypothesis posits that all natural data lies on a low-dimensional manifold within the high-dimensional space where it is encoded. That’s a pretty strong statement about the structure of information in the universe. As far as we know, it’s accurate, and it’s the reason why deep learning works. It’s true for MNIST

### PDF p.170 (printed 146) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Why deep learning works
Remember the crumpled paper ball metaphor from chapter 2? A sheet of paper represents a 2D manifold within 3D space (figure 5.9). A deep learning model is a tool for uncrumpling paper balls—that is, for disentangling latent manifolds.

Figure 5.9 Uncrumpling a complicated manifold of data

A deep learning model is basically a very high-dimensional curve. The curve is smooth and continuous (with additional constraints on its structure, originating from model architecture priors) because it needs to be differentiable. And that curve is fitted to data points via gradient descent—smoothly and incrementally. By construction, deep learning is about taking a big, complex curve—a manifold—and incrementally adjusting its parameters until it fits some training data points.

The curve involves enough parameters that it could fit anything. Indeed, if you let your model train for long enough, it will effectively end up purely memorizing its training data and won’t generalize at all. However, the data you’re fitting to isn’t made of isolated points sparsely distributed across the underlying space. Your data forms a highly structured, low-dimensional manifold within the input space—that’s the man

### PDF p.169 (printed 145) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
digits, as well as for human faces, tree morphology, the sounds of the human voice, and even natural language.

The manifold hypothesis implies

Machine learning models only have to fit relatively simple, low-dimensional, highly structured subspaces within their potential input space (latent manifolds).

Within one of these manifolds, it’s always possible to interpolate between two inputs—that is, morph one into another via a continuous path along which all points fall on the manifold.

The ability to interpolate between samples is the key to understanding generalization in deep learning.

### PDF p.308 (printed 284) — 10 Interpreting what ConvNets learn ›  › This chapter covers
Interpreting how ConvNets decompose an input

image

Visualizing the filters learned by ConvNets

Visualizing areas in an image responsible for a

certain classification decision

A fundamental problem when building a computer vision application is that of interpretability: Why did your classifier think a particular image contained a fridge, when all you can see is a truck? This is especially relevant to use cases where deep learning is used to complement human expertise, such as medical imaging use cases. This chapter will get you familiar with a range of different techniques for visualizing what ConvNets learn and understanding the decisions they make.

It’s often said that deep learning models are “black boxes”: they learn representations that are difficult to extract and present in a human-readable form. Although this is partially true for certain types of deep learning models, it’s definitely not true for ConvNets. The representations learned by ConvNets are highly amenable to visualization, in large part because they’re representations of visual concepts. Since 2013, a

### PDF p.322 (printed 298) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
Figure 10.6 Some filter patterns for layers `block2_sepconv1`, `block4_sepconv1`, and `block8_sepconv1`

These filter visualizations tell you a lot about how ConvNet layers see the world: each layer in a ConvNet learns a collection of filters such that their inputs can be expressed as a combination of the filters. This is similar to how the Fourier transform decomposes signals onto a bank of cosine functions. The filters in these ConvNet filter banks get increasingly complex and refined as you go higher in the model:

The filters from the first layers in the model encode simple directional edges and colors (or colored edges, in some cases).

The filters from layers a bit further up the stack, such as `block4_sepconv1`, encode simple textures made from combinations of edges and colors.

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
wide array of techniques has been developed for visualizing and interpreting these representations. We won’t survey all of them, but we’ll cover three of the most accessible and useful ones:

Visualizing intermediate ConvNet outputs (intermediate activations)—Useful for understanding how successive ConvNet layers transform their input, and for getting a first idea of the meaning of individual ConvNet filters

Visualizing ConvNets filters—Useful for understanding precisely what visual pattern or concept each filter in a ConvNet is receptive to

Visualizing heatmaps of class activation in an image—Useful for understanding which parts of an image were identified as belonging to a given class, thus allowing you to localize objects in images

Visualizing the way ConvNets learn semantically organized manifolds of images—Useful for detecting outliers and getting a sense for what images are seen by your model as semantically ambiguous, thus allowing you to clean and improve your training dataset as well as refine your evaluation dataset

For the first method—activation visualization—you’ll use the small ConvNet that you trained from scratch on the dogs-versus-cats classification problem in

### PDF p.557 (printed 533) — 17 Image generation › 17.3 Text-to-image models › 17.3.1 Exploring the latent space of a text-to-image model
There is probably no better way to see the interpolative nature of deep neural networks than text diffusion models. The text encoder used by our model will learn a smooth, low-dimensional manifold to represent our input prompts. It’s continuous, meaning we have learned a space where we can walk from the text representation of one prompt to another, and each intermediate point will have semantic meaning. We can couple that with our diffusion process to morph between two images by simply describing each end state with a text prompt.

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
