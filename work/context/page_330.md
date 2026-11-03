# Generation bundle for page_330  (PDF page 330, printed page 306)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.4 Visualizing the latent space of a ConvNet
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_330.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_330.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p330_e03] (code)
```
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.utils.array_to_img(superimposed_img)
save_path = "elephant_cam.jpg"
superimposed_img.save(save_path)
```
[p330_e04] (body) This visualization technique answers two important questions:
[p330_e05] (bullet) Why did the network think this image contained an African elephant?
[p330_e06] (bullet) Where is the African elephant located in the picture?
[p330_e07] (body) In particular, it’s interesting to note that the ears of the elephant calf are strongly activated: this is probably how the network can tell the difference between African and Indian elephants.
[p330_e08] (heading_section) 10.4 Visualizing the latent space of a ConvNet
[p330_e09] (body) I’ll introduce one last model inspection technique: latent space visualization. All deep learning models work by embedding their inputs in a latent space, a manifold (or curve) where points that are spatially close to each other represent semantically similar inputs. On this manifold, different classes of inputs become nicely separable, which is what enables classification.
[p330_e10] (body) Can we visualize in two dimensions what this manifold looks like? The general idea is to pick a certain layer in the model, typically the one right before the classification layer, and record its activations for a set of inputs. The activations for each input represent the coordinates of the corresponding point on the latent manifold. They’re points on a curve. To visualize that curve—the latent manifold—all we need to do is to plot these points. Note that this process can be done with any layer in your model—each layer encodes a different latent space, and these latent spaces get gradually more semantically organized as your go up the model. Remember our analogy about how deep learning models are machines for uncrumpling paper balls? The latent space of each layer is a snapshot of the paper ball at different stages of its uncrumpling. This is why it’s most interesting to target the second-before-last layer, when the paper ball is almost fully flattened into a nice sheet. Now there’s just one obstacle. The human visual system is very much limited to low-dimensional objects. We can interpret 2D, 3D, maybe 4D visualizations. We’re generally happiest with simple 2D spaces. Meanwhile, the point coordinates we’d like to plot are in a space that has typically hundreds of dimensions, and possibly thousands. To visualize them, we’re going to need to project them on a 2D surface, while trying to retain as much of the organization of the original manifold as feasible.

## Code annotations printed next to listings (callouts, in order)
- Saves the
- superimposed image
- Superimposes the heatmap and the original
- image, with the heatmap at 40% opacity

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

## Previous page (PDF 329) text, for continuity
Finally, let’s generate an image that superimposes the original image on the heatmap you just obtained (see figure 10.9).

Figure 10.9 African elephant class activation heatmap over the test picture

Listing 10.25 Superimposing the heatmap with the original picture

## Next page (PDF 331) text, for continuity
This task is not as impossible as it may sound. You see, while the points you’re working with may be, say, 2048-dimensional, the curve they represent has a much lower intrinsic dimensionality. That’s precisely why we refer to it as a manifold—a manifold is a lower-dimensional curve inside a higher-dimensional space. For instance, while planet Earth is a three-dimensional object, its surface is actually 2D, and so it possible to project it on a 2D plane. That’s what world maps are all about. In fact, world maps illustrate both the feasibility and the challenges of lower-dimensional projections of higher-dimensional manifolds. All projections distort reality. The widely used Mercator map projection distorts distances, in particular by making areas near the equator look smaller than they actually are, and inversely by making areas near the poles stretch towards infinity. And that’s even though the surface you’re projecting is in fact 2D—whereas deep learning latent manifolds may be much higher-dimensional than that. There’s no projection that won’t destroy at least some information.

Summary

ConvNets process images by applying a set of learned filters. Filters from earlier layers detect edges and basic textures, while filters from later layers detect increasingly abstract concepts.

You can visualize both the pattern that a filter detects and a filter’s response map across an image.

You can use the Grad-CAM technique to visualize what area(s) in an image were responsible for a classifier’s decision.

Together, these techniques make ConvNets highly interpretable.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.325 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.539 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.312 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models
- PDF p.561 — 17 Image generation › 17.3 Text-to-image models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.169 (printed 145) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Interpolation as a source of generalization
If you work with data points that can be interpolated, you can start making sense of points you’ve never seen before by relating them to other points that lie close on the manifold. In other words, you can make sense of the totality of the space using only a sample of the space. You can use interpolation to fill in the blanks.

Note that interpolation on the latent manifold is different from linear interpolation in the parent space, as illustrated in figure 5.8. For instance, the average of pixels between two MNIST digits is usually not a valid digit. Crucially, while deep learning achieves generalization via interpolation on a learned approximation of

Figure 5.8 Difference between linear interpolation

the data manifold, it would be a mis

and interpolation on the latent manifold. Every point

take to assume that interpolation is

on the latent manifold of digits is a valid digit, but the

all there is to generalization. It’s the

average of two digits usually isn’t.

tip of the iceberg. Interpolation can only help you make sense of things that are very close to what you’ve seen before: it enables local generalization. But remarkably, humans deal with extreme novelty all the time

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

### PDF p.557 (printed 533) — 17 Image generation › 17.3 Text-to-image models › 17.3.1 Exploring the latent space of a text-to-image model
There is probably no better way to see the interpolative nature of deep neural networks than text diffusion models. The text encoder used by our model will learn a smooth, low-dimensional manifold to represent our input prompts. It’s continuous, meaning we have learned a space where we can walk from the text representation of one prompt to another, and each intermediate point will have semantic meaning. We can couple that with our diffusion process to morph between two images by simply describing each end state with a text prompt.

### PDF p.533 (printed 509) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.1 Sampling from latent spaces of images
The key idea of image generation is to develop a low-dimensional latent space of representations (which, like everything else in deep learning, is a vector space) where any point can be mapped to a “valid” image: an image that looks like the real thing. The module capable of realizing this mapping, taking as input a latent point and outputting an image (a grid of pixels), is usually called a generator, or sometimes a decoder. Once such a latent space has been learned, you can sample points from it, and, by mapping them back to image space, generate images that have never been seen before (see figure 17.1)—the in-betweens of the training images.

Figure 17.1 Using a latent vector space to sample new images

Further, text-conditioning makes it possible to map a space of prompts in natural language to the latent space (see figure 17.2), making it possible to do language-guided image generation—generating pictures that correspond to a text description. This category of models is called text-to-image models.

Interpolating between many training images in the latent space enables such models to generate infinite combinations of visual concepts, including many that no one had explicitly c

### PDF p.536 (printed 512) — 17 Image generation › 17.1 Deep learning for image generation › 
and `z_log_variance`

Figure 17.4 A VAE maps an image to two vectors, `z_mean` and `z_log_variance`, which define a probability distribution over the latent space, used to sample a latent point to decode.

similar to `input_img`, thus forcing the latent space to be continuously meaningful. Any two close points in the latent space will decode to highly similar images. Continuity, combined with the low dimensionality of the latent space, forces every direction in the latent space to encode a meaningful axis of variation of the data, making the latent space very structured and thus highly suitable to manipulation via concept vectors.

The parameters of a VAE are trained via two loss functions: a reconstruction loss that forces the decoded samples to match the initial inputs, and a regularization loss that helps learn well-rounded latent distributions and reduces overfitting to the training data. Schematically, the process looks like this:

### PDF p.534 (printed 510) — 17 Image generation › 17.1 Deep learning for image generation › 
Of course, there are still challenges to overcome. Like with all deep learning models, the latent space doesn’t encode a consistent model of the physical world, so you might occasionally see hands with extra fingers, incoherent lighting, or garbled objects. The coherence of generated images is still an area of active research. In the case of figure 17.2, despite having seen tens of thousands of images of people riding bikes, the model doesn’t understand in a human sense what it means to ride a bike—concepts like pedaling, steering, or maintaining upright balance. That’s why your bike-riding horse is unlikely to get depicted pedaling with its hind legs in a believable manner, the way a human artist would draw it.

There’s a range of different strategies for learning such latent spaces of image representations, each with its own characteristics. The most common types of image generation models are

Diffusion models

Variational autoencoders (VAEs)

Generative adversarial networks (GANs)

While previous editions of this book covered GANs, they have gradually fallen out of fashion in recent years and have been all but replaced by diffusion models. In this edition, we’ll cover both VAEs
