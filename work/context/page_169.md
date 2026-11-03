# Generation bundle for page_169  (PDF page 169, printed page 145)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.1 Generalization: The goal of machine learning
Subsection: 5.1.2 The nature of generalization in deep learning
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_169.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_169.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p169_e03] (body) digits, as well as for human faces, tree morphology, the sounds of the human voice, and even natural language.
[p169_e04] (body) The manifold hypothesis implies
[p169_e05] (bullet) Machine learning models only have to fit relatively simple, low-dimensional, highly structured subspaces within their potential input space (latent manifolds).
[p169_e06] (bullet) Within one of these manifolds, it’s always possible to interpolate between two inputs—that is, morph one into another via a continuous path along which all points fall on the manifold.
[p169_e07] (body) The ability to interpolate between samples is the key to understanding generalization in deep learning.
[p169_e08] (heading_subsubsection) Interpolation as a source of generalization
[p169_e09] (body) If you work with data points that can be interpolated, you can start making sense of points you’ve never seen before by relating them to other points that lie close on the manifold. In other words, you can make sense of the totality of the space using only a sample of the space. You can use interpolation to fill in the blanks.
[p169_e10] (body) Note that interpolation on the latent manifold is different from linear interpolation in the parent space, as illustrated in figure 5.8. For instance, the average of pixels between two MNIST digits is usually not a valid digit. Crucially, while deep learning achieves generalization via interpolation on a learned approximation of
[p169_e11] (figure_caption) Figure 5.8 Difference between linear interpolation
[p169_e12] (body) the data manifold, it would be a mis
[p169_e13] (figure_caption) and interpolation on the latent manifold. Every point
[p169_e14] (body) take to assume that interpolation is
[p169_e15] (figure_caption) on the latent manifold of digits is a valid digit, but the
[p169_e16] (body) all there is to generalization. It’s the
[p169_e17] (figure_caption) average of two digits usually isn’t.
[p169_e18] (body) tip of the iceberg. Interpolation can only help you make sense of things that are very close to what you’ve seen before: it enables local generalization. But remarkably, humans deal with extreme novelty all the time, and they do just fine. You don’t need to be trained in advance on countless examples of every situation you’ll ever have to encounter. Every single one of your days is different from any day you’ve experienced before, and different from any day experienced by anyone since the dawn of humanity. You can switch between spending a week in NYC, a week in Shanghai, and a week in Bangalore without requiring thousands of lifetimes of learning and rehearsal for each city.
[p169_e19] (body) Humans are capable of extreme generalization, which is enabled by cognitive mechanisms other than interpolation—abstraction, symbolic models of the world, reasoning, logic, common sense, innate priors about the world—what we generally call reason, as

## Figures
- Figure 5.8: Difference between linear interpolation  | labels inside the figure: Manifold interpolation, (intermediate point on, the latent manifold), Linear interpolation, (average in the, encoding space)
- Figure None: and interpolation on the latent manifold. Every point  | labels inside the figure: (none - image)
- Figure None: on the latent manifold of digits is a valid digit, but the  | labels inside the figure: (none - image)
- Figure None: average of two digits usually isn’t.  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 5 Fundamentals of machine learning (PDF pages 160-194)
  - 5.1 Generalization: The goal of machine learning (p.160)
      - 5.1.1 Underfitting and overfitting (p.161)
      - 5.1.2 The nature of generalization in deep learning (p.167)
  - 5.2 Evaluating machine-learning models (p.173)
      - 5.2.1 Training, validation, and test sets (p.173)
      - 5.2.2 Beating a common-sense baseline (p.176)
      - 5.2.3 Things to keep in mind about model evaluation (p.176)
  - 5.3 Improving model fit (p.177)
      - 5.3.1 Tuning key gradient descent parameters (p.177)
      - 5.3.2 Using better architecture priors (p.179)
      - 5.3.3 Increasing model capacity (p.179)
  - 5.4 Improving generalization (p.182)
      - 5.4.1 Dataset curation (p.183)
      - 5.4.2 Feature engineering (p.183)
      - 5.4.3 Using early stopping (p.185)
      - 5.4.4 Regularizing your model (p.185)

## Previous page (PDF 168) text, for continuity
The manifold hypothesis

The input to an MNIST classifier (before preprocessing) is a 28 × 28 array of integers between 0 and 255. The total number of possible input values is thus 256 to the power of 784—much greater than the number of atoms in the universe. However, very few of these inputs would look like valid MNIST samples: actual handwritten digits only occupy a tiny subspace of the parent space of all possible 28 × 28 `uint8` arrays. What’s more, this subspace isn’t just a set of points sprinkled at random in the parent space: it is highly structured.

First, the subspace of valid handwritten digits is continuous: if you take a sample and modify it a little, it will still be recognizable as the same handwritten digit. Further, all samples in the valid subspace are connected by smooth paths that run through the subspace. This means that if you take two random MNIST digits A and B, there exists a sequence of “intermediate” images that morph A into B, such that two consecutive digits are very close to each other (see figure 5.7). Perhaps there will be a few ambiguous shapes close to the boundary between two classes, but even these shapes would still look very digit-like.

Figure 5.7 Different MNIST digits gradually morphing into one another, showing that the space of handwritten digits forms a “manifold.” This image was generated using code from chapter 17.

In technical terms, you would say that handwritten digits form a manifold within the space of possible 28 × 28 `uint8` arrays. That’s a big word, but the concept is pretty intuitive. A manifold is a lower-dimensional subspace of some parent space that is locally similar to a linear (Euclidean) space. For instance, a smooth curve in the plane is a 1D manifold within a 2D space because for every point of the curve …

## Next page (PDF 170) text, for continuity
opposed to intuition and pattern recognition. The latter are largely interpolative in nature, but the former isn’t. Both are essential to intelligence. We’ll talk more about this in chapter 19.

Why deep learning works

Remember the crumpled paper ball metaphor from chapter 2? A sheet of paper represents a 2D manifold within 3D space (figure 5.9). A deep learning model is a tool for uncrumpling paper balls—that is, for disentangling latent manifolds.

Figure 5.9 Uncrumpling a complicated manifold of data

A deep learning model is basically a very high-dimensional curve. The curve is smooth and continuous (with additional constraints on its structure, originating from model architecture priors) because it needs to be differentiable. And that curve is fitted to data points via gradient descent—smoothly and incrementally. By construction, deep learning is about taking a big, complex curve—a manifold—and incrementally adjusting its parameters until it fits some training data points.

The curve involves enough parameters that it could fit anything. Indeed, if you let your model train for long enough, it will effectively end up purely memorizing its training data and won’t generalize at all. However, the data you’re fitting to isn’t made of isolated points sparsely distributed across the underlying space. Your data forms a highly structured, low-dimensional manifold within the input space—that’s the manifold hypothesis. And because fitting your model curve to this data happens gradually and smoothly over time, as gradient descent progresses, there will be an intermediate point during training at which the model roughly approximates the natural manifold of the data, as you can see in figure 5.10. Moving along the curve learned by the model at that point will come close to movi …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.193 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.171 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.172 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.612 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.561 — 17 Image generation › 17.3 Text-to-image models
- PDF p.593 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.560 — 17 Image generation › 17.3 Text-to-image models
- PDF p.487 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.193 (printed 169) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Summary
The purpose of a machine learning model is to generalize: to perform accurately on never-before-seen inputs. It’s harder than it seems.

A deep neural network achieves generalization by learning a parametric model that can successfully interpolate between training samples. Such a model can be said to have learned the latent manifold of the training data. This is why deep learning models can only make sense of inputs that are very close to what they’ve seen during training.

The fundamental problem in machine learning is the tension between optimization and generalization: to attain generalization, you must first achieve a good fit to the training data, but improving your model’s fit to the training data will inevitably start hurting generalization after a while. Every single deep learning best practice deals with managing this tension.

The ability of deep learning models to generalize comes from the fact that they manage to learn to approximate the latent manifold of their data and can thus make sense of new inputs via interpolation.

It’s essential to be able to accurately evaluate the generalization power of your model while you’re developing it. You have at your disposal an arr

### PDF p.172 (printed 148) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Figure 5.11 A dense sampling of the input space is necessary to learn a model capable of accurate generalization.

past training inputs, without having to use common-sense, abstract reasoning, or external knowledge about the world—all things that machine learning models have no access to. As such, you should always keep in mind that the best way to improve a deep learning model is to train it on more data or better data (of course, adding overly noisy or inaccurate data will harm generalization). A denser coverage of the input data manifold will yield a model that generalizes better. You should never expect a deep learning model to perform anything more than crude interpolation between its training samples, and thus, you should do everything you can to make interpolation as easy as possible. The only thing you will find in a deep learning model is what you put into it: the priors encoded in its architecture and the data it was trained on.

When getting more data isn’t possible, the next-best solution is to modulate the quantity of information that your model is allowed to store, or to add constraints on the smoothness of the model curve. If a network can only afford to memorize a s

### PDF p.560 (printed 536) — 17 Image generation › 17.3 Text-to-image models › 
between these two points would land us inside the sphere. We would no longer be on its surface. We would like to stay on the surface of the smooth manifold learned by our text embedding— that’s where embedding points have meaning for our denoising model. See figure 17.16.

Of course, the manifold learned by our text embedding model is not actually spherical. But it’s a smooth surface of numbers all with the same rough magnitude—it is sphere-like, and interpolating as if we were on a sphere is a better approximation than interpolat

Figure 17.16 Spherical interpolation keeps us

ing as if we were on a line.

close to the surface of our manifold.

With our interpolation defined, let’s try walking between the text embeddings for two prompts and generating an image at each interpolated output. We will run our `slerp` function from 0.5 to 0.6 (out of 0 to 1) to zoom in on the middle of the interpolation right when the “morph” becomes visually obvious (figure 17.17):

### PDF p.171 (printed 147) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Training data is paramount
While deep learning is indeed well suited to manifold learning, the power to generalize is more a consequence of the natural structure of your data than a consequence of any property of your model. You’ll only be able to generalize if your data forms a manifold where points can be interpolated. The more informative and the less noisy your features are, the better you will be able to generalize, since your input space will be simpler and better structured. Data curation and feature engineering are essential to generalization.

Further, because deep learning is curve fitting, for a model to perform well, it needs to be trained on a dense sampling of its input space. A “dense sampling” in this context means that the training data should densely cover the entirety of the input data manifold (see figure 5.11). This is especially true near decision boundaries. With a sufficiently dense sampling, it becomes possible to make sense of new inputs by interpolating between

### PDF p.598 (printed 574) — 19 The future of AI › 19.2 Scale isn’t all you need › 
inputs that deviate a bit from what they’re familiar with, which is precisely what makes them useful. Our dogs-versus-cats model from chapter 8 could classify cat or dog pictures it had not seen before, as long as they were close enough to what it was trained on. However, deep nets are limited to what I call local generalization (see figure 19.6): the mapping from inputs to outputs performed by a deep net quickly stops making sense as inputs start deviating from what the net saw at training time. Deep nets can only generalize to known unknowns, to factors of variation that were anticipated during model development and that are extensively featured in the training data, such as different camera angles or lighting conditions for pet pictures. That’s because deep nets generalize via interpolation on a manifold (remember chapter 5): any factor of variation in their input space needs to be captured by the manifold they learn. That’s why basic data augmentation is so helpful in improving deep net generalization. Unlike humans, these models have no ability to improvise in the face of situations for which little or no data is available.

Figure 19.6 Local generalization vs. extreme general

### PDF p.330 (printed 306) — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet › 10.4 Visualizing the latent space of a ConvNet
I’ll introduce one last model inspection technique: latent space visualization. All deep learning models work by embedding their inputs in a latent space, a manifold (or curve) where points that are spatially close to each other represent semantically similar inputs. On this manifold, different classes of inputs become nicely separable, which is what enables classification.

Can we visualize in two dimensions what this manifold looks like? The general idea is to pick a certain layer in the model, typically the one right before the classification layer, and record its activations for a set of inputs. The activations for each input represent the coordinates of the corresponding point on the latent manifold. They’re points on a curve. To visualize that curve—the latent manifold—all we need to do is to plot these points. Note that this process can be done with any layer in your model—each layer encodes a different latent space, and these latent spaces get gradually more semantically organized as your go up the model. Remember our analogy about how deep learning models are machines for uncrumpling paper balls? The latent space of each layer is a snapshot of the paper ball at different s

### PDF p.557 (printed 533) — 17 Image generation › 17.3 Text-to-image models › 17.3.1 Exploring the latent space of a text-to-image model
There is probably no better way to see the interpolative nature of deep neural networks than text diffusion models. The text encoder used by our model will learn a smooth, low-dimensional manifold to represent our input prompts. It’s continuous, meaning we have learned a space where we can walk from the text representation of one prompt to another, and each intermediate point will have semantic meaning. We can couple that with our diffusion process to morph between two images by simply describing each end state with a text prompt.

### PDF p.612 (printed 588) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.3 Why deep learning isn’t a complete answer to abstraction generation
Meanwhile, writing a sorting algorithm in Python takes just a few lines, and the resulting program, once validated on a couple more examples, will work every time on lists of any size. That’s pretty strong generalization: going from a couple of demonstration examples and test examples to a program that can successfully process literally any list of numbers. In reverse, perception problems are a terrible fit for discrete reasoning processes. Try to write a pure-Python program to classify MNIST digits, without using any machine learning technique: you’re in for a ride. You’ll find yourself painstakingly coding functions that can detect the number of closed loops in a digit, the coordinates of the center of mass of a digit, and so on. After thousands of lines of code, you might achieve 90% test accuracy. In this case, fitting a parametric model is much simpler; it can better utilize the large amount of data that’s available, and it achieves much more robust results. If you have lots of data and you’re faced with a problem where the manifold hypothesis applies, go with deep learning. For this reason, it’s unlikely that we’ll see the rise of an approach that would reduce reasoning probl
