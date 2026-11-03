# Generation bundle for page_170  (PDF page 170, printed page 146)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.1 Generalization: The goal of machine learning
Subsection: 5.1.2 The nature of generalization in deep learning
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_170.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_170.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p170_e03] (body) opposed to intuition and pattern recognition. The latter are largely interpolative in nature, but the former isn’t. Both are essential to intelligence. We’ll talk more about this in chapter 19.
[p170_e04] (heading_subsubsection) Why deep learning works
[p170_e05] (body) Remember the crumpled paper ball metaphor from chapter 2? A sheet of paper represents a 2D manifold within 3D space (figure 5.9). A deep learning model is a tool for uncrumpling paper balls—that is, for disentangling latent manifolds.
[p170_e06] (figure_caption) Figure 5.9 Uncrumpling a complicated manifold of data
[p170_e07] (body) A deep learning model is basically a very high-dimensional curve. The curve is smooth and continuous (with additional constraints on its structure, originating from model architecture priors) because it needs to be differentiable. And that curve is fitted to data points via gradient descent—smoothly and incrementally. By construction, deep learning is about taking a big, complex curve—a manifold—and incrementally adjusting its parameters until it fits some training data points.
[p170_e08] (body) The curve involves enough parameters that it could fit anything. Indeed, if you let your model train for long enough, it will effectively end up purely memorizing its training data and won’t generalize at all. However, the data you’re fitting to isn’t made of isolated points sparsely distributed across the underlying space. Your data forms a highly structured, low-dimensional manifold within the input space—that’s the manifold hypothesis. And because fitting your model curve to this data happens gradually and smoothly over time, as gradient descent progresses, there will be an intermediate point during training at which the model roughly approximates the natural manifold of the data, as you can see in figure 5.10. Moving along the curve learned by the model at that point will come close to moving along the actual latent manifold of the data. As such, the model will be capable of making sense of never-before-seen inputs via interpolation between training inputs. Besides the trivial fact that they have sufficient representational power, there are a few properties of deep learning models that make them particularly well suited to learning latent manifolds:
[p170_e09] (bullet) Deep learning models implement a smooth, continuous mapping from their inputs to their outputs. It has to be smooth and continuous because it must be differentiable, by necessity (you couldn’t do gradient descent otherwise). This smoothness helps approximate latent manifolds, which follow the same properties.

## Figures
- Figure 5.9: Uncrumpling a complicated manifold of data  | labels inside the figure: (none - image)

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

## Previous page (PDF 169) text, for continuity
digits, as well as for human faces, tree morphology, the sounds of the human voice, and even natural language.

The manifold hypothesis implies

Machine learning models only have to fit relatively simple, low-dimensional, highly structured subspaces within their potential input space (latent manifolds).

Within one of these manifolds, it’s always possible to interpolate between two inputs—that is, morph one into another via a continuous path along which all points fall on the manifold.

The ability to interpolate between samples is the key to understanding generalization in deep learning.

Interpolation as a source of generalization

If you work with data points that can be interpolated, you can start making sense of points you’ve never seen before by relating them to other points that lie close on the manifold. In other words, you can make sense of the totality of the space using only a sample of the space. You can use interpolation to fill in the blanks.

Note that interpolation on the latent manifold is different from linear interpolation in the parent space, as illustrated in figure 5.8. For instance, the average of pixels between two MNIST digits is usually not a valid digit. Crucially, while deep learning achieves generalization via interpolation on a learned approximation of

Figure 5.8 Difference between linear interpolation

the data manifold, it would be a mis

and interpolation on the latent manifold. Every point

take to assume that interpolation is

on the latent manifold of digits is a valid digit, but the

all there is to generalization. It’s the

average of two digits usually isn’t.

tip of the iceberg. Interpolation can only help you make sense of things that are very close to what you’ve seen before: it enables local generalization. But remarkably, hum …

## Next page (PDF 171) text, for continuity
Figure 5.10 Going from a random model to an overfit model and achieving a robust fit as an intermediate state

Deep learning models tend to be structured in a way that mirrors the “shape” of the information in their training data (via architecture priors). This is the case in particular for image-processing models (see chapters 8–12) and sequence-processing models (see chapter 13). More generally, deep neural networks structure their learned representations in a hierarchical and modular way, which echoes the way natural data is organized.

Training data is paramount

While deep learning is indeed well suited to manifold learning, the power to generalize is more a consequence of the natural structure of your data than a consequence of any property of your model. You’ll only be able to generalize if your data forms a manifold where points can be interpolated. The more informative and the less noisy your features are, the better you will be able to generalize, since your input space will be simpler and better structured. Data curation and feature engineering are essential to generalization.

Further, because deep learning is curve fitting, for a model to perform well, it needs to be trained on a dense sampling of its input space. A “dense sampling” in this context means that the training data should densely cover the entirety of the input data manifold (see figure 5.11). This is especially true near decision boundaries. With a sufficiently dense sampling, it becomes possible to make sense of new inputs by interpolating between

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.63 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.183 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.193 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.595 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.622 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.331 — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet
- PDF p.589 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.561 — 17 Image generation › 17.3 Text-to-image models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.63 (printed 39) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
out a transformation of the paper ball that would uncrumple it to make the two classes cleanly separable again (see figure 2.14). With deep learning, this would be implemented as a series of simple transformations of the 3D space, such as those you could apply on the paper ball with your fingers, one movement at a time.

Figure 2.14 Uncrumpling a complicated manifold of data

Uncrumpling paper balls is what machine learning is about: finding neat representations for complex, highly folded data manifolds in high-dimensional spaces (a manifold is a continuous surface, like our crumpled sheet of paper). At this point, you should have a pretty good intuition as to why deep learning excels at this: it takes the approach of incrementally decomposing a complicated geometric transformation into a long chain of elementary ones, which is pretty much the strategy a human would follow to uncrumple a paper ball. Each layer in a deep network applies a transformation that disentangles the data a little—and a deep stack of layers makes tractable an extremely complicated disentanglement process.

### PDF p.193 (printed 169) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Summary
The purpose of a machine learning model is to generalize: to perform accurately on never-before-seen inputs. It’s harder than it seems.

A deep neural network achieves generalization by learning a parametric model that can successfully interpolate between training samples. Such a model can be said to have learned the latent manifold of the training data. This is why deep learning models can only make sense of inputs that are very close to what they’ve seen during training.

The fundamental problem in machine learning is the tension between optimization and generalization: to attain generalization, you must first achieve a good fit to the training data, but improving your model’s fit to the training data will inevitably start hurting generalization after a while. Every single deep learning best practice deals with managing this tension.

The ability of deep learning models to generalize comes from the fact that they manage to learn to approximate the latent manifold of their data and can thus make sense of new inputs via interpolation.

It’s essential to be able to accurately evaluate the generalization power of your model while you’re developing it. You have at your disposal an arr

### PDF p.330 (printed 306) — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet › 10.4 Visualizing the latent space of a ConvNet
I’ll introduce one last model inspection technique: latent space visualization. All deep learning models work by embedding their inputs in a latent space, a manifold (or curve) where points that are spatially close to each other represent semantically similar inputs. On this manifold, different classes of inputs become nicely separable, which is what enables classification.

Can we visualize in two dimensions what this manifold looks like? The general idea is to pick a certain layer in the model, typically the one right before the classification layer, and record its activations for a set of inputs. The activations for each input represent the coordinates of the corresponding point on the latent manifold. They’re points on a curve. To visualize that curve—the latent manifold—all we need to do is to plot these points. Note that this process can be done with any layer in your model—each layer encodes a different latent space, and these latent spaces get gradually more semantically organized as your go up the model. Remember our analogy about how deep learning models are machines for uncrumpling paper balls? The latent space of each layer is a snapshot of the paper ball at different s

### PDF p.168 (printed 144) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › The manifold hypothesis
In technical terms, you would say that handwritten digits form a manifold within the space of possible 28 × 28 `uint8` arrays. That’s a big word, but the concept is pretty intuitive. A manifold is a lower-dimensional subspace of some parent space that is locally similar to a linear (Euclidean) space. For instance, a smooth curve in the plane is a 1D manifold within a 2D space because for every point of the curve, you can draw a tangent (the curve can be approximated by a line in every point). A smooth surface within a 3D space is a 2D manifold. And so on.

More generally, the manifold hypothesis posits that all natural data lies on a low-dimensional manifold within the high-dimensional space where it is encoded. That’s a pretty strong statement about the structure of information in the universe. As far as we know, it’s accurate, and it’s the reason why deep learning works. It’s true for MNIST

### PDF p.622 (printed 598) — 20 Conclusions › 20.1 Key concepts in review › 
transformation is that it must be differentiable, which is required for us to be able to learn its parameters via gradient descent. Intuitively, this means the geometric morphing from inputs to outputs must be smooth and continuous—a significant constraint.

The entire process of applying this complex geometric transformation to the input data can be visualized in 3D by imagining a person trying to uncrumple a paper ball: the crumpled paper ball is the manifold of the input data that the model starts with. Each movement operated by the person on the paper ball is similar to a simple geometric transformation operated by one layer. The full uncrumpling gesture sequence is the complex transformation of the entire model. Deep learning models are mathematical machines for uncrumpling complicated manifolds of high-dimensional data. That’s the magic of deep learning—turning meaning into vectors, into geometric spaces, and then incrementally learning complex geometric transformations that map one space to another. All you need are spaces of sufficiently high dimensionality to capture the full scope of the relationships found in the original data. The whole thing hinges on two core ideas: t

### PDF p.172 (printed 148) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Figure 5.11 A dense sampling of the input space is necessary to learn a model capable of accurate generalization.

past training inputs, without having to use common-sense, abstract reasoning, or external knowledge about the world—all things that machine learning models have no access to. As such, you should always keep in mind that the best way to improve a deep learning model is to train it on more data or better data (of course, adding overly noisy or inaccurate data will harm generalization). A denser coverage of the input data manifold will yield a model that generalizes better. You should never expect a deep learning model to perform anything more than crude interpolation between its training samples, and thus, you should do everything you can to make interpolation as easy as possible. The only thing you will find in a deep learning model is what you put into it: the priors encoded in its architecture and the data it was trained on.

When getting more data isn’t possible, the next-best solution is to modulate the quantity of information that your model is allowed to store, or to add constraints on the smoothness of the model curve. If a network can only afford to memorize a s

### PDF p.167 (printed 143) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › The nature of generalization in deep learning
In fact, you don’t even need to do this with MNIST data—you could just generate white noise inputs and random labels. You could fit a model on that, too, as long as it has enough parameters. It would just end up memorizing specific inputs, much like a Python dictionary.

If this is the case, then why do deep learning models generalize at all? Shouldn’t they just learn an ad hoc mapping between training inputs and targets, like a fancy `dict`? What expectation can we have that this mapping will work for new inputs? As it turns out, the nature of generalization in deep learning has rather little to do with deep learning models themselves and much to do with the structure of information in the real world. Let’s take a look at what’s really going on here.

### PDF p.598 (printed 574) — 19 The future of AI › 19.2 Scale isn’t all you need › 
inputs that deviate a bit from what they’re familiar with, which is precisely what makes them useful. Our dogs-versus-cats model from chapter 8 could classify cat or dog pictures it had not seen before, as long as they were close enough to what it was trained on. However, deep nets are limited to what I call local generalization (see figure 19.6): the mapping from inputs to outputs performed by a deep net quickly stops making sense as inputs start deviating from what the net saw at training time. Deep nets can only generalize to known unknowns, to factors of variation that were anticipated during model development and that are extensively featured in the training data, such as different camera angles or lighting conditions for pet pictures. That’s because deep nets generalize via interpolation on a manifold (remember chapter 5): any factor of variation in their input space needs to be captured by the manifold they learn. That’s why basic data augmentation is so helpful in improving deep net generalization. Unlike humans, these models have no ability to improvise in the face of situations for which little or no data is available.

Figure 19.6 Local generalization vs. extreme general
