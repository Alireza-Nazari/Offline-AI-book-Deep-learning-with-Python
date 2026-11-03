# Generation bundle for page_294  (PDF page 294, printed page 270)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.1 Modularity, hierarchy, and reuse
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_294.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_294.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p294_e03] (figure_caption) Figure 9.1 Complex systems follow a hierarchical structure and are organized into distinct modules, which are reused multiple times (such as your 4 limbs, which are all variants of the same blueprint, or your 20 fingers).
[p294_e04] (body) over a continuous function space), and you structure the search space into modules (layers), organized into a deep hierarchy (often just a stack, the simplest kind of hierarchy), where you reuse whatever you can (for instance, convolutions are all about reusing the same information in different spatial locations).
[p294_e05] (body) Likewise, deep learning model architecture is primarily about making clever use of modularity, hierarchy, and reuse. You'll notice that all popular ConvNet architectures are not only structured into layers, they're structured into repeated groups of layers (called `_blocks_` or `_modules_`). For instance, Xception architecture (used in the previous chapter) is structured into repeated `SeparableConv` - `SeparableConv` - `MaxPooling` blocks (see figure 9.2). Further, most ConvNets often feature pyramid-like structures (feature hierarchies). Recall, for example, the progression in the number of convolution filters we used in the first ConvNet we built in the previous chapter: 32, 64, 128. The number of filters grows with layer depth, while the size of the feature maps shrinks accordingly. You’ll notice the same pattern in the blocks of the Xception model (see figure 9.2).

## Figures
- Figure 9.1: Complex systems follow a hierarchical structure and are organized into distinct modules, which are reused multiple times (such as your 4 limbs, which are all variants of the same blueprint, or your 20 fingers).  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 293) text, for continuity
images are translation-invariant. To effectively learn from data, you need to make assumptions about what you’re looking for.

Model architecture is often the difference between success and failure. If you make inappropriate architecture choices, your model may be stuck with suboptimal metrics, and no amount of training data will save it. Inversely, a good model architecture will accelerate learning and will enable your model to make efficient use of the training data available, reducing the need for large datasets. A good model architecture is one that reduces the size of the search space or otherwise makes it easier to converge to a good point of the search space. Just like feature engineering and data curation, model architecture is all about making the problem simpler for gradient descent to solve—and remember that gradient descent is a pretty stupid search process, so it needs all the help it can get. Model architecture is more an art than a science. Experienced machine learning engineers are able to intuitively cobble together high-performing models on their first try, while beginners often struggle to create a model that trains at all. The keyword here is intuitively: no one can give you a clear explanation of what works and what doesn’t. Experts rely on pattern matching, an ability that they acquire through extensive practical experience. You’ll develop your own intuition throughout this book. However, it’s not all about intuition either—there isn’t much in the way of actual science, but like in any engineering discipline, there are best practices. In the following sections, we’ll review a few essential ConvNet architecture best practices, in particular, residual connections, batch normalization, and separable convolution. Once you master how to use them, you wi …

## Next page (PDF 295) text, for continuity
Figure 9.2 The “entry flow” of the Xception architecture: note the repeated layer blocks and the gradually shrinking and deepening feature maps, going from 299 x 299 x 3 to 19 x 19 x 728.

Deeper hierarchies are intrinsically good because they encourage feature reuse and, therefore, abstraction. In general, a deep stack of narrow layers performs better than a shallow stack of large layers. However, there’s a limit to how deep you can stack layers: the problem of vanishing gradients. This leads us to our first essential model architecture pattern: residual connections.

On the importance of ablation studies in deep learning research

Deep learning architectures are often more evolved than designed—they were developed by repeatedly trying things and selecting what seemed to work. Much like in biological systems, if you take any complicated experimental deep learning setup,

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.304 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.292 — 9 ConvNet architecture patterns › 
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.307 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.316 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.298 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.297 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.259 — 8 Image classification › 8.1 Introduction to convnets

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.304 (printed 280) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › Putting it together: A mini Xception-like model
As a reminder, here are the ConvNet architecture principles you’ve learned so far:

Your model should be organized into repeated blocks of layers, usually made of multiple convolution layers and a max pooling layer.

The number of filters in your layers should increase as the size of the spatial feature maps decreases.

### PDF p.307 (printed 283) — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers › Summary
The architecture of a deep learning model encodes key assumptions about the nature of the problem at hand.

The modularity-hierarchy-reuse formula underpins the architecture of nearly all complex systems, including deep learning models.

Key architecture patterns for computer vision include residual connections, batch normalization, and depthwise separable convolutions.

Vision Transformers are an up-and-coming alternative to ConvNets for large-scale computer vision tasks.

### PDF p.292 (printed 268) — 9 ConvNet architecture patterns ›  › This chapter covers
The modularity-hierarchy-reuse formula for model

architecture

An overview of standard best practices for

building ConvNets: residual connections,

batch normalization, and depthwise separable

convolutions

Ongoing design trends for computer vision

models

A model’s “architecture” is the sum of the choices that went into creating it: which layers to use, how to configure them, in what arrangement to connect them. These choices define the hypothesis space of your model: the space of possible functions that gradient descent can search over, parameterized by the model’s weights. Like feature engineering, a good hypothesis space encodes prior knowledge that you have about the problem at hand and its solution. For instance, using convolution layers means that you know in advance that the relevant patterns present in your input

### PDF p.305 (printed 281) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

### PDF p.171 (printed 147) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Figure 5.10 Going from a random model to an overfit model and achieving a robust fit as an intermediate state

Deep learning models tend to be structured in a way that mirrors the “shape” of the information in their training data (via architecture priors). This is the case in particular for image-processing models (see chapters 8–12) and sequence-processing models (see chapter 13). More generally, deep neural networks structure their learned representations in a hierarchical and modular way, which echoes the way natural data is organized.

### PDF p.316 (printed 292) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
You’ll notice that the `SeparableConv2D` layers here are all named something like `block6_sepconv1`, `block7_sepconv2`, etc.—Xception is structured into blocks, each containing several convolutional layers.

Now let’s create a second model that returns the output of a specific layer—a “feature extractor” model. Because our model is a Functional API model, it is inspectable: you can query the `output` of one of its layers and reuse it in a new model. No need to copy the entire Xception code.

Listing 10.9 A feature extractor model returning a specific output

### PDF p.626 (printed 602) — 20 Conclusions › 20.1 Key concepts in review › ConvNets
Convolution layers look at spatially local patterns by applying the same geometric transformation to different spatial locations (patches) in an input tensor. This results in representations that are translation invariant, making convolution layers highly data efficient and modular. This idea is applicable to spaces of any dimensionality: 1D (continuous sequences), 2D (images), 3D (volumes), and so on. You can use the `Conv1D` layer to process sequences, the `Conv2D` layer to process images, and the `Conv3D` layer to process volumes. As a leaner, more efficient alternative to convolution layers, you can also use depthwise separable convolution layers, such as `SeparableConv2D`.

ConvNets, or convolutional networks, consist of stacks of convolution and max-pooling layers. The pooling layers let you spatially downsample the data, which is required to keep feature maps to a reasonable size as the number of features grows and to allow subsequent convolution layers to “see” a greater spatial extent of the inputs. ConvNets are often ended with either a `Flatten` operation or a global pooling layer, turning spatial feature maps into vectors, followed by `Dense` layers to achieve classific

### PDF p.269 (printed 245) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Building your model
We will reuse the same general model structure you saw in the first example: the ConvNet will be a stack of alternated `Conv2D` (with `relu` activation) and `MaxPooling2D` layers.

But because we’re dealing with bigger images and a more complex problem, we’ll make our model larger, accordingly: it will have two more `Conv2D MaxPooling2D` stages. This serves both to augment the capacity of the model and to further reduce the size of the feature maps so they aren’t overly large when we reach the pooling layer. Here, because we start from inputs of size 180 × 180 pixels (a somewhat arbitrary choice), we end up with feature maps of size 7 × 7 just before the `GlobalAveragePooling2D` layer. NOTE The depth of the feature maps progressively increases in the model (from 32 to 512), whereas the size of the feature maps decreases (from 180 × 180 to 7 × 7). This is a pattern you’ll see in almost all ConvNets.

Because we’re looking at a binary classification problem, we’ll end the model with a single unit (a `Dense` layer of size 1) and a `sigmoid` activation. This unit will encode the probability that the model is looking at one class or the other.

One last small difference: we will start t
