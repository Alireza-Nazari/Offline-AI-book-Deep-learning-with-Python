# Generation bundle for page_292  (PDF page 292, printed page 268)

Page kind: chapter_opener
Chapter: 9 ConvNet architecture patterns
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_292.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_292.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p292_e01] (chapter_number) 9
[p292_e02] (chapter_title) ConvNet
[p292_e03] (chapter_title) architecture patterns
[p292_e04] (chapter_covers) This chapter covers
[p292_e05] (bullet) The modularity-hierarchy-reuse formula for model
[p292_e06] (bullet) architecture
[p292_e07] (bullet) An overview of standard best practices for
[p292_e08] (bullet) building ConvNets: residual connections,
[p292_e09] (bullet) batch normalization, and depthwise separable
[p292_e10] (bullet) convolutions
[p292_e11] (bullet) Ongoing design trends for computer vision
[p292_e12] (bullet) models
[p292_e13] (body) A model’s “architecture” is the sum of the choices that went into creating it: which layers to use, how to configure them, in what arrangement to connect them. These choices define the hypothesis space of your model: the space of possible functions that gradient descent can search over, parameterized by the model’s weights. Like feature engineering, a good hypothesis space encodes prior knowledge that you have about the problem at hand and its solution. For instance, using convolution layers means that you know in advance that the relevant patterns present in your input

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 291) text, for continuity
Summary

ConvNets excel at computer vision tasks. It’s possible to train one from scratch, even on a very small dataset, with decent results.

ConvNets work by learning a hierarchy of modular patterns and concepts to represent the visual world.

On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when you’re working with image data.

It’s easy to reuse an existing ConvNet on a new dataset via feature extraction. This is a valuable technique for working with small image datasets.

As a complement to feature extraction, you can use fine-tuning, which adapts to a new problem some of the representations previously learned by an existing model. This pushes performance a bit further.

## Next page (PDF 293) text, for continuity
images are translation-invariant. To effectively learn from data, you need to make assumptions about what you’re looking for.

Model architecture is often the difference between success and failure. If you make inappropriate architecture choices, your model may be stuck with suboptimal metrics, and no amount of training data will save it. Inversely, a good model architecture will accelerate learning and will enable your model to make efficient use of the training data available, reducing the need for large datasets. A good model architecture is one that reduces the size of the search space or otherwise makes it easier to converge to a good point of the search space. Just like feature engineering and data curation, model architecture is all about making the problem simpler for gradient descent to solve—and remember that gradient descent is a pretty stupid search process, so it needs all the help it can get. Model architecture is more an art than a science. Experienced machine learning engineers are able to intuitively cobble together high-performing models on their first try, while beginners often struggle to create a model that trains at all. The keyword here is intuitively: no one can give you a clear explanation of what works and what doesn’t. Experts rely on pattern matching, an ability that they acquire through extensive practical experience. You’ll develop your own intuition throughout this book. However, it’s not all about intuition either—there isn’t much in the way of actual science, but like in any engineering discipline, there are best practices. In the following sections, we’ll review a few essential ConvNet architecture best practices, in particular, residual connections, batch normalization, and separable convolution. Once you master how to use them, you wi …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.307 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.294 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.386 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.295 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.306 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.307 (printed 283) — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers › Summary
The architecture of a deep learning model encodes key assumptions about the nature of the problem at hand.

The modularity-hierarchy-reuse formula underpins the architecture of nearly all complex systems, including deep learning models.

Key architecture patterns for computer vision include residual connections, batch normalization, and depthwise separable convolutions.

Vision Transformers are an up-and-coming alternative to ConvNets for large-scale computer vision tasks.

### PDF p.294 (printed 270) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › 
Figure 9.1 Complex systems follow a hierarchical structure and are organized into distinct modules, which are reused multiple times (such as your 4 limbs, which are all variants of the same blueprint, or your 20 fingers).

over a continuous function space), and you structure the search space into modules (layers), organized into a deep hierarchy (often just a stack, the simplest kind of hierarchy), where you reuse whatever you can (for instance, convolutions are all about reusing the same information in different spatial locations).

Likewise, deep learning model architecture is primarily about making clever use of modularity, hierarchy, and reuse. You'll notice that all popular ConvNet architectures are not only structured into layers, they're structured into repeated groups of layers (called `_blocks_` or `_modules_`). For instance, Xception architecture (used in the previous chapter) is structured into repeated `SeparableConv` - `SeparableConv` - `MaxPooling` blocks (see figure 9.2). Further, most ConvNets often feature pyramid-like structures (feature hierarchies). Recall, for example, the progression in the number of convolution filters we used in the first ConvNet we built in

### PDF p.120 (printed 96) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › From layers to models
A deep learning model is a graph of layers. In Keras, that’s the `Model` class. For now, you’ve only seen `Sequential` models (a subclass of `Model`), which are simple stacks of layers, mapping a single input to a single output. But as you move forward, you’ll be exposed to a much broader variety of network topologies. Some common ones are

Two-branch networks

Multihead networks

Residual connections

Network topology can get quite involved. For instance, figure 3.5 shows topology of the graph of layers of a Transformer, a common architecture designed to process text data.

There are generally two ways of building such models in Keras: you can directly subclass the `Model` class, or you can use the Functional API, which lets you do more with less code. We’ll cover both approaches in chapter 7. The topology of a model defines a hypothesis space. You may remember that in chapter 1, we described machine learning as “searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.” By choosing a network topology, you constrain your space of possibilities (hypothesis space) to a specific series of tensor operati

### PDF p.305 (printed 281) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

### PDF p.12 (printed xii) — contents ›  › 
8.2

Training a ConvNet from scratch on a small dataset 241

The relevance of deep learning for small-data problems 242 Downloading the data 242 ■ Building your model 245 Data preprocessing 247 ■ Using data augmentation 252

8.3

Using a pretrained model 256

Feature extraction with a pretrained model 256 ■ Fine-tuning a pretrained model 264

9

ConvNet architecture patterns 268 9.1

Modularity, hierarchy, and reuse 269 9.2 Residual connections 272 9.3 Batch normalization 276 9.4 Depthwise separable convolutions 278 9.5 Putting it together: A mini Xception-like model 280 9.6 Beyond convolution: Vision Transformers 282 10 Interpreting what ConvNets learn 284 10.1 Visualizing intermediate activations 285 10.2 Visualizing ConvNet filters 291

Gradient ascent in TensorFlow 294 ■ Gradient ascent in PyTorch 295 ■ Gradient ascent in JAX 295 ■ The filter visualization loop 296

10.3

Visualizing heatmaps of class activation 299

Getting the gradient of the top class: TensorFlow version 302 Getting the gradient of the top class: PyTorch version 302 Getting the gradient of the top class: JAX version 303 Displaying the class activation heatmap 304

10.4 Visualizing the latent space of a ConvN

### PDF p.295 (printed 271) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › 
Figure 9.2 The “entry flow” of the Xception architecture: note the repeated layer blocks and the gradually shrinking and deepening feature maps, going from 299 x 299 x 3 to 19 x 19 x 728.

Deeper hierarchies are intrinsically good because they encourage feature reuse and, therefore, abstraction. In general, a deep stack of narrow layers performs better than a shallow stack of large layers. However, there’s a limit to how deep you can stack layers: the problem of vanishing gradients. This leads us to our first essential model architecture pattern: residual connections.

### PDF p.307 (printed 283) — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers › 
a sequence-processing architecture. And Transformers are very good at it, which has led to the question: could we also use them for images?

Because ViTs are a type of Transformer, they also process sequences: they split up an image into a 1D sequence of patches, turn each patch into a flat vector, and process the vector sequence. The Transformer architecture allows ViTs to capture long-range relationships between different parts of the image, something ConvNets can sometimes struggle with. Our general experience with Transformers is that they’re a great choice if you’re working with a massive dataset. They’re simply better at utilizing large amounts of data. However, for smaller datasets, they tend to be suboptimal for two reasons. First, they lack the spatial prior of ConvNets—the 2D patch-based architecture of ConvNets incorporates more assumptions about the local structure of the visual space, making them more data efficient. Second, for ViTs to shine, they need to be really large. They end up being unwieldy for anything smaller than ImageNet. The battle for image recognition supremacy is far from over, but ViTs have undoubtedly opened a new and exciting chapter. You’ll probabl

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › 
When building a very deep ConvNet, it’s common to add batch normalization layers as well as residual connections—two architecture patterns that help gradient information flow smoothly through the network.
