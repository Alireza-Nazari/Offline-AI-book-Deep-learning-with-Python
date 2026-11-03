# Generation bundle for page_293  (PDF page 293, printed page 269)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.1 Modularity, hierarchy, and reuse
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_293.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_293.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p293_e03] (body) images are translation-invariant. To effectively learn from data, you need to make assumptions about what you’re looking for.
[p293_e04] (body) Model architecture is often the difference between success and failure. If you make inappropriate architecture choices, your model may be stuck with suboptimal metrics, and no amount of training data will save it. Inversely, a good model architecture will accelerate learning and will enable your model to make efficient use of the training data available, reducing the need for large datasets. A good model architecture is one that reduces the size of the search space or otherwise makes it easier to converge to a good point of the search space. Just like feature engineering and data curation, model architecture is all about making the problem simpler for gradient descent to solve—and remember that gradient descent is a pretty stupid search process, so it needs all the help it can get. Model architecture is more an art than a science. Experienced machine learning engineers are able to intuitively cobble together high-performing models on their first try, while beginners often struggle to create a model that trains at all. The keyword here is intuitively: no one can give you a clear explanation of what works and what doesn’t. Experts rely on pattern matching, an ability that they acquire through extensive practical experience. You’ll develop your own intuition throughout this book. However, it’s not all about intuition either—there isn’t much in the way of actual science, but like in any engineering discipline, there are best practices. In the following sections, we’ll review a few essential ConvNet architecture best practices, in particular, residual connections, batch normalization, and separable convolution. Once you master how to use them, you will be able to build highly effective image models. We will demonstrate how to apply them on our dogs-versus-cats classification problem. Let’s start from the bird’s-eye view: the modularity-hierarchy-reuse (MHR) formula for system architecture.
[p293_e05] (heading_section) 9.1
[p293_e06] (heading_section) Modularity, hierarchy, and reuse
[p293_e07] (body) If you want to make a complex system simpler, there’s a universal recipe you can apply: just structure your amorphous soup of complexity into modules, organize the modules into a hierarchy, and start reusing the same modules in multiple places as appropriate (“reuse” is another word for abstraction). That’s the modularity-hierarchy-reuse (MHR) formula (see figure 9.1), and it underlies system architecture across pretty much every domain where the term architecture is used. It’s at the heart of the organization of any system of meaningful complexity, whether it’s a cathedral, your own body, the US Navy, or the Keras codebase.
[p293_e08] (body) If you’re a software engineer, you’re already keenly familiar with these principles: an effective codebase is one that is modular, hierarchical, and where you don’t reimplement the same thing twice but instead rely on reusable classes and functions. If you factor your code by following these principles, you could say you’re doing “software architecture.” Deep learning itself is simply the application of this recipe to continuous optimization via gradient descent: you take a classic optimization technique (gradient descent

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 292) text, for continuity
ConvNet

architecture patterns

This chapter covers

The modularity-hierarchy-reuse formula for model

architecture

An overview of standard best practices for

building ConvNets: residual connections,

batch normalization, and depthwise separable

convolutions

Ongoing design trends for computer vision

models

A model’s “architecture” is the sum of the choices that went into creating it: which layers to use, how to configure them, in what arrangement to connect them. These choices define the hypothesis space of your model: the space of possible functions that gradient descent can search over, parameterized by the model’s weights. Like feature engineering, a good hypothesis space encodes prior knowledge that you have about the problem at hand and its solution. For instance, using convolution layers means that you know in advance that the relevant patterns present in your input

## Next page (PDF 294) text, for continuity
Figure 9.1 Complex systems follow a hierarchical structure and are organized into distinct modules, which are reused multiple times (such as your 4 limbs, which are all variants of the same blueprint, or your 20 fingers).

over a continuous function space), and you structure the search space into modules (layers), organized into a deep hierarchy (often just a stack, the simplest kind of hierarchy), where you reuse whatever you can (for instance, convolutions are all about reusing the same information in different spatial locations).

Likewise, deep learning model architecture is primarily about making clever use of modularity, hierarchy, and reuse. You'll notice that all popular ConvNet architectures are not only structured into layers, they're structured into repeated groups of layers (called `_blocks_` or `_modules_`). For instance, Xception architecture (used in the previous chapter) is structured into repeated `SeparableConv` - `SeparableConv` - `MaxPooling` blocks (see figure 9.2). Further, most ConvNets often feature pyramid-like structures (feature hierarchies). Recall, for example, the progression in the number of convolution filters we used in the first ConvNet we built in the previous chapter: 32, 64, 128. The number of filters grows with layer depth, while the size of the feature maps shrinks accordingly. You’ll notice the same pattern in the blocks of the Xception model (see figure 9.2).

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.307 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.306 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.295 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.563 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.624 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.300 — 9 ConvNet architecture patterns › 9.3 Batch normalization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.179 (printed 155) — 5 Fundamentals of machine learning › 5.3 Improving model fit › Using better architecture priors
You have a model that fits, but for some reason your validation metrics aren’t improving at all. They remain no better than what a random classifier would achieve: your model trains, but doesn’t generalize. What’s going on?

This is perhaps the worst machine learning situation you can find yourself in. It indicates that something is fundamentally wrong with your approach, and it may not be easy to tell what. Here are some tips. First, it may be that the input data you’re using simply doesn’t contain sufficient information to predict your targets: the problem as formulated is not solvable. This is what happened earlier when we tried to fit an MNIST model where the labels were shuffled: the model would train just fine, but validation accuracy would stay stuck at 10%, because it was plainly impossible to generalize with such a dataset. It may also be that the kind of model you’re using is not suited for the problem at hand. For instance, in chapter 13, you’ll see an example of a timeseries prediction problem where a densely connected architecture isn’t able to beat a trivial baseline, whereas a more appropriate recurrent architecture does manage to generalize well. Using a model that 

### PDF p.307 (printed 283) — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers › Summary
The architecture of a deep learning model encodes key assumptions about the nature of the problem at hand.

The modularity-hierarchy-reuse formula underpins the architecture of nearly all complex systems, including deep learning models.

Key architecture patterns for computer vision include residual connections, batch normalization, and depthwise separable convolutions.

Vision Transformers are an up-and-coming alternative to ConvNets for large-scale computer vision tasks.

### PDF p.120 (printed 96) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › From layers to models
A deep learning model is a graph of layers. In Keras, that’s the `Model` class. For now, you’ve only seen `Sequential` models (a subclass of `Model`), which are simple stacks of layers, mapping a single input to a single output. But as you move forward, you’ll be exposed to a much broader variety of network topologies. Some common ones are

Two-branch networks

Multihead networks

Residual connections

Network topology can get quite involved. For instance, figure 3.5 shows topology of the graph of layers of a Transformer, a common architecture designed to process text data.

There are generally two ways of building such models in Keras: you can directly subclass the `Model` class, or you can use the Functional API, which lets you do more with less code. We’ll cover both approaches in chapter 7. The topology of a model defines a hypothesis space. You may remember that in chapter 1, we described machine learning as “searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.” By choosing a network topology, you constrain your space of possibilities (hypothesis space) to a specific series of tensor operati

### PDF p.624 (printed 600) — 20 Conclusions › 20.1 Key concepts in review › 20.1.6 Key network architectures
The families of network architectures that you should be familiar with after reading this book are densely connected networks, convolutional networks, recurrent networks, Diffusion Models, and Transformers. Each type of model is meant for specific data modalities: a network architecture encodes assumptions about the structure of the data—a hypothesis space within which the search for a good model will proceed. Whether a given architecture will work on a given problem depends entirely on the match between the structure of the data and the assumptions of the network architecture.

These different network types can easily be combined to achieve larger multimodal models, much as you combine LEGO bricks. In a way, deep learning layers are LEGO bricks for information processing. Table 20.1 shows a quick overview of the mapping between input and output modalities and the appropriate network architectures.

Table 20.1 Model architectures for different data types

Input Output Model

Vector data Class probability, Regression value Densely connected network Timeseries data Class probability, Regression value RNN, Transformer Images Class probability, Regression value ConvNet Text Class proba

### PDF p.569 (printed 545) — 18 Best practices for the real world › 18.1 Getting the most out of your models › The art of crafting the right search space
Overall, hyperparameter optimization is a powerful technique that is an absolute requirement to get to state-of-the-art models on any task or to win machine learning competitions. Think about it: once upon a time, people handcrafted the features that went into shallow machine learning models. That was very suboptimal. Now deep learning automates the task of hierarchical feature engineering—features are learned using a feedback signal, not hand-tuned, and that’s the way it should be. In the same way, you shouldn’t handcraft your model architectures; you should optimize them in a principled way.

However, doing hyperparameter tuning is not a replacement for being familiar with model architecture best practices: search spaces grow combinatorially with the number of choices, so it would be far too expensive to turn everything into a hyperparameter and let the tuner sort it out. You need to be smart about designing the right search space. Hyperparameter tuning is automation, not magic: you use it to automate experiments that you would otherwise have run by hand, but you still need to handpick experiment configurations that have the potential to yield good metrics. The good news: by usin

### PDF p.295 (printed 271) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › 
Figure 9.2 The “entry flow” of the Xception architecture: note the repeated layer blocks and the gradually shrinking and deepening feature maps, going from 299 x 299 x 3 to 19 x 19 x 728.

Deeper hierarchies are intrinsically good because they encourage feature reuse and, therefore, abstraction. In general, a deep stack of narrow layers performs better than a shallow stack of large layers. However, there’s a limit to how deep you can stack layers: the problem of vanishing gradients. This leads us to our first essential model architecture pattern: residual connections.

### PDF p.205 (printed 181) — 6 The universal workflow of machine learning › 6.2 Developing a model › Beating a baseline
As you start working on the model itself, your initial goal is to achieve statistical power, as you saw in chapter 5—that is, to develop a small model that is capable of beating a simple baseline.

At this stage, these are the three most important things you should focus on:

Feature engineering—Filter out uninformative features (feature selection) and use your knowledge of the problem to develop new features that are likely to be useful.

Selecting the correct architecture priors—What type of model architecture will you use? A densely connected network, a ConvNet, a recurrent neural network, a Transformer? Is deep learning even a good approach for the task, or should you use something else?

Selecting a good enough training configuration—What loss function should you use? What batch size and learning rate?

### PDF p.307 (printed 283) — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers › 
a sequence-processing architecture. And Transformers are very good at it, which has led to the question: could we also use them for images?

Because ViTs are a type of Transformer, they also process sequences: they split up an image into a 1D sequence of patches, turn each patch into a flat vector, and process the vector sequence. The Transformer architecture allows ViTs to capture long-range relationships between different parts of the image, something ConvNets can sometimes struggle with. Our general experience with Transformers is that they’re a great choice if you’re working with a massive dataset. They’re simply better at utilizing large amounts of data. However, for smaller datasets, they tend to be suboptimal for two reasons. First, they lack the spatial prior of ConvNets—the 2D patch-based architecture of ConvNets incorporates more assumptions about the local structure of the visual space, making them more data efficient. Second, for ViTs to shine, they need to be really large. They end up being unwieldy for anything smaller than ImageNet. The battle for image recognition supremacy is far from over, but ViTs have undoubtedly opened a new and exciting chapter. You’ll probabl
