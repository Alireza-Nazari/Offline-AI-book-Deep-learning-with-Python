# Generation bundle for page_295  (PDF page 295, printed page 271)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.1 Modularity, hierarchy, and reuse
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_295.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_295.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p295_e03] (code)
```
299×299×3 images
Conv 32, 3×3, stride=2×2
ReLU
Conv 64, 3×3
ReLU
```
[p295_e04] (code)
```
SeparableConv 128, 3×3
Conv 1×1
ReLU
stride=2×2
SeparableConv 128, 3×3
MaxPooling 3×3, stride=2×2
+
ReLU
SeparableConv 256, 3×3
Conv 1×1
ReLU
stride=2×2
SeparableConv 256, 3×3
MaxPooling 3×3, stride=2×2
+
ReLU
SeparableConv 728, 3×3
Conv 1×1
ReLU
stride=2×2
SeparableConv 728, 3×3
MaxPooling 3×3, stride=2×2
+
19×19×728 feature maps
```
[p295_e05] (figure_caption) Figure 9.2 The “entry flow” of the Xception architecture: note the repeated layer blocks and the gradually shrinking and deepening feature maps, going from 299 x 299 x 3 to 19 x 19 x 728.
[p295_e06] (body) Deeper hierarchies are intrinsically good because they encourage feature reuse and, therefore, abstraction. In general, a deep stack of narrow layers performs better than a shallow stack of large layers. However, there’s a limit to how deep you can stack layers: the problem of vanishing gradients. This leads us to our first essential model architecture pattern: residual connections.
[p295_e07] (heading_subsection) On the importance of ablation studies in deep learning research
[p295_e08] (sidebar) Deep learning architectures are often more evolved than designed—they were developed by repeatedly trying things and selecting what seemed to work. Much like in biological systems, if you take any complicated experimental deep learning setup,

## Figures
- Figure 9.2: The “entry flow” of the Xception architecture: note the repeated layer blocks and the gradually shrinking and deepening feature maps, going from 299 x 299 x 3 to 19 x 19 x 728.  | labels inside the figure: 299 x 299 x 3, 149 x 149 x 32, 74 x 74 x 128, 37 x 37 x 256, 19 x 19 x 728

## Equation candidates (formulas written inline in the text)
- [p295_e03] 299×299×3 images
Conv 32, 3×3, stride=2×2
ReLU
Conv 64, 3×3
ReLU

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 294) text, for continuity
Figure 9.1 Complex systems follow a hierarchical structure and are organized into distinct modules, which are reused multiple times (such as your 4 limbs, which are all variants of the same blueprint, or your 20 fingers).

over a continuous function space), and you structure the search space into modules (layers), organized into a deep hierarchy (often just a stack, the simplest kind of hierarchy), where you reuse whatever you can (for instance, convolutions are all about reusing the same information in different spatial locations).

Likewise, deep learning model architecture is primarily about making clever use of modularity, hierarchy, and reuse. You'll notice that all popular ConvNet architectures are not only structured into layers, they're structured into repeated groups of layers (called `_blocks_` or `_modules_`). For instance, Xception architecture (used in the previous chapter) is structured into repeated `SeparableConv` - `SeparableConv` - `MaxPooling` blocks (see figure 9.2). Further, most ConvNets often feature pyramid-like structures (feature hierarchies). Recall, for example, the progression in the number of convolution filters we used in the first ConvNet we built in the previous chapter: 32, 64, 128. The number of filters grows with layer depth, while the size of the feature maps shrinks accordingly. You’ll notice the same pattern in the blocks of the Xception model (see figure 9.2).

## Next page (PDF 296) text, for continuity
(continued)

chances are you can remove a few modules (or replace some trained features with random ones) with no loss of performance. This is made worse by the incentives that deep learning researchers face: by making a system more complex than necessary, they can make it appear more interesting or more novel and thus increase their chances of getting a paper through the peer review process. If you read lots of deep learning papers, you will notice that they’re often optimized for peer review in both style and content in ways that actively hurt clarity of explanation and reliability of results. For instance, mathematics in deep learning papers is rarely used for clearly formalizing concepts or deriving unobvious results—rather, it gets used as a signal of seriousness, like an expensive suit on a salesperson. The goal of research shouldn’t be merely to publish but to generate reliable knowledge. Crucially, understanding causality in your system is the most straightforward way to generate reliable knowledge. And there’s a very low-effort way to look into causality: ablation studies. Ablation studies consist of systematically trying to remove parts of a system—-that is, make it simpler—to identify where its performance actually comes from. If you find that X + Y + Z gives you good results, also try X, Y, Z, X + Y, X + Z, Y + Z and see what happens. If you become a deep learning researcher, cut through the noise in the research process: do ablation studies for your models. Always ask, could there be a simpler explanation? Is this added complexity really necessary? Why?

9.2

Residual connections

You probably know about the game of telephone, also called Chinese whispers in the UK and téléphone arabe in France, where an initial message is whispered in the ear of a player,  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.304 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.297 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.263 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.298 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.307 — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.307 (printed 283) — 9 ConvNet architecture patterns › 9.6 Beyond convolution: Vision Transformers › Summary
The architecture of a deep learning model encodes key assumptions about the nature of the problem at hand.

The modularity-hierarchy-reuse formula underpins the architecture of nearly all complex systems, including deep learning models.

Key architecture patterns for computer vision include residual connections, batch normalization, and depthwise separable convolutions.

Vision Transformers are an up-and-coming alternative to ConvNets for large-scale computer vision tasks.

### PDF p.305 (printed 281) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

### PDF p.293 (printed 269) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › Modularity, hierarchy, and reuse
If you want to make a complex system simpler, there’s a universal recipe you can apply: just structure your amorphous soup of complexity into modules, organize the modules into a hierarchy, and start reusing the same modules in multiple places as appropriate (“reuse” is another word for abstraction). That’s the modularity-hierarchy-reuse (MHR) formula (see figure 9.1), and it underlies system architecture across pretty much every domain where the term architecture is used. It’s at the heart of the organization of any system of meaningful complexity, whether it’s a cathedral, your own body, the US Navy, or the Keras codebase.

If you’re a software engineer, you’re already keenly familiar with these principles: an effective codebase is one that is modular, hierarchical, and where you don’t reimplement the same thing twice but instead rely on reusable classes and functions. If you factor your code by following these principles, you could say you’re doing “software architecture.” Deep learning itself is simply the application of this recipe to continuous optimization via gradient descent: you take a classic optimization technique (gradient descent

### PDF p.304 (printed 280) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › Putting it together: A mini Xception-like model
As a reminder, here are the ConvNet architecture principles you’ve learned so far:

Your model should be organized into repeated blocks of layers, usually made of multiple convolution layers and a max pooling layer.

The number of filters in your layers should increase as the size of the spatial feature maps decreases.

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

### PDF p.293 (printed 269) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › 
images are translation-invariant. To effectively learn from data, you need to make assumptions about what you’re looking for.

Model architecture is often the difference between success and failure. If you make inappropriate architecture choices, your model may be stuck with suboptimal metrics, and no amount of training data will save it. Inversely, a good model architecture will accelerate learning and will enable your model to make efficient use of the training data available, reducing the need for large datasets. A good model architecture is one that reduces the size of the search space or otherwise makes it easier to converge to a good point of the search space. Just like feature engineering and data curation, model architecture is all about making the problem simpler for gradient descent to solve—and remember that gradient descent is a pretty stupid search process, so it needs all the help it can get. Model architecture is more an art than a science. Experienced machine learning engineers are able to intuitively cobble together high-performing models on their first try, while beginners often struggle to create a model that trains at all. The keyword here is intuitively: no one

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › 
When building a very deep ConvNet, it’s common to add batch normalization layers as well as residual connections—two architecture patterns that help gradient information flow smoothly through the network.

### PDF p.300 (printed 276) — 9 ConvNet architecture patterns › 9.3 Batch normalization › 
With residual connections, you can build networks of arbitrary depth, without having to worry about vanishing gradients. Now, let’s move on to the next essential ConvNet architecture pattern: batch normalization.
