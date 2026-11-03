# Generation bundle for page_306  (PDF page 306, printed page 282)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.6 Beyond convolution: Vision Transformers
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_306.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_306.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p306_e03] (code)
```
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```
[p306_e04] (body) This ConvNet has a trainable parameter count of 721,857, significantly lower than the 1,569,089 trainable parameters of the model from the previous chapter, yet it achieves better results. Figure 9.5 shows the training and validation curves.
[p306_e05] (figure_caption) Figure 9.5 Training and validation metrics with a Xception-like architecture
[p306_e06] (body) You’ll find that our new model achieves a test accuracy of 90.8%—compared to 83.9% for the previous model. As you can see, following architecture best practices does have an immediate, sizeable effect on model performance!
[p306_e07] (body) At this point, if you want to further improve performance, you should start systematically tuning the hyperparameters of your architecture—a topic we cover in detail in chapter 18. We haven’t gone through this step here, so the configuration of the previous model is purely from the best practices we outlined, plus, when it comes to gauging model size, a small amount of intuition.
[p306_e08] (heading_section) 9.6
[p306_e09] (heading_section) Beyond convolution: Vision Transformers
[p306_e10] (body) While ConvNets have been dominating the field of computer vision since the mid-2010s, they’ve been recently competing with an alternative architecture: Vision Transformers (or ViTs for short). It may well be that ViTs will end up replacing ConvNets in the long term—though, for now, ConvNets remain your best option in most cases.
[p306_e11] (body) You don’t yet know what Transformers are because we’ll cover them in chapter 15. In short, the Transformer architecture was developed to process text—it’s fundamentally

## Code annotations printed next to listings (callouts, in order)
- Like in the original
- model, we add a
- dropout layer for
- regularization.

## Figures
- Figure 9.5: Training and validation metrics with a Xception-like architecture  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p306_e03] x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 305) text, for continuity
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

## Next page (PDF 307) text, for continuity
a sequence-processing architecture. And Transformers are very good at it, which has led to the question: could we also use them for images?

Because ViTs are a type of Transformer, they also process sequences: they split up an image into a 1D sequence of patches, turn each patch into a flat vector, and process the vector sequence. The Transformer architecture allows ViTs to capture long-range relationships between different parts of the image, something ConvNets can sometimes struggle with. Our general experience with Transformers is that they’re a great choice if you’re working with a massive dataset. They’re simply better at utilizing large amounts of data. However, for smaller datasets, they tend to be suboptimal for two reasons. First, they lack the spatial prior of ConvNets—the 2D patch-based architecture of ConvNets incorporates more assumptions about the local structure of the visual space, making them more data efficient. Second, for ViTs to shine, they need to be really large. They end up being unwieldy for anything smaller than ImageNet. The battle for image recognition supremacy is far from over, but ViTs have undoubtedly opened a new and exciting chapter. You’ll probably work with this architecture in the context of large-scale generative image models—a topic we’ll cover in chapter 17. For your small-scale image classification needs, however, ConvNets remain your best bet. This concludes our introduction to essential ConvNet architecture best practices. With these principles in hand, you’ll be able to develop higher-performing models across a wide range of computer vision tasks. You’re now well on your way to becoming a proficient computer vision practitioner. To further deepen your expertise, there’s one last important topic we need to cover: interpreting h …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.256 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.293 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.292 — 9 ConvNet architecture patterns › 
- PDF p.294 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.303 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.12 — contents › 
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.635 (printed 611) — index ›  › 
ConvNet filters 285

achievements of 12 architecture patterns 268, 272–276, 280–282 advantages of 597 convolution operation 235–239 AI winters 14 depthwise separable convolutions 278–280 artificial intelligence, machine learning,

and 2–3, 599 interpreting what they learn 284 depth of model 7 max-pooling operation 239–241 enabling technologies 598 modularity, hierarchy, and reuse 269 for image generation 509–518 training from scratch on small dataset 241 frameworks 61–62 visualizing filters 291–299 generative AI 11 visualizing heatmaps of class activation 299–306 guided program search 606 visualizing intermediate activations 285–291 image classification for small-data problems 242 visualizing latent space of 306 integrating modules into hybrid systems 590 ViTs (Vision Transformers) 282 Keras 206–214 convolutional base 257 limitations of 564–570, 605 convolution kernel 237 network architectures 600–605 convolution operation 235–239, 550 object detection, training YOLO model 342 border effects and padding 238 overview of 8, 597 strides 239 properties of 10 Core Knowledge priors 581 short-term hype 13–14 cosine distance 462 staying up to date 607–608 cost function 9 text classificatio

### PDF p.303 (printed 279) — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions › 
Figure 9.4 Depthwise separable convolution: a depthwise convolution followed by a pointwise convolution

neural networks, it serves as a useful prior that helps the model make more efficient use of its training data. A model with stronger priors about the structure of the information it will have to process is a better model—as long as the priors are accurate.

Depthwise separable convolution requires significantly fewer parameters and involves fewer computations compared to regular convolution, while having comparable representational power. They result in smaller models that converge faster and are less prone to overfitting. These advantages become especially important when you’re training small models from scratch on limited data. When it comes to larger-scale models, depthwise separable convolutions are the basis of the Xception architecture, a high-performing ConvNet that comes packaged with Keras. You can read more about the theoretical grounding for depthwise separable convolutions and Xception in the paper “Xception: Deep Learning with Depthwise Separable Convolutions.”3

### PDF p.303 (printed 279) — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions › The co-evolution of hardware, software, and algorithms
Consider a regular convolution operation with a 3 x 3 window, 64 input channels, and 64 output channels. It uses 3 × 3 × 64 × 64 = 36,864 trainable parameters, and when you apply it to an image, it runs a number of floating-point operations that is proportional to this parameter count. Meanwhile, consider an equivalent depthwise separable convolution: it only involves 3 × 3 × 64 + 64 × 64 = 4,672 trainable parameters and proportionally fewer floating-point operations. This efficiency improvement only increases as the number of filters or the size of the convolution windows gets larger.

3 François Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions,” Conference on Computer Vision and Pattern Recognition (2017), https://arxiv.org/abs/1610.02357.

### PDF p.293 (printed 269) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › 
images are translation-invariant. To effectively learn from data, you need to make assumptions about what you’re looking for.

Model architecture is often the difference between success and failure. If you make inappropriate architecture choices, your model may be stuck with suboptimal metrics, and no amount of training data will save it. Inversely, a good model architecture will accelerate learning and will enable your model to make efficient use of the training data available, reducing the need for large datasets. A good model architecture is one that reduces the size of the search space or otherwise makes it easier to converge to a good point of the search space. Just like feature engineering and data curation, model architecture is all about making the problem simpler for gradient descent to solve—and remember that gradient descent is a pretty stupid search process, so it needs all the help it can get. Model architecture is more an art than a science. Experienced machine learning engineers are able to intuitively cobble together high-performing models on their first try, while beginners often struggle to create a model that trains at all. The keyword here is intuitively: no one

### PDF p.256 (printed 232) — 8 Image classification › 8.1 Introduction to convnets › Introduction to ConvNets
We’re about to dive into the theory of what ConvNets are and why they have been so successful at computer vision tasks. But first, let’s take a practical look at a simple ConvNet example. It uses a ConvNet to classify MNIST digits, a task we performed in chapter 2 using a densely connected network (our test accuracy then was 97.8%). Even though the ConvNet will be basic, its accuracy will blow out of the water that of the densely connected model from chapter 2.

The following lines of code show you what a basic ConvNet looks like. It’s a stack of `Conv2D` and `MaxPooling2D` layers. You’ll see in a minute exactly what they do. We’ll build the model using the Functional API, which we introduced in the previous chapter.

Listing 8.1 Instantiating a small ConvNet

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

### PDF p.478 (printed 454) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4 Classification with a pretrained Transformer
After “Attention is all you need,” people started to notice how far Transformer training could scale, especially compared to models that had come before. As we just mentioned, one big plus was that the model is faster to train than RNNs. No more loops during training, which is always good when working with a GPU or TPU.

It is also a very data hungry model architecture. We actually got a little taste of this in the last section. While our RNN translation model plateaued in validation performance after 5 or so epochs, the Transformer model was still improving its validation score after 30 epochs of training. These observations prompted many to try scaling up the Transformer with more data, layers, and parameters—with great results. This caused a distinctive shift in the field toward large pretrained models that can cost millions to train but perform noticeably better on a wide range of problems in the text domain. For our last code example in the text section, we will revisit our IMDb text-classification problem, this time with a pretrained Transformer model.
