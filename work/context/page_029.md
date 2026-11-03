# Generation bundle for page_029  (PDF page 29, printed page 5)

Page kind: content
Chapter: 1 What is deep learning?
Section: 1.4 Learning rules and representations from data
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_029.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_029.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p029_e03] (body) output. The measurement is used as a feedback signal to adjust the way the algorithm works. This adjustment step is what we call learning. A machine learning model transforms its input data into meaningful outputs, a process that is “learned” from exposure to known examples of inputs and outputs. Therefore, the central problem in machine learning and deep learning is to meaningfully transform data: in other words, to learn useful representations of the input data at hand—representations that get us closer to the expected output. Before we go any further, what’s a representation? At its core, it’s a different way to look at data to represent or encode data. For instance, a color image can be encoded in the RGB format (red-green-blue) or in the HSV format (hue-saturation-value): these are two different representations of the same data. Some tasks that may be difficult with one representation can become easy with another. For example, the task “Select all red pixels in the image” is simpler in the RGB format, whereas “Make the image less saturated” is simpler in the HSV format. Machine learning models are all about finding appropriate representations for their input data—transformations of the data that make it more amenable to the task at hand. Let’s make this concrete. Consider an x-axis, a y-axis, and some points represented by their coordinates in the (x, y) system, as shown in figure 1.3. As you can see, we have a few white points and a few black points. Let’s say we want to develop an algorithm that can take a point’s (x, y) coordinates and output whether that point is likely black or white. In this case,
[p029_e04] (bullet) The inputs are the coordinates of our points.
[p029_e05] (bullet) The expected outputs are the colors of our points.
[p029_e06] (figure_caption) Figure 1.3 Some sample data
[p029_e07] (bullet) A way to measure whether our algorithm is doing a good job could be, for instance, the percentage of points that are being correctly classified.
[p029_e08] (body) What we need here is a new representation of our data that cleanly separates the white points from the black points. One transformation we could use, among many other possibilities, would be a coordinate change, illustrated in figure 1.4.
[p029_e09] (figure_caption) Figure 1.4 Coordinate change

## Figures
- Figure 1.3: Some sample data  | labels inside the figure: y, x
- Figure 1.4: Coordinate change  | labels inside the figure: y, x, y, x, y, x

## Chapter outline
Chapter: 1 What is deep learning? (PDF pages 25-39)
  - 1.1 Artificial intelligence, machine learning, and deep learning (p.26)
  - 1.2 Artificial intelligence (p.26)
  - 1.3 Machine learning (p.27)
  - 1.4 Learning rules and representations from data (p.28)
  - 1.5 The “deep” in “deep learning” (p.31)
  - 1.6 Understanding how deep learning works, in three figures (p.32)
  - 1.7 What makes deep learning different (p.34)
  - 1.8 The age of generative AI (p.35)
  - 1.9 What deep learning has achieved so far (p.35)
  - 1.10 Beware of the short-term hype (p.36)
  - 1.11 Summer can turn to winter (p.38)
  - 1.12 The promise of AI (p.38)

## Previous page (PDF 28) text, for continuity
Engine to perform. Machine learning turns this around: the machine looks at the input data and the corresponding answers, and figures out what the rules should be (figure 1.2).

A machine learning system is trained rather than explicitly programmed. It’s presented with many examples relevant to a task, and

Figure 1.2 Machine learning: a new

it finds statistical structure in these examples

programming paradigm

that eventually allows the system to come up with rules for automating the task. For instance, if you wished to automate the task of tagging your vacation pictures, you could present a machine learning system with many examples of pictures already tagged by humans, and the system would learn statistical rules for associating specific pictures to specific tags like “landscape” or “food.”

Although machine learning only started to flourish in the 1990s, it has quickly become the most popular and most successful subfield of AI, a trend driven by the availability of faster hardware and larger datasets. Machine learning is related to mathematical statistics, but it differs from statistics in several important ways—in the same sense that medicine is related to chemistry but cannot be reduced to chemistry, as medicine deals with its own distinct systems with their own distinct properties. Unlike statistics, machine learning tends to deal with large, complex datasets (such as a dataset of millions of images, each consisting of tens of thousands of pixels) for which classical statistical analysis such as Bayesian analysis would be impractical. As a result, machine learning, and especially deep learning, exhibits comparatively little mathematical theory—maybe too little—and is fundamentally an engineering discipline. Unlike theoretical physics or mathematics, machine lea …

## Next page (PDF 30) text, for continuity
In this new coordinate system, the coordinates of our points can be said to be a new representation of our data. And it’s a good one! With this representation, the black/ white classification problem can be expressed as a simple rule: “Black points are such that x > 0,” or “White points are such that x < 0.” This new representation, combined with this simple rule, neatly solves the classification problem.

In this case, we defined the coordinate change by hand: we used our human intelligence to come up with our own appropriate representation of the data. This is fine for such an extremely simple problem, but could you do the same if the task were to classify images of handwritten digits? Could you write explicit, computer-executable image transformations that would illuminate the difference between a 6 and an 8, between a 1 and a 7, across all kinds of different handwritings? This is possible to an extent. Rules based on representations of digits such as “counting the number of closed loops” or vertical and horizontal pixel histograms can do a decent job at telling apart handwritten digits. But finding such useful representations by hand is hard work, and as you can imagine the resulting rule-based system would be brittle and a nightmare to maintain. Every time you would come across a new example of handwriting that would break your carefully thought-out rules, you would have to add new data transformations and new rules, while taking into account their interaction with every previous rule. You’re probably thinking, if this process is so painful, could we automate it? What if we tried systematically searching for different sets of automatically generated representations of the data and rules based on them, identifying good ones using the percentage of digits being corre …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.32 — 1 What is deep learning? › 1.6 Understanding how deep learning works, in three figures
- PDF p.260 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.183 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.34 — 1 What is deep learning? › 1.7 What makes deep learning different
- PDF p.171 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.31 — 1 What is deep learning? › 1.5 The “deep” in “deep learning”
- PDF p.622 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.63 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.32 (printed 8) — 1 What is deep learning? › 1.6 Understanding how deep learning works, in three figures › Understanding how deep learning works, in three figures
At this point, you know that machine learning is about mapping inputs (such as images) to targets (such as the label “cat”), which is done by observing many examples of inputs and targets. You also know that deep neural networks do this input-to-target mapping via a deep sequence of simple data transformations (layers) and that these data transformations are learned by exposure to examples. Now let’s look at how this learning happens, concretely.

The specification of what a layer does to its input data is stored in the layer’s weights, which in essence are a bunch of numbers. In technical terms, we’d say that the transformation implemented by a layer is parameterized by its weights (see figure 1.7). (Weights are also sometimes called the parameters of a layer.) In this context, learning means finding a set of values for the weights of all layers in a network, such that the network will correctly map example inputs to their associated targets. But here’s the thing: a deep neural network can contain tens of millions of parameters. Finding the correct

### PDF p.260 (printed 236) — 8 Image classification › 8.1 Introduction to convnets › 
Convolutions operate over rank-3 tensors, called feature maps, with two spatial axes (height and width) as well as a depth axis (also called the channels axis). For an RGB image, the dimension of the depth axis is 3, because the image has three color channels: red, green, and blue. For a black-and-white picture, like the MNIST digits, the depth is 1 (levels of gray). The convolution operation extracts patches from its input feature map and applies the same transformation to all of these patches, producing an output feature map. This output feature map is still a rank-3 tensor: it has a width and a height. Its depth can be arbitrary because the output depth is a parameter of the layer, and the different channels in that depth axis no longer stand for specific colors as in RGB input; rather, they stand for filters. Filters encode specific aspects of the input data: at a high level, a single filter could encode the concept “presence of a face in the input,” for instance.

In the MNIST example, the first convolution layer takes a feature map of size `(28, ` `28,` `1)` and outputs a feature map of size `(26,` `26,` `64)`: it computes 64 filters over its input. Each of these 64 output ch

### PDF p.183 (printed 159) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Feature engineering
Feature engineering is the process of using your own knowledge about the data and about the machine learning algorithm at hand (in this case, a neural network) to make the algorithm work better by applying hardcoded (non-learned) transformations to the data before it goes into the model. In many cases, it isn’t reasonable to expect a machine learning model to be able to learn from completely arbitrary data. The data needs to be presented to the model in a way that will make the model’s job easier.

Let’s look at an intuitive example. Suppose you’re trying to develop a model that can take as input an image of a clock and can output the time of day (see figure 5.17). If you choose to use the raw pixels of the image as input data, then you have a difficult machine learning problem on your hands. You’ll need a convolutional neural network to solve it, and you’ll have to expend quite a bit of computational resources to train the network. But if you already understand the problem at a high level (you understand how humans read time on a clock face), then you can come up with much better input features for a machine learning algorithm: for instance, it’s easy to write a five-line Python s

### PDF p.32 (printed 8) — 1 What is deep learning? › 1.6 Understanding how deep learning works, in three figures › 
As you can see in figure 1.6, the network transforms the digit image into representations that are increasingly different from the original image and increasingly informative about the final result. You can think of a deep network as a multistage information-distillation process, where information goes through successive filters and comes out increasingly purified (that is, useful with regard to some task).

Figure 1.6 Deep representations learned by a digit-classification model

So that’s what deep learning is, technically: a multistage way to learn data representations. It’s a simple idea, but, as it turns out, very simple mechanisms, sufficiently scaled, can end up looking like magic.

### PDF p.133 (printed 109) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Having 16 units means the weight matrix `W` will have shape `(input_dimension,` `16)`: the dot product with `W` will project the input data onto a 16-dimensional representation space (and then you’ll add the bias vector `b` and apply the `relu` operation). You can intuitively understand the dimensionality of your representation space as “how much freedom you’re allowing the model to have when learning internal representations.” Having more units (a higher-dimensional representation space) allows your model to learn more complex representations, but it makes the model more computationally expensive and may lead to learning unwanted patterns (patterns that will improve performance on the training data but not on the test data).

The intermediate layers use `relu` as their activation function, and the final layer uses a sigmoid activation to output a probability (a score between 0 and 1, indicating how likely the review is to be positive). A `relu` (rectified linear unit) is a function meant to zero-out negative values (see figure 4.2), whereas a sigmoid “squashes” arbitrary values into the `[0, 1]` interval (see figure 4.3), outputting something that can be interpreted as a probabili

### PDF p.43 (printed 19) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
The core building block of neural networks is the layer. You can think of a layer as a filter for data: some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation. A deep learning model is like a sieve for data processing, made of a succession of increasingly refined data filters—the layers.

Here, our model consists of a sequence of two `Dense` layers, which are densely connected (also called fully connected) neural layers. The second (and last) layer is a 10-way `softmax` classification layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes. To make the model ready for training, we need to pick three more things, as part of the compilation step:

A loss function—How the model will be able to measure its performance on the training data and thus how it will be able to steer itse

### PDF p.535 (printed 511) — 17 Image generation › 17.1 Deep learning for image generation › 
the original image, via a decoder module (see figure 17.3). It’s then trained by using as target data the same images as the input images, meaning the autoencoder learns to reconstruct the original inputs. By imposing various constraints on the code (the output of the encoder), you can get the autoencoder to learn more or less interesting latent representations of the data. Most commonly, you’ll constrain the code to be low dimensional and sparse (mostly zeros), in which case the encoder acts as a way to compress the input data into fewer bits of information.

Figure 17.3 An autoencoder: mapping an input `x` to a compressed representation and then decoding it back as `x'`

In practice, such classical autoencoders don’t lead to particularly useful or nicely structured latent spaces. They’re not much good at compression either. For these reasons, they have largely fallen out of fashion. VAEs, however, augment autoencoders with a little bit of statistical magic that forces them to learn continuous, highly structured latent spaces. They have turned out to be a powerful tool for image generation.

A VAE, instead of compressing its input image into a fixed code in the latent space, turns

### PDF p.308 (printed 284) — 10 Interpreting what ConvNets learn ›  › This chapter covers
Interpreting how ConvNets decompose an input

image

Visualizing the filters learned by ConvNets

Visualizing areas in an image responsible for a

certain classification decision

A fundamental problem when building a computer vision application is that of interpretability: Why did your classifier think a particular image contained a fridge, when all you can see is a truck? This is especially relevant to use cases where deep learning is used to complement human expertise, such as medical imaging use cases. This chapter will get you familiar with a range of different techniques for visualizing what ConvNets learn and understanding the decisions they make.

It’s often said that deep learning models are “black boxes”: they learn representations that are difficult to extract and present in a human-readable form. Although this is partially true for certain types of deep learning models, it’s definitely not true for ConvNets. The representations learned by ConvNets are highly amenable to visualization, in large part because they’re representations of visual concepts. Since 2013, a
