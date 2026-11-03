# Generation bundle for page_031  (PDF page 31, printed page 7)

Page kind: content
Chapter: 1 What is deep learning?
Section: 1.5 The “deep” in “deep learning”
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_031.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_031.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p031_e03] (heading_section) 1.5
[p031_e04] (heading_section) The “deep” in “deep learning”
[p031_e05] (body) Deep learning is a specific subfield of machine learning; it’s a new take on learning representations from data, which emphasizes learning successive layers of increasingly meaningful representations. The “deep” in “deep learning” isn’t a reference to any kind of deeper understanding achieved by the approach; rather, it stands for this idea of successive layers of representations. How many layers contribute to a model of the data is called the depth of the model. Other appropriate names for the field could have been layered representations learning or hierarchical representations learning. Modern deep learning often involves tens or even hundreds of successive layers of representations, and they’re all learned automatically from exposure to training data. Meanwhile, other approaches to machine learning tend to focus on learning only one or two layers of representations of the data (say, taking a pixel histogram and then applying a classification rule); hence they’re sometimes called shallow learning.
[p031_e06] (body) In deep learning, these layered representations are learned via models called neural networks, structured in literal layers stacked on top of each other. The term neural network is a reference to neurobiology, but although some of the central concepts in deep learning were developed in part by drawing inspiration from our understanding of the brain (in particular, the visual cortex), deep learning models are not models of the brain. There’s no evidence that the brain implements anything like the learning mechanisms used in modern deep learning models. You may come across pop science articles proclaiming that deep learning works like the brain or is modeled after the brain, but that isn’t the case. It would be confusing and counterproductive for newcomers to the field to think of deep learning as being in any way related to neurobiology; you don’t need that shroud of “just like our minds” mystique and mystery, and you may as well forget anything you may have read about hypothetical links between deep learning and biology. For our purposes, deep learning is a mathematical framework for learning representations from data. What do the representations learned by a deep learning algorithm look like? Let’s examine how a network several layers deep (see figure 1.5) transforms an image of a digit to recognize what digit it is.
[p031_e07] (figure_caption) Figure 1.5 A deep neural network for digit classification

## Figures
- Figure 1.5: A deep neural network for digit classification  | labels inside the figure: Layer 1, Original, input, Layer 2, Layer 3, Layer 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

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

## Previous page (PDF 30) text, for continuity
In this new coordinate system, the coordinates of our points can be said to be a new representation of our data. And it’s a good one! With this representation, the black/ white classification problem can be expressed as a simple rule: “Black points are such that x > 0,” or “White points are such that x < 0.” This new representation, combined with this simple rule, neatly solves the classification problem.

In this case, we defined the coordinate change by hand: we used our human intelligence to come up with our own appropriate representation of the data. This is fine for such an extremely simple problem, but could you do the same if the task were to classify images of handwritten digits? Could you write explicit, computer-executable image transformations that would illuminate the difference between a 6 and an 8, between a 1 and a 7, across all kinds of different handwritings? This is possible to an extent. Rules based on representations of digits such as “counting the number of closed loops” or vertical and horizontal pixel histograms can do a decent job at telling apart handwritten digits. But finding such useful representations by hand is hard work, and as you can imagine the resulting rule-based system would be brittle and a nightmare to maintain. Every time you would come across a new example of handwriting that would break your carefully thought-out rules, you would have to add new data transformations and new rules, while taking into account their interaction with every previous rule. You’re probably thinking, if this process is so painful, could we automate it? What if we tried systematically searching for different sets of automatically generated representations of the data and rules based on them, identifying good ones using the percentage of digits being corre …

## Next page (PDF 32) text, for continuity
As you can see in figure 1.6, the network transforms the digit image into representations that are increasingly different from the original image and increasingly informative about the final result. You can think of a deep network as a multistage information-distillation process, where information goes through successive filters and comes out increasingly purified (that is, useful with regard to some task).

Figure 1.6 Deep representations learned by a digit-classification model

So that’s what deep learning is, technically: a multistage way to learn data representations. It’s a simple idea, but, as it turns out, very simple mechanisms, sufficiently scaled, can end up looking like magic.

1.6

Understanding how deep learning works, in three figures

At this point, you know that machine learning is about mapping inputs (such as images) to targets (such as the label “cat”), which is done by observing many examples of inputs and targets. You also know that deep neural networks do this input-to-target mapping via a deep sequence of simple data transformations (layers) and that these data transformations are learned by exposure to examples. Now let’s look at how this learning happens, concretely.

The specification of what a layer does to its input data is stored in the layer’s weights, which in essence are a bunch of numbers. In technical terms, we’d say that the transformation implemented by a layer is parameterized by its weights (see figure 1.7). (Weights are also sometimes called the parameters of a layer.) In this context, learning means finding a set of values for the weights of all layers in a network, such that the network will correctly map example inputs to their associated targets. But here’s the thing: a deep neural network can contain tens of millions of parame …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.34 — 1 What is deep learning? › 1.7 What makes deep learning different
- PDF p.29 — 1 What is deep learning? › 1.4 Learning rules and representations from data
- PDF p.28 — 1 What is deep learning? › 1.4 Learning rules and representations from data
- PDF p.171 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.308 — 10 Interpreting what ConvNets learn › 
- PDF p.622 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.8 — contents › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.620 (printed 596) — 20 Conclusions › 20.1 Key concepts in review › 20.1.1 Various approaches to artificial intelligence
First, deep learning isn’t synonymous with artificial intelligence (AI), or even with machine learning:

Artificial intelligence (AI) is an ancient, broad field that can generally be understood as “all attempts to automate human cognitive processes.” This can range from the very basic, such as an Excel spreadsheet, to the very advanced, like a humanoid robot that can walk and talk.

Machine learning is a specific subfield of AI that aims at automatically developing programs (called models) purely from exposure to training data. This process of turning data into a program is called learning. Although machine learning has been around for a long time, it only started to take off in the 1990s, before becoming the dominant form of AI in the 2000s.

Deep learning is one of many branches of machine learning, where the models are long chains of geometric transformations, applied one after the other. These operations are structured into modules called layers: deep learning models are typically stacks of layers—or, more generally, graphs of layers. These layers are parameterized by weights, which are the parameters learned during training. The knowledge of a model is stored in its weights, a

### PDF p.34 (printed 10) — 1 What is deep learning? › 1.7 What makes deep learning different › What makes deep learning different
Is there anything special about deep neural networks that makes them the “right” approach for companies to invest in and for researchers to flock to? Will we still be using deep neural networks in 20 years?

Deep learning has several properties that justify its status as an AI revolution, and it’s here to stay. We may not be using neural networks many decades from now, but whatever we use will directly inherit from modern deep learning and its core concepts. These important properties can be broadly sorted into three categories:

Simplicity—Deep learning makes problem solving much easier, because it automates what used to be the most crucial step in a machine learning workflow: feature engineering. Previous machine learning techniques—shallow learning—only involved transforming the input data into one or two successive representation spaces, which wasn’t expressive enough for most problems. As such, humans had to go to great lengths to make the initial input data more amenable to processing by these methods: they had to manually engineer good representations for their data. This is called feature engineering. Deep learning, on the other hand, completely automates this step: with de

### PDF p.43 (printed 19) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
The core building block of neural networks is the layer. You can think of a layer as a filter for data: some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation. A deep learning model is like a sieve for data processing, made of a succession of increasingly refined data filters—the layers.

Here, our model consists of a sequence of two `Dense` layers, which are densely connected (also called fully connected) neural layers. The second (and last) layer is a 10-way `softmax` classification layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes. To make the model ready for training, we need to pick three more things, as part of the compilation step:

A loss function—How the model will be able to measure its performance on the training data and thus how it will be able to steer itse

### PDF p.171 (printed 147) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Figure 5.10 Going from a random model to an overfit model and achieving a robust fit as an intermediate state

Deep learning models tend to be structured in a way that mirrors the “shape” of the information in their training data (via architecture priors). This is the case in particular for image-processing models (see chapters 8–12) and sequence-processing models (see chapter 13). More generally, deep neural networks structure their learned representations in a hierarchical and modular way, which echoes the way natural data is organized.

### PDF p.308 (printed 284) — 10 Interpreting what ConvNets learn ›  › This chapter covers
Interpreting how ConvNets decompose an input

image

Visualizing the filters learned by ConvNets

Visualizing areas in an image responsible for a

certain classification decision

A fundamental problem when building a computer vision application is that of interpretability: Why did your classifier think a particular image contained a fridge, when all you can see is a truck? This is especially relevant to use cases where deep learning is used to complement human expertise, such as medical imaging use cases. This chapter will get you familiar with a range of different techniques for visualizing what ConvNets learn and understanding the decisions they make.

It’s often said that deep learning models are “black boxes”: they learn representations that are difficult to extract and present in a human-readable form. Although this is partially true for certain types of deep learning models, it’s definitely not true for ConvNets. The representations learned by ConvNets are highly amenable to visualization, in large part because they’re representations of visual concepts. Since 2013, a

### PDF p.620 (printed 596) — 20 Conclusions › 20.1 Key concepts in review › 20.1.2 What makes deep learning special within the field of machine learning
In the span of only a few years, deep learning has achieved tremendous breakthroughs across a wide range of tasks that have been historically perceived as extremely difficult for computers, especially in the area of machine perception: extracting useful information from images, videos, sound, and more. Given sufficient training data (in particular, training data appropriately labeled by humans), deep learning makes it possible

### PDF p.8 (printed viii) — contents ›  › contents
preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

### PDF p.622 (printed 598) — 20 Conclusions › 20.1 Key concepts in review › 
transformation is that it must be differentiable, which is required for us to be able to learn its parameters via gradient descent. Intuitively, this means the geometric morphing from inputs to outputs must be smooth and continuous—a significant constraint.

The entire process of applying this complex geometric transformation to the input data can be visualized in 3D by imagining a person trying to uncrumple a paper ball: the crumpled paper ball is the manifold of the input data that the model starts with. Each movement operated by the person on the paper ball is similar to a simple geometric transformation operated by one layer. The full uncrumpling gesture sequence is the complex transformation of the entire model. Deep learning models are mathematical machines for uncrumpling complicated manifolds of high-dimensional data. That’s the magic of deep learning—turning meaning into vectors, into geometric spaces, and then incrementally learning complex geometric transformations that map one space to another. All you need are spaces of sufficiently high dimensionality to capture the full scope of the relationships found in the original data. The whole thing hinges on two core ideas: t
