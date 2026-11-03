# Generation bundle for page_030  (PDF page 30, printed page 6)

Page kind: content
Chapter: 1 What is deep learning?
Section: 1.4 Learning rules and representations from data
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_030.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_030.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p030_e03] (body) In this new coordinate system, the coordinates of our points can be said to be a new representation of our data. And it’s a good one! With this representation, the black/ white classification problem can be expressed as a simple rule: “Black points are such that x > 0,” or “White points are such that x < 0.” This new representation, combined with this simple rule, neatly solves the classification problem.
[p030_e04] (body) In this case, we defined the coordinate change by hand: we used our human intelligence to come up with our own appropriate representation of the data. This is fine for such an extremely simple problem, but could you do the same if the task were to classify images of handwritten digits? Could you write explicit, computer-executable image transformations that would illuminate the difference between a 6 and an 8, between a 1 and a 7, across all kinds of different handwritings? This is possible to an extent. Rules based on representations of digits such as “counting the number of closed loops” or vertical and horizontal pixel histograms can do a decent job at telling apart handwritten digits. But finding such useful representations by hand is hard work, and as you can imagine the resulting rule-based system would be brittle and a nightmare to maintain. Every time you would come across a new example of handwriting that would break your carefully thought-out rules, you would have to add new data transformations and new rules, while taking into account their interaction with every previous rule. You’re probably thinking, if this process is so painful, could we automate it? What if we tried systematically searching for different sets of automatically generated representations of the data and rules based on them, identifying good ones using the percentage of digits being correctly classified in some development dataset as feedback? We would then be doing machine learning. Learning, in the context of machine learning, describes an automatic search process for data transformations that produce useful representations of some data, guided by some feedback signal—representations that are amenable to simpler rules solving the task at hand. These transformations can be coordinate changes (like in our 2D coordinates classification example) or a histogram of pixels and counting loops (like in our digits classification example), but they could also be linear projections, translations, and nonlinear operations (such as “Select all points such that x > 0”), and so on. Machine learning algorithms aren’t usually creative in finding these transformations; they’re merely searching through a predefined set of operations, called a hypothesis space. For instance, the space of all possible coordinate changes would be our hypothesis space in the 2D coordinates classification example. So that’s what machine learning is, concisely: searching for useful representations and rules over some input data, within a predefined space of possibilities, using guidance from a feedback signal. This simple idea allows us to solve a remarkably broad range of intellectual tasks, from autonomous driving to natural language question-answering. Now that you understand what we mean by learning, let’s take a look at what makes deep learning special.

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

## Previous page (PDF 29) text, for continuity
output. The measurement is used as a feedback signal to adjust the way the algorithm works. This adjustment step is what we call learning. A machine learning model transforms its input data into meaningful outputs, a process that is “learned” from exposure to known examples of inputs and outputs. Therefore, the central problem in machine learning and deep learning is to meaningfully transform data: in other words, to learn useful representations of the input data at hand—representations that get us closer to the expected output. Before we go any further, what’s a representation? At its core, it’s a different way to look at data to represent or encode data. For instance, a color image can be encoded in the RGB format (red-green-blue) or in the HSV format (hue-saturation-value): these are two different representations of the same data. Some tasks that may be difficult with one representation can become easy with another. For example, the task “Select all red pixels in the image” is simpler in the RGB format, whereas “Make the image less saturated” is simpler in the HSV format. Machine learning models are all about finding appropriate representations for their input data—transformations of the data that make it more amenable to the task at hand. Let’s make this concrete. Consider an x-axis, a y-axis, and some points represented by their coordinates in the (x, y) system, as shown in figure 1.3. As you can see, we have a few white points and a few black points. Let’s say we want to develop an algorithm that can take a point’s (x, y) coordinates and output whether that point is likely black or white. In this case,

The inputs are the coordinates of our points.

The expected outputs are the colors of our points.

Figure 1.3 Some sample data

A way to measure whether our algori …

## Next page (PDF 31) text, for continuity
1.5

The “deep” in “deep learning”

Deep learning is a specific subfield of machine learning; it’s a new take on learning representations from data, which emphasizes learning successive layers of increasingly meaningful representations. The “deep” in “deep learning” isn’t a reference to any kind of deeper understanding achieved by the approach; rather, it stands for this idea of successive layers of representations. How many layers contribute to a model of the data is called the depth of the model. Other appropriate names for the field could have been layered representations learning or hierarchical representations learning. Modern deep learning often involves tens or even hundreds of successive layers of representations, and they’re all learned automatically from exposure to training data. Meanwhile, other approaches to machine learning tend to focus on learning only one or two layers of representations of the data (say, taking a pixel histogram and then applying a classification rule); hence they’re sometimes called shallow learning.

In deep learning, these layered representations are learned via models called neural networks, structured in literal layers stacked on top of each other. The term neural network is a reference to neurobiology, but although some of the central concepts in deep learning were developed in part by drawing inspiration from our understanding of the brain (in particular, the visual cortex), deep learning models are not models of the brain. There’s no evidence that the brain implements anything like the learning mechanisms used in modern deep learning models. You may come across pop science articles proclaiming that deep learning works like the brain or is modeled after the brain, but that isn’t the case. It would be confusing and counterproduct …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.28 — 1 What is deep learning? › 1.4 Learning rules and representations from data
- PDF p.32 — 1 What is deep learning? › 1.6 Understanding how deep learning works, in three figures
- PDF p.183 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.34 — 1 What is deep learning? › 1.7 What makes deep learning different
- PDF p.41 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.26 — 1 What is deep learning? › 1.2 Artificial intelligence
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.184 — 5 Fundamentals of machine learning › 5.4 Improving generalization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.28 (printed 4) — 1 What is deep learning? › 1.4 Learning rules and representations from data › Learning rules and representations from data
To define deep learning and understand the difference between deep learning and other machine learning approaches, first we need some idea of what machine learning algorithms do. We just stated that machine learning discovers rules to execute a data processing task, given examples of what’s expected. So, to do machine learning, we need three things:

Input data points—For instance, if the task is speech recognition, these data points could be sound files of people speaking. If the task is image tagging, they could be pictures.

Examples of the expected output—In a speech-recognition task, these could be human-generated transcripts of sound files. In an image task, expected outputs could be tags such as “dog,” “cat,” and so on.

A way to measure whether the algorithm is doing a good job—This is necessary to determine the distance between the algorithm’s current output and its expected

### PDF p.184 (printed 160) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.17 Feature engineering for reading the time on a clock

You can go even further: do a coordinate change and express the `(x, y)` coordinates as polar coordinates with regard to the center of the image. Your input will become the angle `theta` of each clock hand. At this point, your features are making the problem so easy that no machine learning is required; a simple rounding operation and dictionary lookup are enough to recover the approximate time of day.

That’s the essence of feature engineering: making a problem easier by expressing it in a simpler way. Make the latent manifold smoother, simpler, and better organized. It usually requires understanding the problem in depth. Before deep learning, feature engineering used to be the most important part of the machine learning workflow because classical shallow algorithms didn’t have hypothesis spaces rich enough to learn useful features by themselves. The way you presented the data to the algorithm was absolutely critical to its success. For instance, before convolutional neural networks became successful on the MNIST digit-classification problem, solutions were typically based on hardcoded features such as the number of 

### PDF p.41 (printed 17) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › A first look at a neural network
Let’s look at a concrete example of a neural network that uses the machine learning library Keras to learn to classify handwritten digits. We will use Keras extensively throughout this book. It’s a simple, high-level library that will allow us to stay focused on the concepts we would like to cover.

Unless you already have experience with Keras or similar libraries, you won’t understand everything about this first example right away. That’s fine. In a few sections, we’ll review each element in the example and explain it in detail. So don’t worry if some steps seem arbitrary or look like magic to you! We’ve got to start somewhere. The problem we’re trying to solve here is to classify grayscale images of handwritten digits (28 × 28 pixels) into their 10 categories (0 through 9). We’ll use the MNIST dataset, a classic in the machine learning community, which has been around almost as long as the field itself and has been intensively studied. It’s a set of 60,000 training images, plus 10,000 test images, assembled by the National Institute of Standards and Technology (the NIST in MNIST) in the 1980s. You can think of “solving” MNIST as the “Hello World” of deep learning—it’s what you d

### PDF p.134 (printed 110) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
So the layer could only learn linear transformations (affine transformations) of the input data: the hypothesis space of the layer would be the set of all possible linear transformations of the input data into a 16-dimensional space. Such a hypothesis space is too restricted and wouldn’t benefit from multiple layers of representations because a deep stack of linear layers would still implement a linear operation: adding more layers wouldn’t extend the hypothesis space (as you saw in chapter 2). To get access to a much richer hypothesis space that would benefit from deep representations, you need a non-linearity or activation function. `relu` is the most popular activation function in deep learning, but there are many other candidates, which all come with similarly strange names: `prelu`, `elu`, and so on.

Figure 4.2 The rectified linear unit function Figure 4.3 The sigmoid function

Finally, you need to choose a loss function and an optimizer. Because you’re facing a binary classification problem and the output of your model is a probability (you end your model with a single-unit layer with a sigmoid activation), it’s best to use the `binary_crossentropy` loss. It isn’t the only v

### PDF p.406 (printed 382) — 14 Text classification › 14.1 A brief history of natural language processing › 
Despite the limited nature of the demo, the authors claimed that within five years, translation would be a solved problem. Funding poured in for the better part of a decade. However, generalizing such a system proved to be maddeningly difficult. Words change their meaning dramatically depending on context. Any grammar rules needed countless exceptions. Developing a program that could shine on a few handpicked examples was simple enough, but building a robust system that could compete with human translators was another matter. An influential US report a decade later picked apart the lack of progress, and funding dried up. Despite these setbacks and repeated swings from excitement to disillusionment, handcrafted rules held out as the dominant approach well into the 1990s. The problems were obvious, but there was simply no viable alternative to writing down symbolic rules to describe grammar. However, as faster computers and greater quantities of data became available in the late 1980s, research began to head in a new direction. When you find yourself building systems that are big piles of ad hoc rules, as a clever engineer, you’re likely to start asking, “Could I use a corpus of data

### PDF p.168 (printed 144) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › The manifold hypothesis
The input to an MNIST classifier (before preprocessing) is a 28 × 28 array of integers between 0 and 255. The total number of possible input values is thus 256 to the power of 784—much greater than the number of atoms in the universe. However, very few of these inputs would look like valid MNIST samples: actual handwritten digits only occupy a tiny subspace of the parent space of all possible 28 × 28 `uint8` arrays. What’s more, this subspace isn’t just a set of points sprinkled at random in the parent space: it is highly structured.

First, the subspace of valid handwritten digits is continuous: if you take a sample and modify it a little, it will still be recognizable as the same handwritten digit. Further, all samples in the valid subspace are connected by smooth paths that run through the subspace. This means that if you take two random MNIST digits A and B, there exists a sequence of “intermediate” images that morph A into B, such that two consecutive digits are very close to each other (see figure 5.7). Perhaps there will be a few ambiguous shapes close to the boundary between two classes, but even these shapes would still look very digit-like.

Figure 5.7 Different MNIST dig

### PDF p.133 (printed 109) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Having 16 units means the weight matrix `W` will have shape `(input_dimension,` `16)`: the dot product with `W` will project the input data onto a 16-dimensional representation space (and then you’ll add the bias vector `b` and apply the `relu` operation). You can intuitively understand the dimensionality of your representation space as “how much freedom you’re allowing the model to have when learning internal representations.” Having more units (a higher-dimensional representation space) allows your model to learn more complex representations, but it makes the model more computationally expensive and may lead to learning unwanted patterns (patterns that will improve performance on the training data but not on the test data).

The intermediate layers use `relu` as their activation function, and the final layer uses a sigmoid activation to output a probability (a score between 0 and 1, indicating how likely the review is to be positive). A `relu` (rectified linear unit) is a function meant to zero-out negative values (see figure 4.2), whereas a sigmoid “squashes” arbitrary values into the `[0, 1]` interval (see figure 4.3), outputting something that can be interpreted as a probabili

### PDF p.42 (printed 18) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
NOTE In machine learning, a category in a classification problem is called a class. Data points are called samples. The class associated with a specific sample is called

Figure 2.1 MNIST sample digits

a label.

The MNIST dataset comes preloaded in Keras, in the form of a set of four NumPy arrays.

Listing 2.1 Loading the MNIST dataset in Keras
