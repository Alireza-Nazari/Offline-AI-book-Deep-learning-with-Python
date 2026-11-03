# Generation bundle for page_042  (PDF page 42, printed page 18)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.1 A first look at a neural network
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_042.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_042.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p042_e03] (body) NOTE In machine learning, a category in a classification problem is called a class. Data points are called samples. The class associated with a specific sample is called
[p042_e04] (figure_caption) Figure 2.1 MNIST sample digits
[p042_e05] (body) a label.
[p042_e06] (body) The MNIST dataset comes preloaded in Keras, in the form of a set of four NumPy arrays.
[p042_e07] (listing_caption) Listing 2.1 Loading the MNIST dataset in Keras
[p042_e08] (code)
```
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```
[p042_e09] (body) `train_images` and `train_labels` form the training set, the data that the model will learn from. The model will then be tested on the test set, `test_images` and `test_labels`. The images are encoded as NumPy arrays, and the labels are an array of digits, ranging from 0 to 9. The images and labels have a one-to-one correspondence.
[p042_e10] (body) NOTE NumPy is a highly popular Python library for numerical computation. You will see it pop up frequently in your machine learning journey. It is rarely used to implement modern machine learning algorithms, due to lacking GPU and autodifferentiation support, but NumPy arrays are often used as a numerical data exchange format—like here, for MNIST digits and their labels.
[p042_e11] (body) Let’s look at the training data:
[p042_e12] (code)
```
>>> train_images.shape
(60000, 28, 28)
>>> len(train_labels)
60000
>>> train_labels
array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
```
[p042_e13] (body) And here’s the test data:
[p042_e14] (code)
```
>>> test_images.shape
(10000, 28, 28)
>>> len(test_labels)
10000
>>> test_labels
array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
```
[p042_e15] (body) The workflow will be as follows. First, we’ll feed the neural network the training data, `train_images` and `train_labels`. The network will then learn to associate images and

## Listings
- Listing 2.1: Loading the MNIST dataset in Keras (code elements p042_e08)

## Figures
- Figure 2.1: MNIST sample digits  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 2 The mathematical building blocks of neural networks (PDF pages 40-83)
  - 2.1 A first look at a neural network (p.41)
  - 2.2 Data representations for neural networks (p.45)
      - 2.2.1 Scalars (rank-0 tensors) (p.46)
      - 2.2.2 Vectors (rank-1 tensors) (p.46)
      - 2.2.3 Matrices (rank-2 tensors) (p.46)
      - 2.2.4 Rank-3 tensors and higher-rank tensors (p.47)
      - 2.2.5 Key attributes (p.47)
      - 2.2.6 Manipulating tensors in NumPy (p.49)
      - 2.2.7 The notion of data batches (p.49)
      - 2.2.8 Real-world examples of data tensors (p.50)
  - 2.3 The gears of neural networks: Tensor operations (p.52)
      - 2.3.1 Element-wise operations (p.53)
      - 2.3.2 Broadcasting (p.54)
      - 2.3.3 Tensor product (p.56)
      - 2.3.4 Tensor reshaping (p.58)
      - 2.3.5 Geometric interpretation of tensor operations (p.59)
      - 2.3.6 A geometric interpretation of deep learning (p.62)
  - 2.4 The engine of neural networks: Gradient-based optimization (p.63)
      - 2.4.1 What’s a derivative? (p.65)
      - 2.4.2 Derivative of a tensor operation: The gradient (p.66)
      - 2.4.3 Stochastic gradient descent (p.67)
      - 2.4.4 Chaining derivatives: The Backpropagation algorithm (p.70)
  - 2.5 Looking back at our first example (p.75)
      - 2.5.1 Reimplementing our first example from scratch (p.77)
      - 2.5.2 Running one training step (p.79)
      - 2.5.3 The full training loop (p.81)
      - 2.5.4 Evaluating the model (p.82)

## Previous page (PDF 41) text, for continuity
After reading this chapter, you’ll have an intuitive understanding of the mathematical theory behind deep learning, and you’ll be ready to start diving into modern deep learning frameworks, in chapter 3.

Running the code in this book

This book is full of runnable Python code. Each chapter is paired with a Jupyter notebook that contains all of the code from the chapter. A Jupyter notebook is a live Python scratch pad of sorts, where you can interactively run code, graph data, view images, and a lot more. You will gain a lot more practical knowledge from this book if you run and experiment with the code as you read. By far the easiest way to set up a deep learning environment to run these notebooks is Google Colaboratory (or Colab for short), a hosted environment for Jupyter notebooks that has become the industry standard for ML practitioners. With Colab, you can run the code for this book interactively in the browser, connecting to cloud runtimes with configurable hardware. By default, the notebooks in this book will run on Colab’s free GPU runtime. If you would like, you can also run these notebooks locally on your own machine. A GPU is recommended, especially as you get to the larger and more compute-intensive models later in this book. Instructions for running locally and on Colab, along with the code, can be found at https://github.com/fchollet/deep-learning-with-python-notebooks.

2.1

A first look at a neural network

Let’s look at a concrete example of a neural network that uses the machine learning library Keras to learn to classify handwritten digits. We will use Keras extensively throughout this book. It’s a simple, high-level library that will allow us to stay focused on the concepts we would like to cover.

Unless you already have experience with Keras or s …

## Next page (PDF 43) text, for continuity
labels. Finally, we’ll ask the network to produce predictions for `test_images`, and we’ll verify whether these predictions match the labels from `test_labels`.

Let’s build the network—again, remember that you aren’t expected to understand everything about this example yet.

Listing 2.2 The network architecture

The core building block of neural networks is the layer. You can think of a layer as a filter for data: some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation. A deep learning model is like a sieve for data processing, made of a succession of increasingly refined data filters—the layers.

Here, our model consists of a sequence of two `Dense` layers, which are densely connected (also called fully connected) neural layers. The second (and last) layer is a 10-way `softmax` classification layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes. To make the model ready for training, we need to pick three more things, as part of the compilation step:

A loss function—How the model will be able to measure its performance on the training data and thus how it will be able to steer itself in the right direction.

An optimizer—The mechanism through which the model will update itself based on the training data it sees, to improve its performance.

Metrics to monitor during training and testing—Here, we’ll only care about accuracy (the fraction of the images that were  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.48 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.81 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.75 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.45 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.44 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.82 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.49 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.47 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.49 (printed 25) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › The notion of data batches
In general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll come across in deep learning will be the samples axis. In the MNIST example, “samples” are images of digits.

In addition, deep learning models don’t process an entire dataset at once; rather, they break the data into small “batches,” or groups of samples with a fixed size. Concretely, here’s one batch of our MNIST digits, with a batch size of 128:

### PDF p.258 (printed 234) — 8 Image classification › 8.1 Introduction to convnets › 
Listing 8.3 Training the ConvNet on MNIST images
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_images, train_labels, epochs=5, batch_size=64)

### PDF p.230 (printed 206) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › Using built-in training and evaluation loops
Listing 7.17 The standard workflow: `compile()`, `fit()`, `evaluate()`, `predict()`
from keras.datasets import mnist
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
test_metrics = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)

### PDF p.81 (printed 57) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › The full training loop
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))

### PDF p.48 (printed 24) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › 
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

### PDF p.168 (printed 144) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › The manifold hypothesis
The input to an MNIST classifier (before preprocessing) is a 28 × 28 array of integers between 0 and 255. The total number of possible input values is thus 256 to the power of 784—much greater than the number of atoms in the universe. However, very few of these inputs would look like valid MNIST samples: actual handwritten digits only occupy a tiny subspace of the parent space of all possible 28 × 28 `uint8` arrays. What’s more, this subspace isn’t just a set of points sprinkled at random in the parent space: it is highly structured.

First, the subspace of valid handwritten digits is continuous: if you take a sample and modify it a little, it will still be recognizable as the same handwritten digit. Further, all samples in the valid subspace are connected by smooth paths that run through the subspace. This means that if you take two random MNIST digits A and B, there exists a sequence of “intermediate” images that morph A into B, such that two consecutive digits are very close to each other (see figure 5.7). Perhaps there will be a few ambiguous shapes close to the boundary between two classes, but even these shapes would still look very digit-like.

Figure 5.7 Different MNIST dig

### PDF p.164 (printed 140) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Rare features and spurious correlations
Listing 5.1 Adding white noise channels or all-zeros channels to MNIST
from keras.datasets import mnist
import numpy as np
(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
train_images_with_noise_channels = np.concatenate(
    [train_images, np.random.random((len(train_images), 784))], axis=1
)
train_images_with_zeros_channels = np.concatenate(
    [train_images, np.zeros((len(train_images), 784))], axis=1
)

### PDF p.45 (printed 21) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Data representations for neural networks
In the previous example, we started from data stored in multidimensional NumPy arrays, also called tensors. In general, all current machine learning systems use tensors as their basic data structure. Tensors are fundamental to the field—so fundamental that the TensorFlow framework was named after them. So what’s a tensor?

At its core, a tensor is a container for data—usually numerical data. So it’s a container for numbers. You may already be familiar with matrices, which are rank-2 tensors: tensors are a generalization of matrices to an arbitrary number of dimensions (note that in the context of tensors, a dimension is often called an axis). Going over the details of tensors might seem a bit abstract at first. But it’s well worth it—manipulating tensors will be the bread and butter of any machine learning code you ever write.
