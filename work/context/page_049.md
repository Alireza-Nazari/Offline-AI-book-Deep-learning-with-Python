# Generation bundle for page_049  (PDF page 49, printed page 25)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.2 Data representations for neural networks
Subsection: 2.2.7 The notion of data batches
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_049.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_049.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p049_e03] (heading_subsection) 2.2.6
[p049_e04] (heading_subsection) Manipulating tensors in NumPy
[p049_e05] (body) In the previous example, we selected a specific digit alongside the first axis using the syntax `train_images[i]`. Selecting specific elements in a tensor is called tensor slicing. Let’s look at the tensor-slicing operations you can do on NumPy arrays.
[p049_e06] (body) The following example selects digits #10 to #100 (#100 isn’t included) and puts them in an array of shape `(90,` `28,` `28)`:
[p049_e07] (code)
```
>>> my_slice = train_images[10:100]
>>> my_slice.shape
(90, 28, 28)
```
[p049_e08] (body) It’s equivalent to this more detailed notation, which specifies a start index and stop index for the slice along each tensor axis. Note that `:` is equivalent to selecting the entire axis:
[p049_e09] (code)
```
>>> my_slice = train_images[10:100, :, :]
>>> my_slice.shape
(90, 28, 28)
>>> my_slice = train_images[10:100, 0:28, 0:28]
>>> my_slice.shape
(90, 28, 28)
```
[p049_e10] (body) In general, you may select slices between any two indices along each tensor axis. For instance, to select 14 × 14 pixels in the bottom-right corner of all images, you would do this:
[p049_e11] (code)
```
my_slice = train_images[:, 14:, 14:]
```
[p049_e12] (body) It’s also possible to use negative indices. Much like negative indices in Python lists, they indicate a position relative to the end of the current axis. To crop the images to patches of 14 × 14 pixels centered in the middle, do this:
[p049_e13] (code)
```
my_slice = train_images[:, 7:-7, 7:-7]
```
[p049_e14] (heading_subsection) 2.2.7
[p049_e15] (heading_subsection) The notion of data batches
[p049_e16] (body) In general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll come across in deep learning will be the samples axis. In the MNIST example, “samples” are images of digits.
[p049_e17] (body) In addition, deep learning models don’t process an entire dataset at once; rather, they break the data into small “batches,” or groups of samples with a fixed size. Concretely, here’s one batch of our MNIST digits, with a batch size of 128:

## Code annotations printed next to listings (callouts, in order)
- Equivalent to the
- previous example
- Also equivalent to the
- previous example

## Equation candidates (formulas written inline in the text)
- [p049_e11] my_slice = train_images[:, 14:, 14:]
- [p049_e13] my_slice = train_images[:, 7:-7, 7:-7]

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

## Previous page (PDF 48) text, for continuity
Next, we display the number of axes of the tensor `train_images`, the `ndim` attribute:

Here’s its shape:

And this is its data type, the `dtype` attribute:

So what we have here is a rank-3 tensor of 8-bit integers. More precisely, it’s an array of 60,000 matrices of 28 × 28 integers. Each such matrix is a grayscale image, with coefficients between 0 and 255.

Let’s display the fourth digit in this rank-3 tensor, using the library Matplotlib (part of the standard scientific Python suite); see figure 2.2.

Listing 2.8 Displaying the fourth digit

Naturally, the corresponding label is just the

Figure 2.2 The fourth sample in our dataset

integer 9:

## Next page (PDF 50) text, for continuity
And here’s the next batch:

And the `n`th batch:

When considering such a batch tensor, the first axis (axis 0) is called the batch axis (or batch dimension). You’ll frequently encounter this term when using Keras and other deep learning libraries.

2.2.8

Real-world examples of data tensors

Let’s make data tensors more concrete with a few examples similar to what you’ll encounter later. The data you’ll manipulate will almost always fall into one of the following categories:

Vector data—Rank-2 tensors of shape `(samples,` `features)`, where each sample is a vector of numerical attributes (“features”)

Timeseries data or sequence data—Rank-3 tensors of shape `(samples,` `timesteps, ` `features)`, where each sample is a sequence (of length `timesteps`) of feature vectors

Images—Rank-4 tensors of shape `(samples,` `height,` `width,` `channels)`, where each sample is a 2D grid of pixels, and each pixel is represented by a vector of values (“channels”)

Video—Rank-5 tensors of shape `(samples,` `frames,` `height,` `width,` `channels)`, where each sample is a sequence (of length `frames`) of images

Vector data

Vector data is one of the most common cases. In such a dataset, each single data point can be encoded as a vector, and thus a batch of data will be encoded as a rank-2 tensor (that is, an array of vectors), where the first axis is the samples axis and the second axis is the features axis.

Let’s take a look at two examples:

An actuarial dataset of people, where we consider each person’s age, gender, and income. Each person can be characterized as a vector of three values, and thus an entire dataset of 100,000 people can be stored in a rank-2 tensor of shape `(100000,` `3)`.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.47 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.46 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.55 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.42 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.51 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.45 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.58 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.52 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.45 (printed 21) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Data representations for neural networks
In the previous example, we started from data stored in multidimensional NumPy arrays, also called tensors. In general, all current machine learning systems use tensors as their basic data structure. Tensors are fundamental to the field—so fundamental that the TensorFlow framework was named after them. So what’s a tensor?

At its core, a tensor is a container for data—usually numerical data. So it’s a container for numbers. You may already be familiar with matrices, which are rank-2 tensors: tensors are a generalization of matrices to an arbitrary number of dimensions (note that in the context of tensors, a dimension is often called an axis). Going over the details of tensors might seem a bit abstract at first. But it’s well worth it—manipulating tensors will be the bread and butter of any machine learning code you ever write.

### PDF p.47 (printed 23) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Key attributes
A tensor is defined by three key attributes:

Number of axes (rank)—For instance, a rank-3 tensor has three axes, and a matrix has two axes. This is also called the tensor’s `ndim` in Python libraries such as NumPy, JAX, TensorFlow, and PyTorch.

Shape—This is a tuple of integers that describes how many dimensions the tensor has along each axis. For instance, the previous matrix example has shape `(3,` `5)`, and the rank-3 tensor example has shape `(3,` `3,` `5)`. A vector has a shape with a single element, such as `(5,)`, whereas a scalar has an empty shape, `()`.

Data type (usually called `dtype` in Python libraries)—This is the type of the data contained in the tensor; for instance, a tensor’s type could be `float16`, `float32`, `float64`, `uint8`, `bool`, and so on. In TensorFlow, you are also likely to come across `string` tensors.

To make this more concrete, let’s look back at the data we processed in the MNIST example. First, we load the MNIST dataset:

### PDF p.46 (printed 22) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Vectors (rank-1 tensors)
An array of numbers is called a vector (or rank-1 tensor or 1D tensor). A rank-1 tensor has exactly one axis. The following is a NumPy vector:

### PDF p.131 (printed 107) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Preparing the data
You can’t directly feed lists of integers into a neural network. They have all different lengths, while a neural network expects to process contiguous batches of data. You have to turn your lists into tensors. There are two ways to do that:

Pad your lists so that they all have the same length, then turn them into an integer tensor of shape `(samples,` `max_length)`, and start your model with a layer capable of handling such integer tensors (the `Embedding` layer, which we’ll cover in detail later in the book).

Multi-hot encode your lists to turn them into vectors of 0s and 1s reflecting the presence or absence of all possible words. This would mean, for instance, turning the sequence `[8, 5]` into a 10,000-dimensional vector that would be all 0s except for indices 5 and 8, which would be 1s.

Let’s go with the latter solution to vectorize the data. When done manually, the process looks like the following.

### PDF p.203 (printed 179) — 6 The universal workflow of machine learning › 6.2 Developing a model › Vectorization
All inputs and targets in a neural network must be typically tensors of floating-point data (or, in specific cases, tensors of integers or strings). Whatever data you need to process—sound, images, text—you must first turn into tensors, a step called data vectorization. For instance, in the two previous text classification examples from chapter 4, we started from text represented as lists of integers (standing for sequences of words), and we used multi-hot encoding to turn them into a tensor of `float32` data. In the examples of classifying digits and predicting house prices, the data already came in vectorized form, so you were able to skip this step.

### PDF p.42 (printed 18) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
`train_images` and `train_labels` form the training set, the data that the model will learn from. The model will then be tested on the test set, `test_images` and `test_labels`. The images are encoded as NumPy arrays, and the labels are an array of digits, ranging from 0 to 9. The images and labels have a one-to-one correspondence.

NOTE NumPy is a highly popular Python library for numerical computation. You will see it pop up frequently in your machine learning journey. It is rarely used to implement modern machine learning algorithms, due to lacking GPU and autodifferentiation support, but NumPy arrays are often used as a numerical data exchange format—like here, for MNIST digits and their labels.

Let’s look at the training data:

### PDF p.51 (printed 27) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Image data
Images typically have three dimensions: height, width, and color depth. Although grayscale images (like our MNIST digits) have only a single color channel and could thus be stored in rank-2 tensors, by convention image tensors are always rank-3, with a one-dimensional color channel for grayscale images. A batch of 128 grayscale images of size 256 × 256 could thus be stored in a tensor of shape `(128,` `256,` `256, 1)`, and a batch of 128 color images could be stored in a tensor of shape `(128,` `256,` `256,` `3)` (see figure 2.4).

There are two conventions for the shapes of image tensors: the channels-last convention (which is standard in JAX and TensorFlow, as well as most other deep learning tools out there) and the channels-first convention (which is standard in PyTorch). The channels-last convention places the color-depth axis at the end: `(samples,` `height,` `width,` `color_depth)`. Meanwhile, the channels-first convention places the color depth axis right after the batch axis: `(samples,` `color_depth,` `height,` `width)`.

### PDF p.82 (printed 58) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Summary
Tensors form the foundation of modern machine learning systems. They come in various flavors of `dtype`, `rank`, and `shape`.

You can manipulate numerical tensors via tensor operations (such as addition, tensor product, or element-wise multiplication), which can be interpreted as encoding geometric transformations. In general, everything in deep learning is amenable to a geometric interpretation.

Deep learning models consist of chains of simple tensor operations, parameterized by weights, which are themselves tensors. The weights of a model are where its “knowledge” is stored.

Learning means finding a set of values for the model’s weights that minimizes a loss function for a given set of training data samples and their corresponding targets.

Learning happens by drawing random batches of data samples and their targets and computing the gradient of the model parameters with respect to the loss on the batch. The model parameters are then moved a bit (the magnitude of the
