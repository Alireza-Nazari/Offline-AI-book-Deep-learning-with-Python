# Generation bundle for page_052  (PDF page 52, printed page 28)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.3 The gears of neural networks: Tensor operations
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_052.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_052.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p052_e03] (body) With the channels-first convention, the previous examples would become `(128,` `1,` `256,` `256)` and `(128,` `3,` `256,` `256)`. The Keras API provides support for both formats.
[p052_e04] (heading_subsubsection) Video data
[p052_e05] (body) Video data is one of the few types of real-world data for which you’ll need rank-5 tensors. A video can be understood as a sequence of frames, each frame being a color image. Because each frame can be stored in a rank-3 tensor `(height,` `width,`
[p052_e06] (figure_caption) Figure 2.4 A rank-4 image data tensor
[p052_e07] (body) `color_depth)`, a sequence of frames can be stored in a rank-4 tensor `(frames,` `height,` `width, color_depth)`, and thus a batch of different videos can be stored in a rank-5 tensor of shape `(samples,` `frames,` `height,` `width,` `color_depth)`.
[p052_e08] (body) For instance, a 60-second, 144 × 256 YouTube video clip sampled at 4 frames per second would have 240 frames. A batch of four such video clips would be stored in a tensor of shape `(4,` `240,` `144,` `256,` `3)`. That’s a total of 106,168,320 values! If the `dtype` of the tensor was `float32`, then each value would be stored in 32 bits, so the tensor would represent 425 MB. Heavy! Videos you encounter in real life are much lighter because they aren’t stored in `float32`, and they’re typically compressed by a large factor (such as the MPEG format).
[p052_e09] (heading_section) 2.3
[p052_e10] (heading_section) The gears of neural networks: Tensor operations
[p052_e11] (body) Just like any computer program can be ultimately reduced to a small set of binary operations on binary inputs (`AND`, `OR`, `NOR`, and so on), all transformations learned by deep neural networks can be reduced to a handful of tensor operations (or tensor functions) applied to tensors of numeric data. For instance, it’s possible to add tensors, multiply tensors, and so on.
[p052_e12] (body) In our initial example, we were building our model by stacking `Dense` layers on top of each other. A Keras layer instance looks like this:
[p052_e13] (code)
```
keras.layers.Dense(512, activation="relu")
```
[p052_e14] (body) This layer can be interpreted as a function, which takes as input a matrix and returns another matrix—a new representation for the input tensor. Specifically, the function is as follows (where `W` is a matrix and `b` is a vector, both attributes of the layer):
[p052_e15] (code)
```
output = relu(matmul(input, W) + b)
```

## Figures
- Figure 2.4: A rank-4 image data tensor  | labels inside the figure: Color channels, Height, Width, Samples

## Equation candidates (formulas written inline in the text)
- [p052_e13] keras.layers.Dense(512, activation="relu")
- [p052_e15] output = relu(matmul(input, W) + b)

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

## Previous page (PDF 51) text, for continuity
A dataset of text documents, where we represent each document by the counts of how many times each word appears in it (out of a dictionary of 20,000 common words). Each document can be encoded as a vector of 20,000 values (one count per word in the dictionary), and thus an entire dataset of 500 documents can be stored in a tensor of shape `(500,` `20000)`.

Timeseries data or sequence data

Whenever time matters in your data (or the notion of sequence order), it makes sense to store it in a rank-3 tensor with an explicit time axis. Each sample can be encoded as a sequence of vectors (a rank-2 tensor), and thus a batch of data will be encoded as a rank-3 tensor (see figure 2.3).

Figure 2.3 A rank-3 timeseries data tensor

The time axis is always the second axis (axis of index 1), by convention. Let’s look at a few examples:

A dataset of stock prices—Every minute, we store the current price of the stock, the highest price in the past minute, and the lowest price in the past minute. Thus every minute is encoded as a 3D vector, an entire day of trading is encoded as a matrix of shape `(390,` `3)` (there are 390 minutes in a trading day), and 250 days’ worth of data can be stored in a rank-3 tensor of shape `(250,` `390,` `3)`. Here, each sample would be one day’s worth of data.

A dataset of tweets, where we encode each tweet as a sequence of 280 characters out of an alphabet of 128 unique characters—In this setting, each character can be encoded as a binary vector of size 128 (an all-zeros vector except for a 1 entry at the index corresponding to the character). Then each tweet can be encoded as a rank-2 tensor of shape `(280,` `128)`, and a dataset of 1 million tweets can be stored in a tensor of shape `(1000000,` `280,` `128)`.

Image data

Images typically have three  …

## Next page (PDF 53) text, for continuity
Let’s unpack this. We have three tensor operations here:

A tensor product (`matmul`) between the input tensor and a tensor named `W`.

An addition (`+`) between the resulting matrix and a vector `b`.

A `relu` operation: `relu(x)` is `max(x,` `0)`. `"relu"` stands for “REctified Linear Unit.”

NOTE Although this section deals entirely with linear algebra expressions, you won’t find any mathematical notation in this book. I’ve found that mathematical concepts can be more readily mastered by programmers with no mathematical background if they’re expressed as short Python snippets instead of mathematical equations. So we’ll use NumPy code throughout.

2.3.1

Element-wise operations

The `relu` operation and addition are element-wise operations: operations that are applied independently to each entry in the tensors being considered. This means these operations are highly amenable to massively parallel implementations (vectorized implementations, a term that comes from the vector processor supercomputer architecture from the 1970–1990 period). If you want to write a naive Python implementation of an element-wise operation, you use a `for` loop, as in this naive implementation of an element-wise `relu` operation:

You could do the same for addition:

On the same principle, you can do element-wise multiplication, subtraction, and so on.

In practice, when dealing with NumPy arrays, these operations are available as well-optimized built-in NumPy functions, which themselves delegate the heavy lifting to a Basic Linear Algebra Subprograms (BLAS) implementation. BLAS are low-level,

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.47 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.59 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.50 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.82 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.54 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.56 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.45 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.50 (printed 26) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Real-world examples of data tensors
Let’s make data tensors more concrete with a few examples similar to what you’ll encounter later. The data you’ll manipulate will almost always fall into one of the following categories:

Vector data—Rank-2 tensors of shape `(samples,` `features)`, where each sample is a vector of numerical attributes (“features”)

Timeseries data or sequence data—Rank-3 tensors of shape `(samples,` `timesteps, ` `features)`, where each sample is a sequence (of length `timesteps`) of feature vectors

Images—Rank-4 tensors of shape `(samples,` `height,` `width,` `channels)`, where each sample is a 2D grid of pixels, and each pixel is represented by a vector of values (“channels”)

Video—Rank-5 tensors of shape `(samples,` `frames,` `height,` `width,` `channels)`, where each sample is a sequence (of length `frames`) of images

### PDF p.117 (printed 93) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Different types of layers are appropriate for different tensor formats and different types of data processing. For instance, simple vector data, stored in 2D tensors of shape `(samples,` `features)`, is often processed by densely connected layers, also called fully connected or dense layers (the `Dense` class in Keras). Sequence data, stored in 3D tensors of shape `(samples,` `timesteps,` `features)`, is typically processed by recurrent layers, such as an LSTM layer, or 1D convolution layers (`Conv1D`). Image data, stored in rank-4 tensors, is usually processed by 2D convolution layers (`Conv2D`). You can think of layers as the LEGO bricks of deep learning, a metaphor that is made explicit by Keras. Building deep learning models in Keras is done by clipping together compatible layers to form useful data transformation pipelines.

### PDF p.9 (printed ix) — contents ›  › 
2

The mathematical building blocks of neural networks 16 2.1

A first look at a neural network 17 2.2 Data representations for neural networks 21

Scalars (rank-0 tensors) 22 ■ Vectors (rank-1 tensors) 22 Matrices (rank-2 tensors) 22 ■ Rank-3 tensors and higher-rank tensors 23 ■ Key attributes 23 ■ Manipulating tensors in NumPy 25 ■ The notion of data batches 25 Real-world examples of data tensors 26

2.3

The gears of neural networks: Tensor operations 28

Element-wise operations 29 ■ Broadcasting 30 Tensor product 32 ■ Tensor reshaping 34 ■ Geometric interpretation of tensor operations 35 ■ A geometric interpretation of deep learning 38

2.4

The engine of neural networks: Gradient-based optimization 39

What’s a derivative? 41 ■ Derivative of a tensor operation: The gradient 42 ■ Stochastic gradient descent 43 ■ Chaining derivatives: The Backpropagation algorithm 46

2.5

Looking back at our first example 51

Reimplementing our first example from scratch 53 Running one training step 55 ■ The full training loop 57 Evaluating the model 58

3

Introduction to TensorFlow, PyTorch, JAX, and Keras 60 3.1

A brief history of deep learning frameworks 61 3.2 How these frameworks relate 

### PDF p.47 (printed 23) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Rank-3 tensors and higher-rank tensors
By packing rank-3 tensors in an array, you can create a rank-4 tensor, and so on. In deep learning, you’ll generally manipulate tensors with ranks 0 to 4, although you may go up to 5 if you process video data.

### PDF p.82 (printed 58) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Summary
Tensors form the foundation of modern machine learning systems. They come in various flavors of `dtype`, `rank`, and `shape`.

You can manipulate numerical tensors via tensor operations (such as addition, tensor product, or element-wise multiplication), which can be interpreted as encoding geometric transformations. In general, everything in deep learning is amenable to a geometric interpretation.

Deep learning models consist of chains of simple tensor operations, parameterized by weights, which are themselves tensors. The weights of a model are where its “knowledge” is stored.

Learning means finding a set of values for the model’s weights that minimizes a loss function for a given set of training data samples and their corresponding targets.

Learning happens by drawing random batches of data samples and their targets and computing the gradient of the model parameters with respect to the loss on the batch. The model parameters are then moved a bit (the magnitude of the

### PDF p.47 (printed 23) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Key attributes
A tensor is defined by three key attributes:

Number of axes (rank)—For instance, a rank-3 tensor has three axes, and a matrix has two axes. This is also called the tensor’s `ndim` in Python libraries such as NumPy, JAX, TensorFlow, and PyTorch.

Shape—This is a tuple of integers that describes how many dimensions the tensor has along each axis. For instance, the previous matrix example has shape `(3,` `5)`, and the rank-3 tensor example has shape `(3,` `3,` `5)`. A vector has a shape with a single element, such as `(5,)`, whereas a scalar has an empty shape, `()`.

Data type (usually called `dtype` in Python libraries)—This is the type of the data contained in the tensor; for instance, a tensor’s type could be `float16`, `float32`, `float64`, `uint8`, `bool`, and so on. In TensorFlow, you are also likely to come across `string` tensors.

To make this more concrete, let’s look back at the data we processed in the MNIST example. First, we load the MNIST dataset:

### PDF p.45 (printed 21) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Data representations for neural networks
In the previous example, we started from data stored in multidimensional NumPy arrays, also called tensors. In general, all current machine learning systems use tensors as their basic data structure. Tensors are fundamental to the field—so fundamental that the TensorFlow framework was named after them. So what’s a tensor?

At its core, a tensor is a container for data—usually numerical data. So it’s a container for numbers. You may already be familiar with matrices, which are rank-2 tensors: tensors are a generalization of matrices to an arbitrary number of dimensions (note that in the context of tensors, a dimension is often called an axis). Going over the details of tensors might seem a bit abstract at first. But it’s well worth it—manipulating tensors will be the bread and butter of any machine learning code you ever write.

### PDF p.257 (printed 233) — 8 Image classification › 8.1 Introduction to convnets › Image data formats in deep learning frameworks
Some deep learning libraries flip the location of channels in image tensors to the first rank (notably much of the PyTorch ecosystem). Rather than passing images with

shape `(height, width, channels)` you would pass `(channels, height, width)`.

This is purely a matter of convention, and in Keras is configurable. You can call `keras
`

`.config.set_image_data_format("channels_first")` to change Keras’ default, or

pass a `data_format` argument to any conv or pooling layer. In general, you can leave the default as is unless you have a specific need for `"channels_first"`.
