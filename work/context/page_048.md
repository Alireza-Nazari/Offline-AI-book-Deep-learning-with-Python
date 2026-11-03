# Generation bundle for page_048  (PDF page 48, printed page 24)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.2 Data representations for neural networks
Subsection: 2.2.5 Key attributes
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_048.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_048.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p048_e03] (code)
```
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```
[p048_e04] (body) Next, we display the number of axes of the tensor `train_images`, the `ndim` attribute:
[p048_e05] (code)
```
>>> train_images.ndim
3
```
[p048_e06] (body) Here’s its shape:
[p048_e07] (code)
```
>>> train_images.shape
(60000, 28, 28)
```
[p048_e08] (body) And this is its data type, the `dtype` attribute:
[p048_e09] (code)
```
>>> train_images.dtype
uint8
```
[p048_e10] (body) So what we have here is a rank-3 tensor of 8-bit integers. More precisely, it’s an array of 60,000 matrices of 28 × 28 integers. Each such matrix is a grayscale image, with coefficients between 0 and 255.
[p048_e11] (body) Let’s display the fourth digit in this rank-3 tensor, using the library Matplotlib (part of the standard scientific Python suite); see figure 2.2.
[p048_e12] (listing_caption) Listing 2.8 Displaying the fourth digit
[p048_e13] (code)
```
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
```
[p048_e14] (body) Naturally, the corresponding label is just the
[p048_e15] (figure_caption) Figure 2.2 The fourth sample in our dataset
[p048_e16] (body) integer 9:
[p048_e17] (code)
```
>>> train_labels[4]
9
```

## Listings
- Listing 2.8: Displaying the fourth digit (code elements p048_e13)

## Figures
- Figure 2.2: The fourth sample in our dataset  | labels inside the figure: (none - image)

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

## Previous page (PDF 47) text, for continuity
The entries from the first axis are called the rows, and the entries from the second axis are called the columns. In the previous example, `[5,` `78,` `2,` `34,` `0]` is the first row of `x`, and `[5,` `6,` `7]` is the first column.

2.2.4

Rank-3 tensors and higher-rank tensors

If you pack such matrices in a new array, you obtain a rank-3 tensor (or 3D tensor), which you can visually interpret as a cube of numbers. The following is a NumPy rank-3 tensor:

By packing rank-3 tensors in an array, you can create a rank-4 tensor, and so on. In deep learning, you’ll generally manipulate tensors with ranks 0 to 4, although you may go up to 5 if you process video data.

2.2.5

Key attributes

A tensor is defined by three key attributes:

Number of axes (rank)—For instance, a rank-3 tensor has three axes, and a matrix has two axes. This is also called the tensor’s `ndim` in Python libraries such as NumPy, JAX, TensorFlow, and PyTorch.

Shape—This is a tuple of integers that describes how many dimensions the tensor has along each axis. For instance, the previous matrix example has shape `(3,` `5)`, and the rank-3 tensor example has shape `(3,` `3,` `5)`. A vector has a shape with a single element, such as `(5,)`, whereas a scalar has an empty shape, `()`.

Data type (usually called `dtype` in Python libraries)—This is the type of the data contained in the tensor; for instance, a tensor’s type could be `float16`, `float32`, `float64`, `uint8`, `bool`, and so on. In TensorFlow, you are also likely to come across `string` tensors.

To make this more concrete, let’s look back at the data we processed in the MNIST example. First, we load the MNIST dataset:

## Next page (PDF 49) text, for continuity
2.2.6

Manipulating tensors in NumPy

In the previous example, we selected a specific digit alongside the first axis using the syntax `train_images[i]`. Selecting specific elements in a tensor is called tensor slicing. Let’s look at the tensor-slicing operations you can do on NumPy arrays.

The following example selects digits #10 to #100 (#100 isn’t included) and puts them in an array of shape `(90,` `28,` `28)`:

It’s equivalent to this more detailed notation, which specifies a start index and stop index for the slice along each tensor axis. Note that `:` is equivalent to selecting the entire axis:

In general, you may select slices between any two indices along each tensor axis. For instance, to select 14 × 14 pixels in the bottom-right corner of all images, you would do this:

It’s also possible to use negative indices. Much like negative indices in Python lists, they indicate a position relative to the end of the current axis. To crop the images to patches of 14 × 14 pixels centered in the middle, do this:

2.2.7

The notion of data batches

In general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll come across in deep learning will be the samples axis. In the MNIST example, “samples” are images of digits.

In addition, deep learning models don’t process an entire dataset at once; rather, they break the data into small “batches,” or groups of samples with a fixed size. Concretely, here’s one batch of our MNIST digits, with a batch size of 128:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.42 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.81 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.44 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.75 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.46 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.58 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.50 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.82 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.51 (printed 27) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Timeseries data or sequence data
Whenever time matters in your data (or the notion of sequence order), it makes sense to store it in a rank-3 tensor with an explicit time axis. Each sample can be encoded as a sequence of vectors (a rank-2 tensor), and thus a batch of data will be encoded as a rank-3 tensor (see figure 2.3).

Figure 2.3 A rank-3 timeseries data tensor

The time axis is always the second axis (axis of index 1), by convention. Let’s look at a few examples:

A dataset of stock prices—Every minute, we store the current price of the stock, the highest price in the past minute, and the lowest price in the past minute. Thus every minute is encoded as a 3D vector, an entire day of trading is encoded as a matrix of shape `(390,` `3)` (there are 390 minutes in a trading day), and 250 days’ worth of data can be stored in a rank-3 tensor of shape `(250,` `390,` `3)`. Here, each sample would be one day’s worth of data.

A dataset of tweets, where we encode each tweet as a sequence of 280 characters out of an alphabet of 128 unique characters—In this setting, each character can be encoded as a binary vector of size 128 (an all-zeros vector except for a 1 entry at the index corresponding to the character). Then e

### PDF p.46 (printed 22) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Scalars (rank-0 tensors)
A tensor that contains only one number is called a scalar (or scalar tensor, rank-0 tensor, or 0D tensor). In NumPy, a `float32` or `float64` number is a scalar tensor (or scalar array). You can display the number of axes of a NumPy tensor via the `ndim` attribute; a scalar tensor has 0 axes (`ndim` `==` `0`). The number of axes of a tensor is also called its rank. Here’s a NumPy scalar:

### PDF p.51 (printed 27) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Image data
Images typically have three dimensions: height, width, and color depth. Although grayscale images (like our MNIST digits) have only a single color channel and could thus be stored in rank-2 tensors, by convention image tensors are always rank-3, with a one-dimensional color channel for grayscale images. A batch of 128 grayscale images of size 256 × 256 could thus be stored in a tensor of shape `(128,` `256,` `256, 1)`, and a batch of 128 color images could be stored in a tensor of shape `(128,` `256,` `256,` `3)` (see figure 2.4).

There are two conventions for the shapes of image tensors: the channels-last convention (which is standard in JAX and TensorFlow, as well as most other deep learning tools out there) and the channels-first convention (which is standard in PyTorch). The channels-last convention places the color-depth axis at the end: `(samples,` `height,` `width,` `color_depth)`. Meanwhile, the channels-first convention places the color depth axis right after the batch axis: `(samples,` `color_depth,` `height,` `width)`.

### PDF p.45 (printed 21) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Data representations for neural networks
In the previous example, we started from data stored in multidimensional NumPy arrays, also called tensors. In general, all current machine learning systems use tensors as their basic data structure. Tensors are fundamental to the field—so fundamental that the TensorFlow framework was named after them. So what’s a tensor?

At its core, a tensor is a container for data—usually numerical data. So it’s a container for numbers. You may already be familiar with matrices, which are rank-2 tensors: tensors are a generalization of matrices to an arbitrary number of dimensions (note that in the context of tensors, a dimension is often called an axis). Going over the details of tensors might seem a bit abstract at first. But it’s well worth it—manipulating tensors will be the bread and butter of any machine learning code you ever write.

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › Video data
Video data is one of the few types of real-world data for which you’ll need rank-5 tensors. A video can be understood as a sequence of frames, each frame being a color image. Because each frame can be stored in a rank-3 tensor `(height,` `width,`

Figure 2.4 A rank-4 image data tensor

`color_depth)`, a sequence of frames can be stored in a rank-4 tensor `(frames,` `height,` `width, color_depth)`, and thus a batch of different videos can be stored in a rank-5 tensor of shape `(samples,` `frames,` `height,` `width,` `color_depth)`.

For instance, a 60-second, 144 × 256 YouTube video clip sampled at 4 frames per second would have 240 frames. A batch of four such video clips would be stored in a tensor of shape `(4,` `240,` `144,` `256,` `3)`. That’s a total of 106,168,320 values! If the `dtype` of the tensor was `float32`, then each value would be stored in 32 bits, so the tensor would represent 425 MB. Heavy! Videos you encounter in real life are much lighter because they aren’t stored in `float32`, and they’re typically compressed by a large factor (such as the MPEG format).

### PDF p.46 (printed 22) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Vectors (rank-1 tensors)
This vector has five entries and so is called a 5-dimensional vector. Don’t confuse a 5D vector with a 5D tensor! A 5D vector has only one axis and has five dimensions along its axis, whereas a 5D tensor has five axes (and may have any number of dimensions along each axis). Dimensionality can denote either the number of entries along a specific axis (as in the case of our 5D vector) or the number of axes in a tensor (such as a 5D tensor), which can be confusing at times. In the latter case, it’s technically more correct to talk about a tensor of rank 5 (the rank of a tensor being the number of axes), but the ambiguous notation 5D tensor is common regardless.

### PDF p.50 (printed 26) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Vector data
Vector data is one of the most common cases. In such a dataset, each single data point can be encoded as a vector, and thus a batch of data will be encoded as a rank-2 tensor (that is, an array of vectors), where the first axis is the samples axis and the second axis is the features axis.

Let’s take a look at two examples:

An actuarial dataset of people, where we consider each person’s age, gender, and income. Each person can be characterized as a vector of three values, and thus an entire dataset of 100,000 people can be stored in a rank-2 tensor of shape `(100000,` `3)`.

### PDF p.50 (printed 26) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Real-world examples of data tensors
Let’s make data tensors more concrete with a few examples similar to what you’ll encounter later. The data you’ll manipulate will almost always fall into one of the following categories:

Vector data—Rank-2 tensors of shape `(samples,` `features)`, where each sample is a vector of numerical attributes (“features”)

Timeseries data or sequence data—Rank-3 tensors of shape `(samples,` `timesteps, ` `features)`, where each sample is a sequence (of length `timesteps`) of feature vectors

Images—Rank-4 tensors of shape `(samples,` `height,` `width,` `channels)`, where each sample is a 2D grid of pixels, and each pixel is represented by a vector of values (“channels”)

Video—Rank-5 tensors of shape `(samples,` `frames,` `height,` `width,` `channels)`, where each sample is a sequence (of length `frames`) of images
