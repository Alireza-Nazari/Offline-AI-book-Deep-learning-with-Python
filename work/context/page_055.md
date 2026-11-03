# Generation bundle for page_055  (PDF page 55, printed page 31)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.3 The gears of neural networks: Tensor operations
Subsection: 2.3.2 Broadcasting
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_055.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_055.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p055_e03] (bullet) The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor.
[p055_e04] (body) Let’s look at a concrete example. Consider `X` with shape `(32,` `10)` and `y` with shape `(10,)`:
[p055_e05] (code)
```
import numpy as np
X = np.random.random((32, 10))
y = np.random.random((10,))
```
[p055_e06] (body) First, we add an empty first axis to `y`, whose shape becomes `(1, 10)`:
[p055_e07] (code)
```
y = np.expand_dims(y, axis=0)
```
[p055_e08] (body) Then, we repeat `y` 32 times alongside this new axis, so that we end up with a tensor `Y` with shape `(32,` `10)`, where `Y[i,` `:] == y` for `i` in `range(0,` `32)`:
[p055_e09] (code)
```
Y = np.tile(y, (32, 1))
```
[p055_e10] (body) At this point, we can add `X` and `Y` because they have the same shape.
[p055_e11] (body) In terms of implementation, no new rank-2 tensor is created because that would be terribly inefficient. The repetition operation is entirely virtual: it happens at the algorithmic level rather than at the memory level. But thinking of the vector being repeated 32 times alongside a new axis is a helpful mental model. Here’s what a naive implementation would look like:
[p055_e12] (code)
```
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x
```
[p055_e13] (body) With broadcasting, you can generally apply two-tensor element-wise operations if one tensor has shape `(a,` `b,` `…` `n,` `n` `+` `1,` `…` `m)` and the other has shape `(n,` `n` `+` `1,` `…` `m)`. The broadcasting will then automatically happen for axes `a` through `n` `-` `1`.

## Code annotations printed next to listings (callouts, in order)
- X is a random matrix with shape (32, 10).
- y is a random vector with shape (10,).
- The shape of y is now (1, 10).
- Repeat y 32 times along axis 0
- to obtain Y with shape (32, 10).
- x is a rank-2 NumPy tensor.
- y is a NumPy vector.
- Avoids overwriting the input tensor

## Equation candidates (formulas written inline in the text)
- [p055_e07] y = np.expand_dims(y, axis=0)
- [p055_e09] Y = np.tile(y, (32, 1))

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

## Previous page (PDF 54) text, for continuity
highly parallel, efficient tensor-manipulation routines that are typically implemented in Fortran or C.

So, in NumPy, you can do the following element-wise operation, and it will be blazing fast:

Let’s actually time the difference:

This takes 0.02 seconds. Meanwhile, the naive version takes a stunning 2.45 seconds:

Likewise, when running JAX/TensorFlow/PyTorch code on a GPU, element-wise operations are executed via fully vectorized CUDA implementations that can best utilize the highly parallel GPU chip architecture.

2.3.2

Broadcasting

Our earlier naive implementation of `naive_add` only supports the addition of rank-2 tensors with identical shapes. But in the `Dense` layer introduced earlier, we added a rank-2 tensor with a vector. What happens with addition when the shapes of the two tensors being added differ?

When possible, and if there’s no ambiguity, the smaller tensor will be broadcast to match the shape of the larger tensor. Broadcasting consists of two steps:

Axes (called broadcast axes) are added to the smaller tensor to match the `ndim` of the larger tensor.

## Next page (PDF 56) text, for continuity
The following example applies the element-wise `maximum` operation to two tensors of different shapes via broadcasting:

2.3.3

Tensor product

The tensor product, also called dot product or matmul (short for “matrix multiplication”) is one of the most common, most useful tensor operations.

In NumPy, a tensor product is done using the `np.matmul` function, and in Keras, with the `keras.ops.matmul` function. Its shorthand is the `@` operator in Python:

In mathematical notation, you’d note the operation with a dot (•) (hence the name “dot product”):

Mathematically, what does the `matmul` operation do? Let’s start with the product of two vectors `x` and `y`. It’s computed as follows:

You’ll have noticed that the product between two vectors is a scalar and that only vectors with the same number of elements are compatible for this operation.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.53 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.47 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.50 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.57 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.59 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.58 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.52 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.49 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.49 (printed 25) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Manipulating tensors in NumPy
In the previous example, we selected a specific digit alongside the first axis using the syntax `train_images[i]`. Selecting specific elements in a tensor is called tensor slicing. Let’s look at the tensor-slicing operations you can do on NumPy arrays.

The following example selects digits #10 to #100 (#100 isn’t included) and puts them in an array of shape `(90,` `28,` `28)`:

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
This layer can be interpreted as a function, which takes as input a matrix and returns another matrix—a new representation for the input tensor. Specifically, the function is as follows (where `W` is a matrix and `b` is a vector, both attributes of the layer):

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
Just like any computer program can be ultimately reduced to a small set of binary operations on binary inputs (`AND`, `OR`, `NOR`, and so on), all transformations learned by deep neural networks can be reduced to a handful of tensor operations (or tensor functions) applied to tensors of numeric data. For instance, it’s possible to add tensors, multiply tensors, and so on.

In our initial example, we were building our model by stacking `Dense` layers on top of each other. A Keras layer instance looks like this:

### PDF p.47 (printed 23) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Key attributes
A tensor is defined by three key attributes:

Number of axes (rank)—For instance, a rank-3 tensor has three axes, and a matrix has two axes. This is also called the tensor’s `ndim` in Python libraries such as NumPy, JAX, TensorFlow, and PyTorch.

Shape—This is a tuple of integers that describes how many dimensions the tensor has along each axis. For instance, the previous matrix example has shape `(3,` `5)`, and the rank-3 tensor example has shape `(3,` `3,` `5)`. A vector has a shape with a single element, such as `(5,)`, whereas a scalar has an empty shape, `()`.

Data type (usually called `dtype` in Python libraries)—This is the type of the data contained in the tensor; for instance, a tensor’s type could be `float16`, `float32`, `float64`, `uint8`, `bool`, and so on. In TensorFlow, you are also likely to come across `string` tensors.

To make this more concrete, let’s look back at the data we processed in the MNIST example. First, we load the MNIST dataset:

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

### PDF p.59 (printed 35) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Reshaping a tensor means rearranging its rows and columns to match a target shape. Naturally, the reshaped tensor has the same total number of coefficients as the initial tensor. Reshaping is best understood via simple examples:

### PDF p.59 (printed 35) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › Geometric interpretation of tensor operations
It’s a point in a 2D space (see figure 2.6). It’s common to picture a vector as an arrow linking the origin to the point, as shown in figure 2.7.

Let’s consider a new point, `B` `=` `[1,` `0.25]`, which we’ll add to the previous one. This is done geometrically by chaining together the vector arrows, with the resulting location being the vector representing the sum of the previous two vectors (see figure 2.8). As you can see, adding a vector `B` to a vector `A` represents the action of copying point A in

### PDF p.60 (printed 36) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Figure 2.7 A point in a 2D Figure 2.6 A point in a 2D space space pictured as an arrow

new location, whose distance and direction from the original point A is determined by the vector `B`. If you apply the same vector addition to a group of points in the plane (an “object”), you would be creating a copy of the entire object in a new location (see figure 2.9). Tensor addition thus represents the action of translating an object (moving the object without distorting it) by a certain amount in a certain direction.

In general, elementary geometric operations, such as translation, rotation, scaling, skewing,

Figure 2.8 Geometric interpretation of

and so on, can be expressed as tensor operations.

the sum of two vectors

Here are a few examples:

Translation—As you just saw, adding a vector to a point will move this point by a fixed amount in a fixed direction. Applied to a set of points (such as a 2D object), this is called a “translation” (see figure 2.9).

Figure 2.9 2D translation as a vector addition
