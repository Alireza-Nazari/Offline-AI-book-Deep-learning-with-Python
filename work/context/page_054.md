# Generation bundle for page_054  (PDF page 54, printed page 30)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.3 The gears of neural networks: Tensor operations
Subsection: 2.3.2 Broadcasting
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_054.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_054.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p054_e03] (body) highly parallel, efficient tensor-manipulation routines that are typically implemented in Fortran or C.
[p054_e04] (body) So, in NumPy, you can do the following element-wise operation, and it will be blazing fast:
[p054_e05] (code)
```
import numpy as np
z = x + y
z = np.maximum(z, 0.0)
```
[p054_e06] (body) Let’s actually time the difference:
[p054_e07] (code)
```
import time
x = np.random.random((20, 100))
y = np.random.random((20, 100))
t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(z, 0.0)
print("Took: {0:.2f} s".format(time.time() - t0))
```
[p054_e08] (body) This takes 0.02 seconds. Meanwhile, the naive version takes a stunning 2.45 seconds:
[p054_e09] (code)
```
t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("Took: {0:.2f} s".format(time.time() - t0))
```
[p054_e10] (body) Likewise, when running JAX/TensorFlow/PyTorch code on a GPU, element-wise operations are executed via fully vectorized CUDA implementations that can best utilize the highly parallel GPU chip architecture.
[p054_e11] (heading_subsection) 2.3.2
[p054_e12] (heading_subsection) Broadcasting
[p054_e13] (body) Our earlier naive implementation of `naive_add` only supports the addition of rank-2 tensors with identical shapes. But in the `Dense` layer introduced earlier, we added a rank-2 tensor with a vector. What happens with addition when the shapes of the two tensors being added differ?
[p054_e14] (body) When possible, and if there’s no ambiguity, the smaller tensor will be broadcast to match the shape of the larger tensor. Broadcasting consists of two steps:
[p054_e15] (bullet) Axes (called broadcast axes) are added to the smaller tensor to match the `ndim` of the larger tensor.

## Code annotations printed next to listings (callouts, in order)
- Element-wise addition
- Element-wise relu

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

## Previous page (PDF 53) text, for continuity
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

## Next page (PDF 55) text, for continuity
The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor.

Let’s look at a concrete example. Consider `X` with shape `(32,` `10)` and `y` with shape `(10,)`:

First, we add an empty first axis to `y`, whose shape becomes `(1, 10)`:

Then, we repeat `y` 32 times alongside this new axis, so that we end up with a tensor `Y` with shape `(32,` `10)`, where `Y[i,` `:] == y` for `i` in `range(0,` `32)`:

At this point, we can add `X` and `Y` because they have the same shape.

In terms of implementation, no new rank-2 tensor is created because that would be terribly inefficient. The repetition operation is entirely virtual: it happens at the algorithmic level rather than at the memory level. But thinking of the vector being repeated 32 times alongside a new axis is a helpful mental model. Here’s what a naive implementation would look like:

With broadcasting, you can generally apply two-tensor element-wise operations if one tensor has shape `(a,` `b,` `…` `n,` `n` `+` `1,` `…` `m)` and the other has shape `(n,` `n` `+` `1,` `…` `m)`. The broadcasting will then automatically happen for axes `a` through `n` `-` `1`.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.56 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.47 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.59 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.52 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.77 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.46 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.49 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.56 (printed 32) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
The following example applies the element-wise `maximum` operation to two tensors of different shapes via broadcasting:

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

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
Just like any computer program can be ultimately reduced to a small set of binary operations on binary inputs (`AND`, `OR`, `NOR`, and so on), all transformations learned by deep neural networks can be reduced to a handful of tensor operations (or tensor functions) applied to tensors of numeric data. For instance, it’s possible to add tensors, multiply tensors, and so on.

In our initial example, we were building our model by stacking `Dense` layers on top of each other. A Keras layer instance looks like this:

### PDF p.49 (printed 25) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Manipulating tensors in NumPy
In the previous example, we selected a specific digit alongside the first axis using the syntax `train_images[i]`. Selecting specific elements in a tensor is called tensor slicing. Let’s look at the tensor-slicing operations you can do on NumPy arrays.

The following example selects digits #10 to #100 (#100 isn’t included) and puts them in an array of shape `(90,` `28,` `28)`:

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
output = relu(matmul(input, W) + b)

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Tensor operations: Doing math in PyTorch
Here’s a dense layer:

### PDF p.82 (printed 58) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Summary
Tensors form the foundation of modern machine learning systems. They come in various flavors of `dtype`, `rank`, and `shape`.

You can manipulate numerical tensors via tensor operations (such as addition, tensor product, or element-wise multiplication), which can be interpreted as encoding geometric transformations. In general, everything in deep learning is amenable to a geometric interpretation.

Deep learning models consist of chains of simple tensor operations, parameterized by weights, which are themselves tensors. The weights of a model are where its “knowledge” is stored.

Learning means finding a set of values for the model’s weights that minimizes a loss function for a given set of training data samples and their corresponding targets.

Learning happens by drawing random batches of data samples and their targets and computing the gradient of the model parameters with respect to the loss on the batch. The model parameters are then moved a bit (the magnitude of the

### PDF p.109 (printed 85) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Tensor assignment
The second difference between JAX and NumPy is tensor assignment. Like in TensorFlow, JAX arrays are not assignable in place. That’s because any sort of in-place modification would go against JAX’s stateless design. Instead, if you need to update a tensor, you must create a new tensor with the desired value. JAX makes this easy by providing the `at()`/`set()` API. These methods allow you to create a new tensor with an updated element at a specific index. Here’s an example of how you would update the first element of a JAX array to a new value.

Listing 3.31 Modifying values in a JAX array
