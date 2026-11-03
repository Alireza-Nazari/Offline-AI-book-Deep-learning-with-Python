# Generation bundle for page_056  (PDF page 56, printed page 32)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.3 The gears of neural networks: Tensor operations
Subsection: 2.3.3 Tensor product
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_056.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_056.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p056_e03] (body) The following example applies the element-wise `maximum` operation to two tensors of different shapes via broadcasting:
[p056_e04] (code)
```
import numpy as np
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)
```
[p056_e05] (heading_subsection) 2.3.3
[p056_e06] (heading_subsection) Tensor product
[p056_e07] (body) The tensor product, also called dot product or matmul (short for “matrix multiplication”) is one of the most common, most useful tensor operations.
[p056_e08] (body) In NumPy, a tensor product is done using the `np.matmul` function, and in Keras, with the `keras.ops.matmul` function. Its shorthand is the `@` operator in Python:
[p056_e09] (code)
```
x = np.random.random((32,))
y = np.random.random((32,))
z = np.matmul(x, y)
z = x @ y
```
[p056_e10] (body) In mathematical notation, you’d note the operation with a dot (•) (hence the name “dot product”):
[p056_e11] (code)
```
z = x • y
```
[p056_e12] (body) Mathematically, what does the `matmul` operation do? Let’s start with the product of two vectors `x` and `y`. It’s computed as follows:
[p056_e13] (code)
```
def naive_vector_product(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z
```
[p056_e14] (body) You’ll have noticed that the product between two vectors is a scalar and that only vectors with the same number of elements are compatible for this operation.

## Code annotations printed next to listings (callouts, in order)
- x is a random tensor with
- shape (64, 3, 32, 10).
- y is a random tensor with shape (32, 10).
- The output z has shape (64, 3, 32, 10) like x.
- Takes the product between x and y
- This is equivalent.
- x and y are NumPy vectors.

## Equation candidates (formulas written inline in the text)
- [p056_e09] x = np.random.random((32,))
y = np.random.random((32,))
z = np.matmul(x, y)
z = x @ y
- [p056_e11] z = x • y

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

## Previous page (PDF 55) text, for continuity
The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor.

Let’s look at a concrete example. Consider `X` with shape `(32,` `10)` and `y` with shape `(10,)`:

First, we add an empty first axis to `y`, whose shape becomes `(1, 10)`:

Then, we repeat `y` 32 times alongside this new axis, so that we end up with a tensor `Y` with shape `(32,` `10)`, where `Y[i,` `:] == y` for `i` in `range(0,` `32)`:

At this point, we can add `X` and `Y` because they have the same shape.

In terms of implementation, no new rank-2 tensor is created because that would be terribly inefficient. The repetition operation is entirely virtual: it happens at the algorithmic level rather than at the memory level. But thinking of the vector being repeated 32 times alongside a new axis is a helpful mental model. Here’s what a naive implementation would look like:

With broadcasting, you can generally apply two-tensor element-wise operations if one tensor has shape `(a,` `b,` `…` `n,` `n` `+` `1,` `…` `m)` and the other has shape `(n,` `n` `+` `1,` `…` `m)`. The broadcasting will then automatically happen for axes `a` through `n` `-` `1`.

## Next page (PDF 57) text, for continuity
You can also take the product between a matrix `x` and a vector `y`, which returns a vector where the coefficients are the products between `y` and the rows of `x`. You implement it as follows:

You could also reuse the code we wrote previously, which highlights the relationship between a matrix-vector product and a vector product:

Note that as soon as one of the two tensors has an `ndim` greater than 1, `matmul` is no longer symmetric, which is to say that `matmul(x,` `y)` isn’t the same as `matmul(y, x)`.

Of course, a tensor product generalizes to tensors with an arbitrary number of axes. The most common applications may be the product between two matrices. You can take the product of two matrices `x` and `y` (`matmul(x,` `y)`) if and only if `x.shape[1] == ` `y.shape[0]`. The result is a matrix with shape `(x.shape[0],` `y.shape[1])`, where the coefficients are the vector products between the rows of `x` and the columns of `y`. Here’s the naive implementation:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.53 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.54 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.46 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.58 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.47 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.66 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.52 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.49 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.53 (printed 29) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Let’s unpack this. We have three tensor operations here:

A tensor product (`matmul`) between the input tensor and a tensor named `W`.

An addition (`+`) between the resulting matrix and a vector `b`.

A `relu` operation: `relu(x)` is `max(x,` `0)`. `"relu"` stands for “REctified Linear Unit.”

NOTE Although this section deals entirely with linear algebra expressions, you won’t find any mathematical notation in this book. I’ve found that mathematical concepts can be more readily mastered by programmers with no mathematical background if they’re expressed as short Python snippets instead of mathematical equations. So we’ll use NumPy code throughout.

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
output = relu(matmul(input, W) + b)

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
This layer can be interpreted as a function, which takes as input a matrix and returns another matrix—a new representation for the input tensor. Specifically, the function is as follows (where `W` is a matrix and `b` is a vector, both attributes of the layer):

### PDF p.66 (printed 42) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › Derivative of a tensor operation: The gradient
y_pred = matmul(x, W)
loss_value = loss(y_pred, y_true)

### PDF p.54 (printed 30) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
highly parallel, efficient tensor-manipulation routines that are typically implemented in Fortran or C.

So, in NumPy, you can do the following element-wise operation, and it will be blazing fast:

### PDF p.82 (printed 58) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Summary
Tensors form the foundation of modern machine learning systems. They come in various flavors of `dtype`, `rank`, and `shape`.

You can manipulate numerical tensors via tensor operations (such as addition, tensor product, or element-wise multiplication), which can be interpreted as encoding geometric transformations. In general, everything in deep learning is amenable to a geometric interpretation.

Deep learning models consist of chains of simple tensor operations, parameterized by weights, which are themselves tensors. The weights of a model are where its “knowledge” is stored.

Learning means finding a set of values for the model’s weights that minimizes a loss function for a given set of training data samples and their corresponding targets.

Learning happens by drawing random batches of data samples and their targets and computing the gradient of the model parameters with respect to the loss on the batch. The model parameters are then moved a bit (the magnitude of the

### PDF p.49 (printed 25) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Manipulating tensors in NumPy
In the previous example, we selected a specific digit alongside the first axis using the syntax `train_images[i]`. Selecting specific elements in a tensor is called tensor slicing. Let’s look at the tensor-slicing operations you can do on NumPy arrays.

The following example selects digits #10 to #100 (#100 isn’t included) and puts them in an array of shape `(90,` `28,` `28)`:

### PDF p.61 (printed 37) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Rotation—A counterclockwise rotation of a 2D vector by an angle theta (see figure 2.10) can be achieved via a product with a 2 × 2 matrix `R` `=` `[[cos(theta), ` `-sin(theta)], [sin(theta), cos(theta)]].`

Figure 2.10 2D rotation (counterclockwise) as a matrix product

Scaling—A vertical and horizontal scaling of the image (see figure 2.11) can be achieved via a product with a 2 × 2 matrix `S = `(note that such a matrix is called a “diagonal matrix” because it only has non-zero coefficients in its “diagonal,” going from the top left to the bottom right).

Figure 2.11 2D scaling as a matrix product

Linear transform—A product with an arbitrary matrix implements a linear transform. Note that scaling and rotation, seen previously, are, by definition, linear transforms.

Affine transform—An affine transform (see figure 2.12) is the combination of a linear transform (achieved via a matrix product) and a translation (achieved via a vector addition). As you have probably recognized, that’s exactly the `y = W @ x ` `+ b` computation implemented by the `Dense` layer! A `Dense` layer without an activation function is an affine layer.
