# Generation bundle for page_057  (PDF page 57, printed page 33)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.3 The gears of neural networks: Tensor operations
Subsection: 2.3.3 Tensor product
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_057.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_057.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p057_e03] (body) You can also take the product between a matrix `x` and a vector `y`, which returns a vector where the coefficients are the products between `y` and the rows of `x`. You implement it as follows:
[p057_e04] (code)
```
def naive_matrix_vector_product(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z
```
[p057_e05] (body) You could also reuse the code we wrote previously, which highlights the relationship between a matrix-vector product and a vector product:
[p057_e06] (code)
```
def naive_matrix_vector_product(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_product(x[i, :], y)
    return z
```
[p057_e07] (body) Note that as soon as one of the two tensors has an `ndim` greater than 1, `matmul` is no longer symmetric, which is to say that `matmul(x,` `y)` isn’t the same as `matmul(y, x)`.
[p057_e08] (body) Of course, a tensor product generalizes to tensors with an arbitrary number of axes. The most common applications may be the product between two matrices. You can take the product of two matrices `x` and `y` (`matmul(x,` `y)`) if and only if `x.shape[1] == ` `y.shape[0]`. The result is a matrix with shape `(x.shape[0],` `y.shape[1])`, where the coefficients are the vector products between the rows of `x` and the columns of `y`. Here’s the naive implementation:
[p057_e09] (code)
```
def naive_matrix_product(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
```

## Code annotations printed next to listings (callouts, in order)
- x is a NumPy matrix.
- y is a NumPy vector.
- The 1st dimension of x must
- equal the 0th dimension of y!
- This operation returns a vector
- of 0s with as many rows as x.
- x and y are NumPy matrices.
- The 1st dimension of x must
- equal the 0th dimension of y!
- This operation returns a matrix
- of 0s with a specific shape.
- Iterates over the rows of x …
- … and over the columns of y.

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

## Previous page (PDF 56) text, for continuity
The following example applies the element-wise `maximum` operation to two tensors of different shapes via broadcasting:

2.3.3

Tensor product

The tensor product, also called dot product or matmul (short for “matrix multiplication”) is one of the most common, most useful tensor operations.

In NumPy, a tensor product is done using the `np.matmul` function, and in Keras, with the `keras.ops.matmul` function. Its shorthand is the `@` operator in Python:

In mathematical notation, you’d note the operation with a dot (•) (hence the name “dot product”):

Mathematically, what does the `matmul` operation do? Let’s start with the product of two vectors `x` and `y`. It’s computed as follows:

You’ll have noticed that the product between two vectors is a scalar and that only vectors with the same number of elements are compatible for this operation.

## Next page (PDF 58) text, for continuity
To understand vector product shape compatibility, it helps to visualize the input and output tensors by aligning them as shown in figure 2.5.

`x`, `y`, and `z` are pictured as rectangles (literal boxes of coefficients). Because the rows of `x` and the columns of `y` must have the same size, it follows that the width of `x` must match the height of `y`. If you go on to develop new machine learning algorithms, you’ll likely be drawing such diagrams often. More generally, you can take the product between higher-dimensional tensors, following the same rules for shape compatibility as outlined earlier for the 2D case:

And so on.

Figure 2.5 Matrix product box diagram

2.3.4

Tensor reshaping

A third type of tensor operation that’s essential to understand is tensor reshaping. Although it wasn’t used in the `Dense` layers in our first neural network example, we used it when we preprocessed the digits data before feeding it into our model:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.55 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.59 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.53 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.54 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.46 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.47 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.50 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.78 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.53 (printed 29) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Let’s unpack this. We have three tensor operations here:

A tensor product (`matmul`) between the input tensor and a tensor named `W`.

An addition (`+`) between the resulting matrix and a vector `b`.

A `relu` operation: `relu(x)` is `max(x,` `0)`. `"relu"` stands for “REctified Linear Unit.”

NOTE Although this section deals entirely with linear algebra expressions, you won’t find any mathematical notation in this book. I’ve found that mathematical concepts can be more readily mastered by programmers with no mathematical background if they’re expressed as short Python snippets instead of mathematical equations. So we’ll use NumPy code throughout.

### PDF p.61 (printed 37) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Rotation—A counterclockwise rotation of a 2D vector by an angle theta (see figure 2.10) can be achieved via a product with a 2 × 2 matrix `R` `=` `[[cos(theta), ` `-sin(theta)], [sin(theta), cos(theta)]].`

Figure 2.10 2D rotation (counterclockwise) as a matrix product

Scaling—A vertical and horizontal scaling of the image (see figure 2.11) can be achieved via a product with a 2 × 2 matrix `S = `(note that such a matrix is called a “diagonal matrix” because it only has non-zero coefficients in its “diagonal,” going from the top left to the bottom right).

Figure 2.11 2D scaling as a matrix product

Linear transform—A product with an arbitrary matrix implements a linear transform. Note that scaling and rotation, seen previously, are, by definition, linear transforms.

Affine transform—An affine transform (see figure 2.12) is the combination of a linear transform (achieved via a matrix product) and a translation (achieved via a vector addition). As you have probably recognized, that’s exactly the `y = W @ x ` `+ b` computation implemented by the `Dense` layer! A `Dense` layer without an activation function is an affine layer.

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
This layer can be interpreted as a function, which takes as input a matrix and returns another matrix—a new representation for the input tensor. Specifically, the function is as follows (where `W` is a matrix and `b` is a vector, both attributes of the layer):

### PDF p.46 (printed 22) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Matrices (rank-2 tensors)
An array of vectors is a matrix (or rank-2 tensor or 2D tensor). A matrix has two axes (often referred to as rows and columns). You can visually interpret a matrix as a rectangular grid of numbers. This is a NumPy matrix:

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
output = relu(matmul(input, W) + b)

### PDF p.47 (printed 23) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Key attributes
A tensor is defined by three key attributes:

Number of axes (rank)—For instance, a rank-3 tensor has three axes, and a matrix has two axes. This is also called the tensor’s `ndim` in Python libraries such as NumPy, JAX, TensorFlow, and PyTorch.

Shape—This is a tuple of integers that describes how many dimensions the tensor has along each axis. For instance, the previous matrix example has shape `(3,` `5)`, and the rank-3 tensor example has shape `(3,` `3,` `5)`. A vector has a shape with a single element, such as `(5,)`, whereas a scalar has an empty shape, `()`.

Data type (usually called `dtype` in Python libraries)—This is the type of the data contained in the tensor; for instance, a tensor’s type could be `float16`, `float32`, `float64`, `uint8`, `bool`, and so on. In TensorFlow, you are also likely to come across `string` tensors.

To make this more concrete, let’s look back at the data we processed in the MNIST example. First, we load the MNIST dataset:

### PDF p.59 (printed 35) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › Geometric interpretation of tensor operations
It’s a point in a 2D space (see figure 2.6). It’s common to picture a vector as an arrow linking the origin to the point, as shown in figure 2.7.

Let’s consider a new point, `B` `=` `[1,` `0.25]`, which we’ll add to the previous one. This is done geometrically by chaining together the vector arrows, with the resulting location being the vector representing the sum of the previous two vectors (see figure 2.8). As you can see, adding a vector `B` to a vector `A` represents the action of copying point A in

### PDF p.46 (printed 22) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Vectors (rank-1 tensors)
This vector has five entries and so is called a 5-dimensional vector. Don’t confuse a 5D vector with a 5D tensor! A 5D vector has only one axis and has five dimensions along its axis, whereas a 5D tensor has five axes (and may have any number of dimensions along each axis). Dimensionality can denote either the number of entries along a specific axis (as in the case of our 5D vector) or the number of axes in a tensor (such as a 5D tensor), which can be confusing at times. In the latter case, it’s technically more correct to talk about a tensor of rank 5 (the rank of a tensor being the number of axes), but the ambiguous notation 5D tensor is common regardless.
