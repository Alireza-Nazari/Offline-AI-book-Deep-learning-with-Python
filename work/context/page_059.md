# Generation bundle for page_059  (PDF page 59, printed page 35)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.3 The gears of neural networks: Tensor operations
Subsection: 2.3.5 Geometric interpretation of tensor operations
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_059.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_059.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p059_e03] (body) Reshaping a tensor means rearranging its rows and columns to match a target shape. Naturally, the reshaped tensor has the same total number of coefficients as the initial tensor. Reshaping is best understood via simple examples:
[p059_e04] (code)
```
>>> x = np.array([[0., 1.],
...               [2., 3.],
...               [4., 5.]])
>>> x.shape
(3, 2)
>>> x = x.reshape((6, 1))
>>> x
array([[ 0.],
       [ 1.],
       [ 2.],
       [ 3.],
       [ 4.],
       [ 5.]])
>>> x = x.reshape((2, 3))
>>> x
array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.]])
```
[p059_e05] (body) A special case of reshaping that’s commonly encountered is transposition. Transposing a matrix means exchanging its rows and its columns, so that `x[i, :]` becomes `x[:, i]:`
[p059_e06] (code)
```
>>> x = np.zeros((300, 20))
>>> x = np.transpose(x)
>>> x.shape
(20, 300)
```
[p059_e07] (heading_subsection) 2.3.5
[p059_e08] (heading_subsection) Geometric interpretation of tensor operations
[p059_e09] (body) Because the contents of the tensors manipulated by tensor operations can be interpreted as coordinates of points in some geometric space, all tensor operations have a geometric interpretation. For instance, let’s consider addition. We’ll start with the following vector:
[p059_e10] (code)
```
A = [0.5, 1]
```
[p059_e11] (body) It’s a point in a 2D space (see figure 2.6). It’s common to picture a vector as an arrow linking the origin to the point, as shown in figure 2.7.
[p059_e12] (body) Let’s consider a new point, `B` `=` `[1,` `0.25]`, which we’ll add to the previous one. This is done geometrically by chaining together the vector arrows, with the resulting location being the vector representing the sum of the previous two vectors (see figure 2.8). As you can see, adding a vector `B` to a vector `A` represents the action of copying point A in

## Code annotations printed next to listings (callouts, in order)
- Creates an all-zeros
- matrix of shape (300, 20)

## Equation candidates (formulas written inline in the text)
- [p059_e10] A = [0.5, 1]

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

## Previous page (PDF 58) text, for continuity
To understand vector product shape compatibility, it helps to visualize the input and output tensors by aligning them as shown in figure 2.5.

`x`, `y`, and `z` are pictured as rectangles (literal boxes of coefficients). Because the rows of `x` and the columns of `y` must have the same size, it follows that the width of `x` must match the height of `y`. If you go on to develop new machine learning algorithms, you’ll likely be drawing such diagrams often. More generally, you can take the product between higher-dimensional tensors, following the same rules for shape compatibility as outlined earlier for the 2D case:

And so on.

Figure 2.5 Matrix product box diagram

2.3.4

Tensor reshaping

A third type of tensor operation that’s essential to understand is tensor reshaping. Although it wasn’t used in the `Dense` layers in our first neural network example, we used it when we preprocessed the digits data before feeding it into our model:

## Next page (PDF 60) text, for continuity
Figure 2.7 A point in a 2D Figure 2.6 A point in a 2D space space pictured as an arrow

new location, whose distance and direction from the original point A is determined by the vector `B`. If you apply the same vector addition to a group of points in the plane (an “object”), you would be creating a copy of the entire object in a new location (see figure 2.9). Tensor addition thus represents the action of translating an object (moving the object without distorting it) by a certain amount in a certain direction.

In general, elementary geometric operations, such as translation, rotation, scaling, skewing,

Figure 2.8 Geometric interpretation of

and so on, can be expressed as tensor operations.

the sum of two vectors

Here are a few examples:

Translation—As you just saw, adding a vector to a point will move this point by a fixed amount in a fixed direction. Applied to a set of points (such as a 2D object), this is called a “translation” (see figure 2.9).

Figure 2.9 2D translation as a vector addition

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.52 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.50 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.46 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.55 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.54 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.47 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.57 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.49 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
This layer can be interpreted as a function, which takes as input a matrix and returns another matrix—a new representation for the input tensor. Specifically, the function is as follows (where `W` is a matrix and `b` is a vector, both attributes of the layer):

### PDF p.82 (printed 58) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Summary
Tensors form the foundation of modern machine learning systems. They come in various flavors of `dtype`, `rank`, and `shape`.

You can manipulate numerical tensors via tensor operations (such as addition, tensor product, or element-wise multiplication), which can be interpreted as encoding geometric transformations. In general, everything in deep learning is amenable to a geometric interpretation.

Deep learning models consist of chains of simple tensor operations, parameterized by weights, which are themselves tensors. The weights of a model are where its “knowledge” is stored.

Learning means finding a set of values for the model’s weights that minimizes a loss function for a given set of training data samples and their corresponding targets.

Learning happens by drawing random batches of data samples and their targets and computing the gradient of the model parameters with respect to the loss on the batch. The model parameters are then moved a bit (the magnitude of the

### PDF p.62 (printed 38) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › A geometric interpretation of deep learning
You just learned that neural networks consist entirely of chains of tensor operations and that all of these tensor operations are just simple geometric transformations of the input data. It follows that you can interpret a neural network as a very complex geometric transformation in a high-dimensional space, implemented via a series of simple steps.

In 3D, the following mental image may prove useful. Imagine two sheets of colored paper: one red and one blue. Put one on top of the other. Now crumple them together into a small ball. That crumpled paper ball is your input data, and each sheet of paper is a class of data in a classification problem. What a neural network is meant to do is figure

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
Just like any computer program can be ultimately reduced to a small set of binary operations on binary inputs (`AND`, `OR`, `NOR`, and so on), all transformations learned by deep neural networks can be reduced to a handful of tensor operations (or tensor functions) applied to tensors of numeric data. For instance, it’s possible to add tensors, multiply tensors, and so on.

In our initial example, we were building our model by stacking `Dense` layers on top of each other. A Keras layer instance looks like this:

### PDF p.53 (printed 29) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Let’s unpack this. We have three tensor operations here:

A tensor product (`matmul`) between the input tensor and a tensor named `W`.

An addition (`+`) between the resulting matrix and a vector `b`.

A `relu` operation: `relu(x)` is `max(x,` `0)`. `"relu"` stands for “REctified Linear Unit.”

NOTE Although this section deals entirely with linear algebra expressions, you won’t find any mathematical notation in this book. I’ve found that mathematical concepts can be more readily mastered by programmers with no mathematical background if they’re expressed as short Python snippets instead of mathematical equations. So we’ll use NumPy code throughout.

### PDF p.46 (printed 22) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Matrices (rank-2 tensors)
An array of vectors is a matrix (or rank-2 tensor or 2D tensor). A matrix has two axes (often referred to as rows and columns). You can visually interpret a matrix as a rectangular grid of numbers. This is a NumPy matrix:

### PDF p.54 (printed 30) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › Broadcasting
Our earlier naive implementation of `naive_add` only supports the addition of rank-2 tensors with identical shapes. But in the `Dense` layer introduced earlier, we added a rank-2 tensor with a vector. What happens with addition when the shapes of the two tensors being added differ?

When possible, and if there’s no ambiguity, the smaller tensor will be broadcast to match the shape of the larger tensor. Broadcasting consists of two steps:

Axes (called broadcast axes) are added to the smaller tensor to match the `ndim` of the larger tensor.

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
