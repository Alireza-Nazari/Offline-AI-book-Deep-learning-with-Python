# Generation bundle for page_060  (PDF page 60, printed page 36)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.3 The gears of neural networks: Tensor operations
Subsection: 2.3.5 Geometric interpretation of tensor operations
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_060.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_060.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p060_e03] (figure_caption) Figure 2.7 A point in a 2D Figure 2.6 A point in a 2D space space pictured as an arrow
[p060_e04] (body) new location, whose distance and direction from the original point A is determined by the vector `B`. If you apply the same vector addition to a group of points in the plane (an “object”), you would be creating a copy of the entire object in a new location (see figure 2.9). Tensor addition thus represents the action of translating an object (moving the object without distorting it) by a certain amount in a certain direction.
[p060_e05] (body) In general, elementary geometric operations, such as translation, rotation, scaling, skewing,
[p060_e06] (figure_caption) Figure 2.8 Geometric interpretation of
[p060_e07] (body) and so on, can be expressed as tensor operations.
[p060_e08] (figure_caption) the sum of two vectors
[p060_e09] (body) Here are a few examples:
[p060_e10] (bullet) Translation—As you just saw, adding a vector to a point will move this point by a fixed amount in a fixed direction. Applied to a set of points (such as a 2D object), this is called a “translation” (see figure 2.9).
[p060_e11] (figure_caption) Figure 2.9 2D translation as a vector addition

## Figures
- Figure 2.7: A point in a 2D Figure 2.6 A point in a 2D space space pictured as an arrow  | labels inside the figure: 1, 1, A, [0.5, 1], 1, 1, A, [0.5, 1]
- Figure 2.8: Geometric interpretation of  | labels inside the figure: 1, 1, A, B, A + B
- Figure None: the sum of two vectors  | labels inside the figure: (none - image)
- Figure 2.9: 2D translation as a vector addition  | labels inside the figure: x, y, K, K, vertical_factor, horizontal_factor, horizontal_factor, vertical_factor, +

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

## Previous page (PDF 59) text, for continuity
Reshaping a tensor means rearranging its rows and columns to match a target shape. Naturally, the reshaped tensor has the same total number of coefficients as the initial tensor. Reshaping is best understood via simple examples:

A special case of reshaping that’s commonly encountered is transposition. Transposing a matrix means exchanging its rows and its columns, so that `x[i, :]` becomes `x[:, i]:`

2.3.5

Geometric interpretation of tensor operations

Because the contents of the tensors manipulated by tensor operations can be interpreted as coordinates of points in some geometric space, all tensor operations have a geometric interpretation. For instance, let’s consider addition. We’ll start with the following vector:

It’s a point in a 2D space (see figure 2.6). It’s common to picture a vector as an arrow linking the origin to the point, as shown in figure 2.7.

Let’s consider a new point, `B` `=` `[1,` `0.25]`, which we’ll add to the previous one. This is done geometrically by chaining together the vector arrows, with the resulting location being the vector representing the sum of the previous two vectors (see figure 2.8). As you can see, adding a vector `B` to a vector `A` represents the action of copying point A in

## Next page (PDF 61) text, for continuity
Rotation—A counterclockwise rotation of a 2D vector by an angle theta (see figure 2.10) can be achieved via a product with a 2 × 2 matrix `R` `=` `[[cos(theta), ` `-sin(theta)], [sin(theta), cos(theta)]].`

Figure 2.10 2D rotation (counterclockwise) as a matrix product

Scaling—A vertical and horizontal scaling of the image (see figure 2.11) can be achieved via a product with a 2 × 2 matrix `S = `(note that such a matrix is called a “diagonal matrix” because it only has non-zero coefficients in its “diagonal,” going from the top left to the bottom right).

Figure 2.11 2D scaling as a matrix product

Linear transform—A product with an arbitrary matrix implements a linear transform. Note that scaling and rotation, seen previously, are, by definition, linear transforms.

Affine transform—An affine transform (see figure 2.12) is the combination of a linear transform (achieved via a matrix product) and a translation (achieved via a vector addition). As you have probably recognized, that’s exactly the `y = W @ x ` `+ b` computation implemented by the `Dense` layer! A `Dense` layer without an activation function is an affine layer.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.62 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.53 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.54 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.50 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.46 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.55 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.65 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.66 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
This layer can be interpreted as a function, which takes as input a matrix and returns another matrix—a new representation for the input tensor. Specifically, the function is as follows (where `W` is a matrix and `b` is a vector, both attributes of the layer):

### PDF p.62 (printed 38) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › A geometric interpretation of deep learning
You just learned that neural networks consist entirely of chains of tensor operations and that all of these tensor operations are just simple geometric transformations of the input data. It follows that you can interpret a neural network as a very complex geometric transformation in a high-dimensional space, implemented via a series of simple steps.

In 3D, the following mental image may prove useful. Imagine two sheets of colored paper: one red and one blue. Put one on top of the other. Now crumple them together into a small ball. That crumpled paper ball is your input data, and each sheet of paper is a class of data in a classification problem. What a neural network is meant to do is figure

### PDF p.82 (printed 58) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Summary
Tensors form the foundation of modern machine learning systems. They come in various flavors of `dtype`, `rank`, and `shape`.

You can manipulate numerical tensors via tensor operations (such as addition, tensor product, or element-wise multiplication), which can be interpreted as encoding geometric transformations. In general, everything in deep learning is amenable to a geometric interpretation.

Deep learning models consist of chains of simple tensor operations, parameterized by weights, which are themselves tensors. The weights of a model are where its “knowledge” is stored.

Learning means finding a set of values for the model’s weights that minimizes a loss function for a given set of training data samples and their corresponding targets.

Learning happens by drawing random batches of data samples and their targets and computing the gradient of the model parameters with respect to the loss on the batch. The model parameters are then moved a bit (the magnitude of the

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
Just like any computer program can be ultimately reduced to a small set of binary operations on binary inputs (`AND`, `OR`, `NOR`, and so on), all transformations learned by deep neural networks can be reduced to a handful of tensor operations (or tensor functions) applied to tensors of numeric data. For instance, it’s possible to add tensors, multiply tensors, and so on.

In our initial example, we were building our model by stacking `Dense` layers on top of each other. A Keras layer instance looks like this:

### PDF p.54 (printed 30) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › Broadcasting
Our earlier naive implementation of `naive_add` only supports the addition of rank-2 tensors with identical shapes. But in the `Dense` layer introduced earlier, we added a rank-2 tensor with a vector. What happens with addition when the shapes of the two tensors being added differ?

When possible, and if there’s no ambiguity, the smaller tensor will be broadcast to match the shape of the larger tensor. Broadcasting consists of two steps:

Axes (called broadcast axes) are added to the smaller tensor to match the `ndim` of the larger tensor.

### PDF p.53 (printed 29) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Let’s unpack this. We have three tensor operations here:

A tensor product (`matmul`) between the input tensor and a tensor named `W`.

An addition (`+`) between the resulting matrix and a vector `b`.

A `relu` operation: `relu(x)` is `max(x,` `0)`. `"relu"` stands for “REctified Linear Unit.”

NOTE Although this section deals entirely with linear algebra expressions, you won’t find any mathematical notation in this book. I’ve found that mathematical concepts can be more readily mastered by programmers with no mathematical background if they’re expressed as short Python snippets instead of mathematical equations. So we’ll use NumPy code throughout.

### PDF p.40 (printed 16) — 2 The mathematical building blocks of neural networks ›  › This chapter covers
A first example of a neural network

Tensors and tensor operations

How neural networks learn via backpropagation

and gradient descent

Understanding deep learning requires familiarity with many simple mathematical

concepts: tensors, tensor operations, differentiation, gradient descent, and so on. Our goal

in this chapter will be to build up your intuition about these notions without getting overly technical. In particular, we’ll steer away from mathematical notation, which can introduce unnecessary barriers for those without any mathematics background, and isn’t necessary to explain things well. The most precise, unambiguous description of a mathematical operation is its executable code.

To provide sufficient context for introducing tensors and gradient descent, we’ll begin the chapter with a practical example of a neural network. Then we’ll go over every new concept that’s been introduced, point by point. Keep in mind that these concepts will be essential for you to understand the practical examples that will come in the following chapters!

### PDF p.66 (printed 42) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › Derivative of a tensor operation: The gradient
Now, we’d like to use gradients to figure out how to update `W` to make `loss_value` smaller. How do we do that?

Given fixed inputs `x` and `y_true`, the previous operations can be interpreted as a function mapping values of `W` (the model’s weights) to loss values:
