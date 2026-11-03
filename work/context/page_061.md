# Generation bundle for page_061  (PDF page 61, printed page 37)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.3 The gears of neural networks: Tensor operations
Subsection: 2.3.5 Geometric interpretation of tensor operations
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_061.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_061.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p061_e03] (bullet) Rotation—A counterclockwise rotation of a 2D vector by an angle theta (see figure 2.10) can be achieved via a product with a 2 × 2 matrix `R` `=` `[[cos(theta), ` `-sin(theta)], [sin(theta), cos(theta)]].`
[p061_e04] (figure_caption) Figure 2.10 2D rotation (counterclockwise) as a matrix product
[p061_e05] (bullet) Scaling—A vertical and horizontal scaling of the image (see figure 2.11) can be achieved via a product with a 2 × 2 matrix `S = `(note that such a matrix is called a “diagonal matrix” because it only has non-zero coefficients in its “diagonal,” going from the top left to the bottom right).
[p061_e06] (figure_caption) Figure 2.11 2D scaling as a matrix product
[p061_e07] (bullet) Linear transform—A product with an arbitrary matrix implements a linear transform. Note that scaling and rotation, seen previously, are, by definition, linear transforms.
[p061_e08] (bullet) Affine transform—An affine transform (see figure 2.12) is the combination of a linear transform (achieved via a matrix product) and a translation (achieved via a vector addition). As you have probably recognized, that’s exactly the `y = W @ x ` `+ b` computation implemented by the `Dense` layer! A `Dense` layer without an activation function is an affine layer.

## Figures
- Figure 2.10: 2D rotation (counterclockwise) as a matrix product  | labels inside the figure: cos(theta) –sin(theta), sin(theta), cos(theta), x, y, Theta, K, K
- Figure 2.11: 2D scaling as a matrix product  | labels inside the figure: 1, 0, 0 –0.5, x, y, K, K

## Equation candidates (formulas written inline in the text)
- [p061_e05] Scaling—A vertical and horizontal scaling of the image (see figure 2.11) can be achieved via a product with a 2 × 2 matrix `S = `(note that such a matrix is called a “diagonal matrix” because it only has non-zero coefficients in its “diagonal,” going from the top left to the bottom right).
- [p061_e08] As you have probably recognized, that’s exactly the `y = W @ x ` `+ b` computation implemented by the `Dense` layer!

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

## Previous page (PDF 60) text, for continuity
Figure 2.7 A point in a 2D Figure 2.6 A point in a 2D space space pictured as an arrow

new location, whose distance and direction from the original point A is determined by the vector `B`. If you apply the same vector addition to a group of points in the plane (an “object”), you would be creating a copy of the entire object in a new location (see figure 2.9). Tensor addition thus represents the action of translating an object (moving the object without distorting it) by a certain amount in a certain direction.

In general, elementary geometric operations, such as translation, rotation, scaling, skewing,

Figure 2.8 Geometric interpretation of

and so on, can be expressed as tensor operations.

the sum of two vectors

Here are a few examples:

Translation—As you just saw, adding a vector to a point will move this point by a fixed amount in a fixed direction. Applied to a set of points (such as a 2D object), this is called a “translation” (see figure 2.9).

Figure 2.9 2D translation as a vector addition

## Next page (PDF 62) text, for continuity
Figure 2.12 Affine transform in the plane

`Dense` layer with `relu` activation—An important observation about affine transforms is that if you apply many of them repeatedly, you still end up with an affine transform (so you could just have applied that one affine transform in the first place). Let’s try it with two: `affine2(affine1(x)) = W2 @ (W1 @ x + b1) + b2 = (W2 @ W1) 
` `@ x + (W2 @ b1 + b2)`. That’s an affine transform where the linear part is the matrix `W2 @ W1` and the translation part is the vector `W2 @ b1 + b2`. As a consequence, a multilayer neural network made entirely of `Dense` layers without activations would be equivalent to a single `Dense` layer. This “deep” neural network would just be a linear model in disguise! This is why we need activation functions, like `relu` (seen in action in figure 2.13). Thanks to activation functions, a chain of `Dense` layers can be made to implement very complex, nonlinear geometric transformation, resulting in very rich hypothesis spaces for your deep neural networks. We cover this idea in more detail in the next chapter.

Figure 2.13 Affine transform followed by `relu` activation

2.3.6

A geometric interpretation of deep learning

You just learned that neural networks consist entirely of chains of tensor operations and that all of these tensor operations are just simple geometric transformations of the input data. It follows that you can interpret a neural network as a very complex geometric transformation in a high-dimensional space, implemented via a series of simple steps.

In 3D, the following mental image may prove useful. Imagine two sheets of colored paper: one red and one blue. Put one on top of the other. Now crumple them together into a small ball. That crumpled paper ball is your input data, and each s …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.57 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.56 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.58 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.53 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.52 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.48 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.59 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.46 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.56 (printed 32) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › Tensor product
The tensor product, also called dot product or matmul (short for “matrix multiplication”) is one of the most common, most useful tensor operations.

In NumPy, a tensor product is done using the `np.matmul` function, and in Keras, with the `keras.ops.matmul` function. Its shorthand is the `@` operator in Python:

### PDF p.57 (printed 33) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Note that as soon as one of the two tensors has an `ndim` greater than 1, `matmul` is no longer symmetric, which is to say that `matmul(x,` `y)` isn’t the same as `matmul(y, x)`.

Of course, a tensor product generalizes to tensors with an arbitrary number of axes. The most common applications may be the product between two matrices. You can take the product of two matrices `x` and `y` (`matmul(x,` `y)`) if and only if `x.shape[1] == ` `y.shape[0]`. The result is a matrix with shape `(x.shape[0],` `y.shape[1])`, where the coefficients are the vector products between the rows of `x` and the columns of `y`. Here’s the naive implementation:

### PDF p.52 (printed 28) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › The gears of neural networks: Tensor operations
This layer can be interpreted as a function, which takes as input a matrix and returns another matrix—a new representation for the input tensor. Specifically, the function is as follows (where `W` is a matrix and `b` is a vector, both attributes of the layer):

### PDF p.53 (printed 29) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Let’s unpack this. We have three tensor operations here:

A tensor product (`matmul`) between the input tensor and a tensor named `W`.

An addition (`+`) between the resulting matrix and a vector `b`.

A `relu` operation: `relu(x)` is `max(x,` `0)`. `"relu"` stands for “REctified Linear Unit.”

NOTE Although this section deals entirely with linear algebra expressions, you won’t find any mathematical notation in this book. I’ve found that mathematical concepts can be more readily mastered by programmers with no mathematical background if they’re expressed as short Python snippets instead of mathematical equations. So we’ll use NumPy code throughout.

### PDF p.59 (printed 35) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › Geometric interpretation of tensor operations
It’s a point in a 2D space (see figure 2.6). It’s common to picture a vector as an arrow linking the origin to the point, as shown in figure 2.7.

Let’s consider a new point, `B` `=` `[1,` `0.25]`, which we’ll add to the previous one. This is done geometrically by chaining together the vector arrows, with the resulting location being the vector representing the sum of the previous two vectors (see figure 2.8). As you can see, adding a vector `B` to a vector `A` represents the action of copying point A in

### PDF p.56 (printed 32) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › Tensor product
Mathematically, what does the `matmul` operation do? Let’s start with the product of two vectors `x` and `y`. It’s computed as follows:

### PDF p.59 (printed 35) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › Geometric interpretation of tensor operations
Because the contents of the tensors manipulated by tensor operations can be interpreted as coordinates of points in some geometric space, all tensor operations have a geometric interpretation. For instance, let’s consider addition. We’ll start with the following vector:

### PDF p.58 (printed 34) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
And so on.

Figure 2.5 Matrix product box diagram
