# Generation bundle for page_094  (PDF page 94, printed page 70)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.3 Introduction to TensorFlow
Subsection: 3.3.2 An end-to-end example: A linear classifier in pure TensorFlow
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_094.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_094.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p094_e03] (body) `negative_samples` and `positive_samples` are both arrays with shape `(1000,` `2)`. Let’s stack them into a single array with shape `(2000,` `2)`.
[p094_e04] (listing_caption) Listing 3.14 Stacking the two classes into an array with shape `(2000, 2)`
[p094_e05] (code)
```
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
```
[p094_e06] (body) Let’s generate the corresponding target labels, an array of 0s and 1s of shape `(2000,` `1)`, where `targets[i,` `0]` is 0 if `inputs[i]` belongs to class 0 (and inversely).
[p094_e07] (listing_caption) Listing 3.15 Generating the corresponding targets (0 and 1)
[p094_e08] (code)
```
targets = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype="float32"),
        np.ones((num_samples_per_class, 1), dtype="float32"),
    )
)
```
[p094_e09] (body) Let’s plot our data with Matplotlib, a well-known Python data visualization library (it comes preinstalled in Colab, so no need for you to install it yourself), as shown in figure 3.1.
[p094_e10] (listing_caption) Listing 3.16 Plotting the two point classes
[p094_e11] (code)
```
import matplotlib.pyplot as plt
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()
```
[p094_e12] (figure_caption) Figure 3.1 Our synthetic data: two classes of random points in the 2D plane

## Listings
- Listing 3.14: Stacking the two classes into an array with shape `(2000, 2)` (code elements p094_e05)
- Listing 3.15: Generating the corresponding targets (0 and 1) (code elements p094_e08)
- Listing 3.16: Plotting the two point classes (code elements p094_e11)

## Figures
- Figure 3.1: Our synthetic data: two classes of random points in the 2D plane  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras (PDF pages 84-127)
  - 3.1 A brief history of deep learning frameworks (p.85)
  - 3.2 How these frameworks relate to each other (p.87)
  - 3.3 Introduction to TensorFlow (p.87)
      - 3.3.1 First steps with TensorFlow (p.88)
      - 3.3.2 An end-to-end example: A linear classifier in pure TensorFlow (p.93)
      - 3.3.3 What makes the TensorFlow approach unique (p.98)
  - 3.4 Introduction to PyTorch (p.98)
      - 3.4.1 First steps with PyTorch (p.99)
      - 3.4.2 An end-to-end example: A linear classifier in pure PyTorch (p.102)
      - 3.4.3 What makes the PyTorch approach unique (p.105)
  - 3.5 Introduction to JAX (p.106)
      - 3.5.1 First steps with JAX (p.106)
      - 3.5.2 Tensors in JAX (p.107)
      - 3.5.3 Random number generation in JAX (p.107)
      - 3.5.4 An end-to-end example: A linear classifier in pure JAX (p.112)
      - 3.5.5 What makes the JAX approach unique (p.114)
  - 3.6 Introduction to Keras (p.114)
      - 3.6.1 First steps with Keras (p.115)
      - 3.6.2 Layers: The building blocks of deep learning (p.116)
      - 3.6.3 From layers to models (p.120)
      - 3.6.4 The “compile” step: Configuring the learning process (p.121)
      - 3.6.5 Picking a loss function (p.123)
      - 3.6.6 Understanding the fit method (p.124)
      - 3.6.7 Monitoring loss and metrics on validation data (p.125)
      - 3.6.8 Inference: Using a model after training (p.126)

## Previous page (PDF 93) text, for continuity
First, the default one, which we refer to as “graph mode.” Any function decorated with `@tf.function` runs in graph mode.

Second, compilation with XLA, a high-performance compiler for ML (it’s short for Accelerated Linear Algebra). You can turn it on by specifying `jit_` `compile=True`, like this:

It is often the case that compiling a function with XLA will make it run faster than graph mode—though it takes more time to execute the function the first time, since the compiler has more work to do.

3.3.2

An end-to-end example: A linear classifier in pure TensorFlow

You know about tensors, variables, and tensor operations, and you know how to compute gradients. That’s enough to build any TensorFlow-based machine learning model based on gradient descent. Let’s walk through an end-to-end example to make sure everything is crystal clear.

In a machine learning job interview, you may be asked to implement a linear classifier from scratch: a very simple task that serves as a filter between candidates who have some minimal machine learning background, and those who don’t. Let’s get you past that filter, and use your newfound knowledge of TensorFlow to implement such a linear classifier. First, let’s come up with some nicely linearly separable synthetic data to work with: two classes of points in a 2D plane.

Listing 3.13 Generating two classes of random points in a 2D plane

## Next page (PDF 95) text, for continuity
Now, let’s create a linear classifier that can learn to separate these two blobs. A linear classifier is an affine transformation (`prediction` `=` `matmul(input,` `W)` `+` `b`) trained to minimize the square of the difference between predictions and the targets.

As you’ll see, it’s actually a much simpler example than the end-to-end example of a toy two-layer neural network from the end of chapter 2. However, this time, you should be able to understand everything about the code, line by line. Let’s create our variables `W` and `b`, initialized with random values and with zeros, respectively.

Listing 3.17 Creating the linear classifier variables

Here’s our forward pass function.

Listing 3.18 The forward pass function

Because our linear classifier operates on 2D inputs, `W` is really just two scalar coefficients: `W` `=` `[[w1],` `[w2]]`. Meanwhile, `b` is a single scalar coefficient. As such, for given input point `[x,` `y]`, its prediction value is `prediction` `=` `[[w1],` `[w2]]` `•` `[x,` `y]` `+` `b` `=` `w1` `*` `x + w2` `*` `y` `+` `b`.

Here’s our loss function.

Listing 3.19 The mean squared error loss function

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.109 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.89 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.107 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.90 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.97 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.88 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.108 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.98 (printed 74) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
This is really what a linear classifier is all about: finding the parameters of a line (or, in higher-dimensional spaces, a hyperplane) neatly separating two classes of data.

### PDF p.96 (printed 72) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
After 40 steps, the training loss seems to have stabilized around 0.025. Let’s plot how our linear model classifies the training data points, as shown in figure 3.2. Because our targets are 0s and 1s a given input point will be classified as “0” if its prediction value is below 0.5, and as “1” if it is above 0.5:

### PDF p.47 (printed 23) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › Key attributes
A tensor is defined by three key attributes:

Number of axes (rank)—For instance, a rank-3 tensor has three axes, and a matrix has two axes. This is also called the tensor’s `ndim` in Python libraries such as NumPy, JAX, TensorFlow, and PyTorch.

Shape—This is a tuple of integers that describes how many dimensions the tensor has along each axis. For instance, the previous matrix example has shape `(3,` `5)`, and the rank-3 tensor example has shape `(3,` `3,` `5)`. A vector has a shape with a single element, such as `(5,)`, whereas a scalar has an empty shape, `()`.

Data type (usually called `dtype` in Python libraries)—This is the type of the data contained in the tensor; for instance, a tensor’s type could be `float16`, `float32`, `float64`, `uint8`, `bool`, and so on. In TensorFlow, you are also likely to come across `string` tensors.

To make this more concrete, let’s look back at the data we processed in the MNIST example. First, we load the MNIST dataset:

### PDF p.102 (printed 78) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › An end-to-end example: A linear classifier in pure PyTorch
def mean_squared_error(targets, predictions):
    per_sample_losses = torch.square(targets - predictions)
    return torch.mean(per_sample_losses)

### PDF p.102 (printed 78) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › An end-to-end example: A linear classifier in pure PyTorch
You now know enough to rewrite our linear classifier in PyTorch. It will stay very similar to the TensorFlow one—the only major difference is how we compute the gradients.

Let’s start by creating our model variables. Don’t forget to pass `requires_grad=True` so we can compute gradients with respect to them:

### PDF p.97 (printed 73) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
Figure 3.2 Our model’s predictions on the training inputs: pretty similar to the training targets

Recall that the prediction value for a given point `[x, y]` is simply `prediction == [[w1], ` `[w2]] • [x, y] + b == w1 * x + w2 * y + b`. Thus, class “0” is defined as `w1 * x + w2 * ` `y + b < 0.5` and class “1” is defined as `w1 * x + w2 * y + b > 0.5`. You’ll notice that what you’re looking at is really the equation of a line in the 2D plane: `w1 * x + w2 * y + b ` `= 0.5`. Class 1 is above the line; class 0 is below the line. You may be used to seeing line equations in the format `y = a * x + b`; in the same format, our line becomes `y = - w1 ` `/ w2 * x + (0.5 - b) / w2`.

Let’s plot this line, as shown in figure 3.3:

### PDF p.88 (printed 64) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Constant tensors
Listing 3.1 All-ones or all-zeros tensors
>>> import tensorflow as tf
>>> tf.ones(shape=(2, 1))
tf.Tensor([[1.], [1.]], shape=(2, 1), dtype=float32)
>>> tf.zeros(shape=(2, 1))
tf.Tensor([[0.], [0.]], shape=(2, 1), dtype=float32)
>>> tf.constant([1, 2, 3], dtype="float32")
tf.Tensor([1., 2., 3.], shape=(3,), dtype=float32)

### PDF p.272 (printed 248) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Understanding TensorFlow Dataset objects
Listing 8.10 Instantiating a `Dataset` from a NumPy array
import numpy as np
import tensorflow as tf
random_numbers = np.random.normal(size=(1000, 16))
dataset = tf.data.Dataset.from_tensor_slices(random_numbers)
