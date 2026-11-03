# Generation bundle for page_095  (PDF page 95, printed page 71)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.3 Introduction to TensorFlow
Subsection: 3.3.2 An end-to-end example: A linear classifier in pure TensorFlow
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_095.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_095.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p095_e03] (body) Now, let’s create a linear classifier that can learn to separate these two blobs. A linear classifier is an affine transformation (`prediction` `=` `matmul(input,` `W)` `+` `b`) trained to minimize the square of the difference between predictions and the targets.
[p095_e04] (body) As you’ll see, it’s actually a much simpler example than the end-to-end example of a toy two-layer neural network from the end of chapter 2. However, this time, you should be able to understand everything about the code, line by line. Let’s create our variables `W` and `b`, initialized with random values and with zeros, respectively.
[p095_e05] (listing_caption) Listing 3.17 Creating the linear classifier variables
[p095_e06] (code)
```
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
```
[p095_e07] (body) Here’s our forward pass function.
[p095_e08] (listing_caption) Listing 3.18 The forward pass function
[p095_e09] (code)
```
def model(inputs, W, b):
    return tf.matmul(inputs, W) + b
```
[p095_e10] (body) Because our linear classifier operates on 2D inputs, `W` is really just two scalar coefficients: `W` `=` `[[w1],` `[w2]]`. Meanwhile, `b` is a single scalar coefficient. As such, for given input point `[x,` `y]`, its prediction value is `prediction` `=` `[[w1],` `[w2]]` `•` `[x,` `y]` `+` `b` `=` `w1` `*` `x + w2` `*` `y` `+` `b`.
[p095_e11] (body) Here’s our loss function.
[p095_e12] (listing_caption) Listing 3.19 The mean squared error loss function
[p095_e13] (code)
```
def mean_squared_error(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)
```

## Code annotations printed next to listings (callouts, in order)
- The inputs will be 2D points.
- The output predictions will be a single score per sample
- (close to 0 if the sample is predicted to be in class 0, and
- close to 1 if the sample is predicted to be in class 1).
- per_sample_losses will be a tensor with the
- same shape as targets and predictions,
- containing per-sample loss scores.
- We need to average these per-sample
- loss scores into a single scalar loss
- value: reduce_mean does this.

## Listings
- Listing 3.17: Creating the linear classifier variables (code elements p095_e06)
- Listing 3.18: The forward pass function (code elements p095_e09)
- Listing 3.19: The mean squared error loss function (code elements p095_e13)

## Equation candidates (formulas written inline in the text)
- [p095_e10] As such, for given input point `[x,` `y]`, its prediction value is `prediction` `=` `[[w1],` `[w2]]` `•` `[x,` `y]` `+` `b` `=` `w1` `*` `x + w2` `*` `y` `+` `b`.

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

## Previous page (PDF 94) text, for continuity
`negative_samples` and `positive_samples` are both arrays with shape `(1000,` `2)`. Let’s stack them into a single array with shape `(2000,` `2)`.

Listing 3.14 Stacking the two classes into an array with shape `(2000, 2)`

Let’s generate the corresponding target labels, an array of 0s and 1s of shape `(2000,` `1)`, where `targets[i,` `0]` is 0 if `inputs[i]` belongs to class 0 (and inversely).

Listing 3.15 Generating the corresponding targets (0 and 1)

Let’s plot our data with Matplotlib, a well-known Python data visualization library (it comes preinstalled in Colab, so no need for you to install it yourself), as shown in figure 3.1.

Listing 3.16 Plotting the two point classes

Figure 3.1 Our synthetic data: two classes of random points in the 2D plane

## Next page (PDF 96) text, for continuity
Now, we move to the training step, which receives some training data and updates the weights `W` and `b` to minimize the loss on the data.

Listing 3.20 The training-step function

For simplicity, we’ll do batch training instead of mini-batch training: we’ll run each training step (gradient computation and weight update) on the entire data, rather than iterate over the data in small batches. On one hand, this means that each training step will take much longer to run, since we compute the forward pass and the gradients for 2,000 samples at once. On the other hand, each gradient update will be much more effective at reducing the loss on the training data, since it will encompass information from all training samples instead of, say, only 128 random samples. As a result, we will need many fewer steps of training, and we should use a larger learning rate than what we would typically use for mini-batch training (we’ll use `learning_rate = 0.1`, as previously defined).

Listing 3.21 The batch training loop

After 40 steps, the training loss seems to have stabilized around 0.025. Let’s plot how our linear model classifies the training data points, as shown in figure 3.2. Because our targets are 0s and 1s a given input point will be classified as “0” if its prediction value is below 0.5, and as “1” if it is above 0.5:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.102 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.93 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.113 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.97 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.112 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.104 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.111 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.93 (printed 69) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › An end-to-end example: A linear classifier in pure TensorFlow
You know about tensors, variables, and tensor operations, and you know how to compute gradients. That’s enough to build any TensorFlow-based machine learning model based on gradient descent. Let’s walk through an end-to-end example to make sure everything is crystal clear.

In a machine learning job interview, you may be asked to implement a linear classifier from scratch: a very simple task that serves as a filter between candidates who have some minimal machine learning background, and those who don’t. Let’s get you past that filter, and use your newfound knowledge of TensorFlow to implement such a linear classifier. First, let’s come up with some nicely linearly separable synthetic data to work with: two classes of points in a 2D plane.

Listing 3.13 Generating two classes of random points in a 2D plane

### PDF p.102 (printed 78) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › An end-to-end example: A linear classifier in pure PyTorch
This is our loss function. We just switch from `tf.square` to `torch.square` and from `tf.reduce_mean` to `torch.mean`:

### PDF p.102 (printed 78) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › An end-to-end example: A linear classifier in pure PyTorch
You now know enough to rewrite our linear classifier in PyTorch. It will stay very similar to the TensorFlow one—the only major difference is how we compute the gradients.

Let’s start by creating our model variables. Don’t forget to pass `requires_grad=True` so we can compute gradients with respect to them:

### PDF p.93 (printed 69) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › An end-to-end example: A linear classifier in pure TensorFlow
Listing 3.13 Generating two classes of random points in a 2D plane
import numpy as np
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)

### PDF p.112 (printed 88) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › An end-to-end example: A linear classifier in pure JAX
Now you know enough JAX to write the JAX version of our linear classifier example. There are two major differences from the TensorFlow and PyTorch versions you’ve already seen:

All functions we will create will be stateless. That means the state (the arrays `W` and `b`) will be provided as function arguments, and if they get modified by the function, their new value will be returned by the function.

Gradients are computed using the JAX `value_and_grad()` utility.

Let’s get started. The model function and the mean squared error function should look familiar:

### PDF p.97 (printed 73) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
Figure 3.2 Our model’s predictions on the training inputs: pretty similar to the training targets

Recall that the prediction value for a given point `[x, y]` is simply `prediction == [[w1], ` `[w2]] • [x, y] + b == w1 * x + w2 * y + b`. Thus, class “0” is defined as `w1 * x + w2 * ` `y + b < 0.5` and class “1” is defined as `w1 * x + w2 * y + b > 0.5`. You’ll notice that what you’re looking at is really the equation of a line in the 2D plane: `w1 * x + w2 * y + b ` `= 0.5`. Class 1 is above the line; class 0 is below the line. You may be used to seeing line equations in the format `y = a * x + b`; in the same format, our line becomes `y = - w1 ` `/ w2 * x + (0.5 - b) / w2`.

Let’s plot this line, as shown in figure 3.3:

### PDF p.98 (printed 74) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
This is really what a linear classifier is all about: finding the parameters of a line (or, in higher-dimensional spaces, a hyperplane) neatly separating two classes of data.

### PDF p.88 (printed 64) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › First steps with TensorFlow
Over the next paragraphs, you’ll get familiar with all the basics of TensorFlow. We’ll cover the following key concepts:

Tensors and variables

Numerical operations in TensorFlow

Computing gradients with a `GradientTape`

Making TensorFlow functions fast by using just-in-time compilation

We’ll then conclude the introduction with an end-to-end example: a pure-TensorFlow implementation of linear regression.

Let’s get those tensors flowing.
