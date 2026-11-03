# Generation bundle for page_102  (PDF page 102, printed page 78)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.4 Introduction to PyTorch
Subsection: 3.4.2 An end-to-end example: A linear classifier in pure PyTorch
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_102.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_102.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p102_e03] (body) in the following code, `input_var.grad` is not the gradient of `square(input_var)` with respect to `input_var`; rather, it is the sum of that gradient and the previously computed gradient—its value has doubled since our last code snippet:
[p102_e04] (code)
```
>>> result = torch.square(input_var)
>>> result.backward()
>>> input_var.grad
tensor(12.)
```
[p102_e05] (body) To reset gradients, you can just set `.grad` to `None`:
[p102_e06] (code)
```
>>> input_var.grad = None
```
[p102_e07] (body) Now let’s put this into practice!
[p102_e08] (heading_subsection) 3.4.2
[p102_e09] (heading_subsection) An end-to-end example: A linear classifier in pure PyTorch
[p102_e10] (body) You now know enough to rewrite our linear classifier in PyTorch. It will stay very similar to the TensorFlow one—the only major difference is how we compute the gradients.
[p102_e11] (body) Let’s start by creating our model variables. Don’t forget to pass `requires_grad=True` so we can compute gradients with respect to them:
[p102_e12] (code)
```
input_dim = 2
output_dim = 1
W = torch.rand(input_dim, output_dim, requires_grad=True)
b = torch.zeros(output_dim, requires_grad=True)
```
[p102_e13] (body) This is our model—no difference so far. We just went from `tf.matmul` to `torch.matmul`:
[p102_e14] (code)
```
def model(inputs, W, b):
    return torch.matmul(inputs, W) + b
```
[p102_e15] (body) This is our loss function. We just switch from `tf.square` to `torch.square` and from `tf.reduce_mean` to `torch.mean`:
[p102_e16] (code)
```
def mean_squared_error(targets, predictions):
    per_sample_losses = torch.square(targets - predictions)
    return torch.mean(per_sample_losses)
```
[p102_e17] (body) Now for the training step. Here’s how it works:

## Code annotations printed next to listings (callouts, in order)
- .grad will sum all gradient values
- from each time backward() is called.

## Equation candidates (formulas written inline in the text)
- [p102_e11] Don’t forget to pass `requires_grad=True` so we can compute gradients with respect to them:
- [p102_e12] input_dim = 2
output_dim = 1
W = torch.rand(input_dim, output_dim, requires_grad=True)
b = torch.zeros(output_dim, requires_grad=True)

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

## Previous page (PDF 101) text, for continuity
Tensor operations: Doing math in PyTorch

Math in PyTorch works just the same as math in NumPy or TensorFlow, although much like TensorFlow, the PyTorch API often diverges in subtle ways from the NumPy API.

Listing 3.25 A few basic math operations in PyTorch

Here’s a dense layer:

Computing gradients with PyTorch

There’s no explicit “gradient tape” in PyTorch. A similar mechanism does exist: when you run any computation in PyTorch, the framework creates a one-time computation graph (a “tape”) that records what just happened. However, that tape is hidden from the user. The public API for using it is at the level of tensors themselves: you can call `tensor.backward()` to run backpropagation through all operations previously executed that led to that tensor. Doing this will populate the `.grad` attribute of all tensors that are tracking gradients.

Listing 3.26 Computing a gradient with `.backward()`

If you call `backward()` multiple times in a row, the `.grad` attribute will “accumulate” gradients: each new call will sum the new gradient with the preexisting one. For instance,

## Next page (PDF 103) text, for continuity
1 `loss.backward()` runs backpropagation starting from the `loss` output node and

populates the `tensor.grad` attribute on all tensors that were involved in the computation of `loss`. `tensor.grad` represents the gradient of the loss with regard to that tensor. 2 We use the `.grad` attribute to recover the gradients of the loss with regard to `W` and `b`. 3 We update `W` and `b` using those gradients. Because these updates are not intended to be part of the backward pass, we do them inside a `torch.no_grad()` scope, which skips gradient computation for everything inside it. 4 We reset the contents of the `.grad` property of our `W` and `b` parameters, by setting it to `None`. If we didn’t do this, gradient values would accumulate across multiple calls to `training_step()`, resulting in invalid values:

This could be made even simpler—let’s see how.

Packaging state and computation with the Module class

PyTorch also has a higher-level, object-oriented API for performing backpropagation, which requires relying on two new classes: the `torch.nn.Module` class and an optimizer class from the `torch.optim` module, such as `torch.optim.SGD` (the equivalent of `keras
` `.optimizers.SGD`).

The general idea is to define a subclass of `torch.nn.Module`, which will

Hold some `Parameters`, to store state variables. Those are defined in the `__init__()`method.

Implement the forward pass computation in the `forward()` method.

It should look just like the following.

Listing 3.27 Defining a `torch.nn.Module`

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.104 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.95 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.113 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.110 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.91 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.112 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.112 (printed 88) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › An end-to-end example: A linear classifier in pure JAX
Now you know enough JAX to write the JAX version of our linear classifier example. There are two major differences from the TensorFlow and PyTorch versions you’ve already seen:

All functions we will create will be stateless. That means the state (the arrays `W` and `b`) will be provided as function arguments, and if they get modified by the function, their new value will be returned by the function.

Gradients are computed using the JAX `value_and_grad()` utility.

Let’s get started. The model function and the mean squared error function should look familiar:

### PDF p.93 (printed 69) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › An end-to-end example: A linear classifier in pure TensorFlow
You know about tensors, variables, and tensor operations, and you know how to compute gradients. That’s enough to build any TensorFlow-based machine learning model based on gradient descent. Let’s walk through an end-to-end example to make sure everything is crystal clear.

In a machine learning job interview, you may be asked to implement a linear classifier from scratch: a very simple task that serves as a filter between candidates who have some minimal machine learning background, and those who don’t. Let’s get you past that filter, and use your newfound knowledge of TensorFlow to implement such a linear classifier. First, let’s come up with some nicely linearly separable synthetic data to work with: two classes of points in a 2D plane.

Listing 3.13 Generating two classes of random points in a 2D plane

### PDF p.241 (printed 217) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A PyTorch training step function
When you use the PyTorch backend, all of your Keras layers and models inherit from the PyTorch `torch.nn.Module` class and expose the native `Module` API. As a result, your model, its trainable weights, and your loss tensor are all aware of each other and interact via three methods: `loss.backward()`, `weight.value.grad`, and `model
` `.zero_grad()`.

As a reminder from chapter 3, the mental model you’ve got to keep in mind is this:

With each forward pass, PyTorch builds up a one-time computation graph that keeps track of the computation that just happened.

Calling `.backward()` on any given scalar node of this graph (like your loss) will run the graph backward starting from that node, automatically populating a `tensor.grad` attribute on all tensors involved (if they satisfy `requires_` `grad=True`), containing the gradient of the output node with respect to that tensor. In particular, it will populate the `grad` attribute of your trainable parameters.

To clear the contents of that `tensor.grad` attribute, you should call `tensor.grad ` `= None` on all your tensors. Because it would be a bit cumbersome to do this on all model variables individually, you can just do it at the mo

### PDF p.319 (printed 295) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2.2 Gradient ascent in PyTorch
In the case of PyTorch, we use `loss.backward()` and `image.grad` to obtain the gradients of the loss with respect to the input image, like this.

Listing 10.12 Loss maximization via stochastic gradient ascent: PyTorch

### PDF p.95 (printed 71) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
Now, let’s create a linear classifier that can learn to separate these two blobs. A linear classifier is an affine transformation (`prediction` `=` `matmul(input,` `W)` `+` `b`) trained to minimize the square of the difference between predictions and the targets.

As you’ll see, it’s actually a much simpler example than the end-to-end example of a toy two-layer neural network from the end of chapter 2. However, this time, you should be able to understand everything about the code, line by line. Let’s create our variables `W` and `b`, initialized with random values and with zeros, respectively.

Listing 3.17 Creating the linear classifier variables

### PDF p.326 (printed 302) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3.2 Getting the gradient of the top class: PyTorch version
Next, here’s the PyTorch version, using `.backward()` and `.grad`.

Listing 10.21 Computing the top class gradients with PyTorch

### PDF p.105 (printed 81) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
Similarly, previously we needed to reset parameter gradients by hand by doing `tensor.grad = None` on each one. Now we can just do `model.zero_grad()`. Overall, this may feel a bit confusing—somehow the loss tensor, the optimizer, and the `Module` instance all seem to be aware of each other through some hidden background mechanism. They’re all interacting with one another via spooky action at a distance. Don’t worry though—you can just treat this sequence of steps (`loss.backward()` - `optimizer.step()` - `model.zero_grad()`) as a magic incantation to be recited any time you need to write a training step function. Just make sure not to forget `model.zero_` `grad()`. That would be a major bug (and it is unfortunately quite common)!

### PDF p.99 (printed 75) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › First steps with PyTorch
Over the next paragraphs, you’ll get familiar with all the basics of PyTorch. We’ll cover the following key concepts:

Tensors and parameters

Numerical operations in PyTorch

Computing gradients with the `backward()` method

Packaging computation with the `Module` class

Speeding up PyTorch by using compilation

We’ll conclude the introduction by reimplementing our linear regression end-to-end example in pure PyTorch.
