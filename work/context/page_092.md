# Generation bundle for page_092  (PDF page 92, printed page 68)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.3 Introduction to TensorFlow
Subsection: 3.3.1 First steps with TensorFlow
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_092.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_092.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p092_e03] (body) an object with regard to time is the speed of that object, and the second-order gradient is its acceleration.
[p092_e04] (body) If you measure the position of a falling apple along a vertical axis over time, and find that it verifies `position(time)` `=` `4.9` `*` `time` `**` `2`, what is its acceleration? Let’s use two nested gradient tapes to find out.
[p092_e05] (listing_caption) Listing 3.12 Using nested gradient tapes to compute second-order gradients
[p092_e06] (code)
```
time = tf.Variable(0.0)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time**2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)
```
[p092_e07] (heading_subsubsection) Making TensorFlow functions fast using compilation
[p092_e08] (body) All the TensorFlow code you’ve written so far has been executing “eagerly.” This means operations are executed one after the other in the Python runtime, much like any Python code or NumPy code. Eager execution is great for debugging, but it is typically quite slow. It can often be beneficial to parallelize some computation, or “fuse” operations—replacing two consecutive operations, like `matmul` followed by `relu`, with a single, more efficient operation that does the same thing without materializing the intermediate output.
[p092_e09] (body) This can be achieved via compilation. The general idea of compilation is to take certain functions you’ve written in Python, lift them out of Python, automatically rewrite them into a faster and more efficient “compiled program,” and then call that program from the Python runtime. The main benefit of compilation is improved performance. There’s a drawback too: the code you write is no longer the code that gets executed, which can make the debugging experience painful. Only turn on compilation after you’ve already debugged your code in the Python runtime. You can apply compilation to any TensorFlow function by wrapping it in a `tf.function` decorator, like this:
[p092_e10] (code)
```
@tf.function
def dense(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
```
[p092_e11] (body) When you do this, any call to `dense()` is replaced with a call to a compiled program that implements a more optimized version of the function. The first call to the function will take a bit longer, because TensorFlow will be compiling your code. This only happens once—all subsequent calls to the same function will be fast.
[p092_e12] (body) TensorFlow has two compilation modes:

## Code annotations printed next to listings (callouts, in order)
- We use the outer tape to
- compute the gradient of
- the gradient from the
- inner tape. Naturally, the
- answer is 4.9 * 2 = 9.8.

## Listings
- Listing 3.12: Using nested gradient tapes to compute second-order gradients (code elements p092_e06)

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

## Previous page (PDF 91) text, for continuity
Here’s an equivalent of the `Dense` layer we saw in chapter 2:

Gradients in TensorFlow: A second look at the GradientTape API

So far, TensorFlow seems to look a lot like NumPy. But here’s something NumPy can’t do: retrieve the gradient of any differentiable expression with respect to any of its inputs. Just open a `GradientTape` scope, apply some computation to one or several input tensors, and retrieve the gradient of the result with respect to the inputs.

Listing 3.10 Using the `GradientTape`

This is most commonly used to retrieve the gradients of the loss of a model with respect to its weights: `gradients` `=` `tape.gradient(loss,` `weights)`.

In chapter 2, you saw how the `GradientTape` works on either a single input or a list of inputs and how inputs could be either scalars or high-dimensional tensors. So far, you’ve only seen the case where the input tensors in `tape.gradient()` were TensorFlow variables. It’s actually possible for these inputs to be any arbitrary tensor. However, only trainable variables are being tracked by default. With a constant tensor, you’d have to manually mark it as being tracked, by calling `tape.watch()` on it.

Listing 3.11 Using the `GradientTape` with constant tensor inputs

Why? Because it would be too expensive to preemptively store the information required to compute the gradient of anything with respect to anything. To avoid wasting resources, the tape needs to know what to watch. Trainable variables are watched by default because computing the gradient of a loss with regard to a list of trainable variables is the most common use case of the gradient tape.

The gradient tape is a powerful utility, even capable of computing second-order gradients—that is, the gradient of a gradient. For instance, the gradient of the position  …

## Next page (PDF 93) text, for continuity
First, the default one, which we refer to as “graph mode.” Any function decorated with `@tf.function` runs in graph mode.

Second, compilation with XLA, a high-performance compiler for ML (it’s short for Accelerated Linear Algebra). You can turn it on by specifying `jit_` `compile=True`, like this:

It is often the case that compiling a function with XLA will make it run faster than graph mode—though it takes more time to execute the function the first time, since the compiler has more work to do.

3.3.2

An end-to-end example: A linear classifier in pure TensorFlow

You know about tensors, variables, and tensor operations, and you know how to compute gradients. That’s enough to build any TensorFlow-based machine learning model based on gradient descent. Let’s walk through an end-to-end example to make sure everything is crystal clear.

In a machine learning job interview, you may be asked to implement a linear classifier from scratch: a very simple task that serves as a filter between candidates who have some minimal machine learning background, and those who don’t. Let’s get you past that filter, and use your newfound knowledge of TensorFlow to implement such a linear classifier. First, let’s come up with some nicely linearly separable synthetic data to work with: two classes of points in a 2D plane.

Listing 3.13 Generating two classes of random points in a 2D plane

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.105 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.112 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.88 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.101 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.110 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.240 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.318 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.88 (printed 64) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › First steps with TensorFlow
Over the next paragraphs, you’ll get familiar with all the basics of TensorFlow. We’ll cover the following key concepts:

Tensors and variables

Numerical operations in TensorFlow

Computing gradients with a `GradientTape`

Making TensorFlow functions fast by using just-in-time compilation

We’ll then conclude the introduction with an end-to-end example: a pure-TensorFlow implementation of linear regression.

Let’s get those tensors flowing.

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A TensorFlow training step function
TensorFlow lets you write code that looks pretty much like our pseudocode snippet. The only difference is that your forward pass should take place inside a `GradientTape` scope. You can then use the `tape` object to retrieve the gradients:

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing custom training step functions
This snippet is pseudocode rather than real code because it includes an imaginary function, `get_gradients_of()`. In reality, retrieving gradients is done in a way that is specific to your current backend—JAX, TensorFlow, or PyTorch.

Let’s use what you learned about each framework in chapter 3 to implement a real version of this `train_step()` function. We’ll start with TensorFlow and PyTorch because these two make the job relatively easy, so they’re a good place to start. We’ll end with JAX, which is quite a bit more complex.

### PDF p.318 (printed 294) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2.1 Gradient ascent in TensorFlow
For TensorFlow, we can just open a `GradientTape` scope and compute the loss inside of it to retrieve the gradients we need. We’ll use a `@tf.function` decorator to speed up computation:

### PDF p.318 (printed 294) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
(continued)

You should use `model(x)` when you need to retrieve the gradients of the model call. And you should use `predict()` if you just need the output value. In other words, always use `predict()`, unless you’re in the middle of writing a low-level gradient descent loop (as we are now).

A non-obvious trick to help the gradient-ascent process go smoothly is to normalize the gradient tensor by dividing it by its L2 norm (the square root of the sum of the squares of the values in the tensor). This ensures that the magnitude of the updates done to the input image is always within the same range.

Let’s set up the gradient ascent step function. Anything that involves gradients requires calling backend-level APIs, such as `GradientTape` in TensorFlow, `.backward()` in PyTorch, and `jax.grad()` in JAX. Let’s line up all the code snippets for each of the three backends, starting with TensorFlow.

### PDF p.105 (printed 81) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Making PyTorch modules fast using compilation
The resulting object is intended to work identically to the original—except the forward and backward pass should run faster.

You can also use `torch.compile()` as a function decorator:

### PDF p.105 (printed 81) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Making PyTorch modules fast using compilation
In practice, most PyTorch code out there does not use compilation and simply runs eagerly, as the compiler may not always work with all models and may not always result in a speedup when it does work. Unlike in TensorFlow and Jax where compilation was built in from the inception of the library, PyTorch’s compiler is a relatively recent addition.

### PDF p.107 (printed 83) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
Computing gradients via `jax.grad` and `jax.value_and_grad`

Making JAX functions fast by leveraging just-in-time compilation

Let’s get started.
