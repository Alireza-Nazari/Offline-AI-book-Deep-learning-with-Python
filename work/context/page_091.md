# Generation bundle for page_091  (PDF page 91, printed page 67)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.3 Introduction to TensorFlow
Subsection: 3.3.1 First steps with TensorFlow
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_091.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_091.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p091_e03] (body) Here’s an equivalent of the `Dense` layer we saw in chapter 2:
[p091_e04] (code)
```
def dense(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
```
[p091_e05] (heading_subsubsection) Gradients in TensorFlow: A second look at the GradientTape API
[p091_e06] (body) So far, TensorFlow seems to look a lot like NumPy. But here’s something NumPy can’t do: retrieve the gradient of any differentiable expression with respect to any of its inputs. Just open a `GradientTape` scope, apply some computation to one or several input tensors, and retrieve the gradient of the result with respect to the inputs.
[p091_e07] (listing_caption) Listing 3.10 Using the `GradientTape`
[p091_e08] (code)
```
input_var = tf.Variable(initial_value=3.0)
with tf.GradientTape() as tape:
    result = tf.square(input_var)
gradient = tape.gradient(result, input_var)
```
[p091_e09] (body) This is most commonly used to retrieve the gradients of the loss of a model with respect to its weights: `gradients` `=` `tape.gradient(loss,` `weights)`.
[p091_e10] (body) In chapter 2, you saw how the `GradientTape` works on either a single input or a list of inputs and how inputs could be either scalars or high-dimensional tensors. So far, you’ve only seen the case where the input tensors in `tape.gradient()` were TensorFlow variables. It’s actually possible for these inputs to be any arbitrary tensor. However, only trainable variables are being tracked by default. With a constant tensor, you’d have to manually mark it as being tracked, by calling `tape.watch()` on it.
[p091_e11] (listing_caption) Listing 3.11 Using the `GradientTape` with constant tensor inputs
[p091_e12] (code)
```
input_const = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(input_const)
    result = tf.square(input_const)
gradient = tape.gradient(result, input_const)
```
[p091_e13] (body) Why? Because it would be too expensive to preemptively store the information required to compute the gradient of anything with respect to anything. To avoid wasting resources, the tape needs to know what to watch. Trainable variables are watched by default because computing the gradient of a loss with regard to a list of trainable variables is the most common use case of the gradient tape.
[p091_e14] (body) The gradient tape is a powerful utility, even capable of computing second-order gradients—that is, the gradient of a gradient. For instance, the gradient of the position of

## Listings
- Listing 3.10: Using the `GradientTape` (code elements p091_e08)
- Listing 3.11: Using the `GradientTape` with constant tensor inputs (code elements p091_e12)

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

## Previous page (PDF 90) text, for continuity
The state of a variable can be modified via its `assign` method.

Listing 3.6 Assigning a value to a `Variable`

Assignment also works for a subset of the coefficients.

Listing 3.7 Assigning a value to a subset of a `Variable`

Similarly, `assign_add` and `assign_sub` are efficient equivalents of `+=` and `-=`.

Listing 3.8 Using `assign_add`

Tensor operations: Doing math in TensorFlow

Just like NumPy, TensorFlow offers a large collection of tensor operations to express mathematical formulas. Here are a few examples.

Listing 3.9 A few basic math operations in TensorFlow

## Next page (PDF 92) text, for continuity
an object with regard to time is the speed of that object, and the second-order gradient is its acceleration.

If you measure the position of a falling apple along a vertical axis over time, and find that it verifies `position(time)` `=` `4.9` `*` `time` `**` `2`, what is its acceleration? Let’s use two nested gradient tapes to find out.

Listing 3.12 Using nested gradient tapes to compute second-order gradients

Making TensorFlow functions fast using compilation

All the TensorFlow code you’ve written so far has been executing “eagerly.” This means operations are executed one after the other in the Python runtime, much like any Python code or NumPy code. Eager execution is great for debugging, but it is typically quite slow. It can often be beneficial to parallelize some computation, or “fuse” operations—replacing two consecutive operations, like `matmul` followed by `relu`, with a single, more efficient operation that does the same thing without materializing the intermediate output.

This can be achieved via compilation. The general idea of compilation is to take certain functions you’ve written in Python, lift them out of Python, automatically rewrite them into a faster and more efficient “compiled program,” and then call that program from the Python runtime. The main benefit of compilation is improved performance. There’s a drawback too: the code you write is no longer the code that gets executed, which can make the debugging experience painful. Only turn on compilation after you’ve already debugged your code in the Python runtime. You can apply compilation to any TensorFlow function by wrapping it in a `tf.function` decorator, like this:

When you do this, any call to `dense()` is replaced with a call to a compiled program that implements a more optimized version  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.101 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.110 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.102 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.318 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.111 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.88 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.318 (printed 294) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2.1 Gradient ascent in TensorFlow
For TensorFlow, we can just open a `GradientTape` scope and compute the loss inside of it to retrieve the gradients we need. We’ll use a `@tf.function` decorator to speed up computation:

### PDF p.80 (printed 56) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Gradient computation
Now, there’s just one thing we’re still missing: gradient computation (represented by the `get_gradients_of_loss_wrt_weights()` function in listing 2.9). In the previous section, we outlined how we could use the chain rule to obtain the gradients of a chain of functions given their individual derivatives, a process known as backpropagation. We could reimplement backpropagation from scratch here, but that would be rather cumbersome, especially since we’re using a `softmax` operation and a crossentropy loss, which have fairly verbose derivatives.

Instead, we can rely on the automatic differentiation mechanism that’s built into one of the low-level frameworks supported by Keras, such as TensorFlow, JAX, or PyTorch. For the sake of the example, let’s go with TensorFlow here. You’ll learn more about TensorFlow, JAX, and PyTorch in the next chapter. The API through which you can use TensorFlow’s automatic differentiation capabilities is the `tf.GradientTape` object. It’s a Python scope that will “record” the tensor operations that run inside it, in the form of a computation graph (sometimes called a tape). This graph can then be used to retrieve the gradient of any scalar value with res

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Computing gradients with PyTorch
There’s no explicit “gradient tape” in PyTorch. A similar mechanism does exist: when you run any computation in PyTorch, the framework creates a one-time computation graph (a “tape”) that records what just happened. However, that tape is hidden from the user. The public API for using it is at the level of tensors themselves: you can call `tensor.backward()` to run backpropagation through all operations previously executed that led to that tensor. Doing this will populate the `.grad` attribute of all tensors that are tracking gradients.

Listing 3.26 Computing a gradient with `.backward()`

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A TensorFlow training step function
TensorFlow lets you write code that looks pretty much like our pseudocode snippet. The only difference is that your forward pass should take place inside a `GradientTape` scope. You can then use the `tape` object to retrieve the gradients:

### PDF p.326 (printed 302) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3.1 Getting the gradient of the top class: TensorFlow version
Let’s start with the TensorFlow version, once again using `GradientTape`.

Listing 10.20 Computing the top class gradients with TensorFlow

### PDF p.318 (printed 294) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
(continued)

You should use `model(x)` when you need to retrieve the gradients of the model call. And you should use `predict()` if you just need the output value. In other words, always use `predict()`, unless you’re in the middle of writing a low-level gradient descent loop (as we are now).

A non-obvious trick to help the gradient-ascent process go smoothly is to normalize the gradient tensor by dividing it by its L2 norm (the square root of the sum of the squares of the values in the tensor). This ensures that the magnitude of the updates done to the input image is always within the same range.

Let’s set up the gradient ascent step function. Anything that involves gradients requires calling backend-level APIs, such as `GradientTape` in TensorFlow, `.backward()` in PyTorch, and `jax.grad()` in JAX. Let’s line up all the code snippets for each of the three backends, starting with TensorFlow.

### PDF p.88 (printed 64) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › First steps with TensorFlow
Over the next paragraphs, you’ll get familiar with all the basics of TensorFlow. We’ll cover the following key concepts:

Tensors and variables

Numerical operations in TensorFlow

Computing gradients with a `GradientTape`

Making TensorFlow functions fast by using just-in-time compilation

We’ll then conclude the introduction with an end-to-end example: a pure-TensorFlow implementation of linear regression.

Let’s get those tensors flowing.

### PDF p.241 (printed 217) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply(gradients, model.trainable_weights)
    return loss
