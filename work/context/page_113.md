# Generation bundle for page_113  (PDF page 113, printed page 89)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.5 Introduction to JAX
Subsection: 3.5.4 An end-to-end example: A linear classifier in pure JAX
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_113.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_113.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p113_e03] (code)
```
def model(inputs, W, b):
    return jnp.matmul(inputs, W) + b
def mean_squared_error(targets, predictions):
    per_sample_losses = jnp.square(targets - predictions)
    return jnp.mean(per_sample_losses)
```
[p113_e04] (body) To compute gradients, we need to package loss computation in a single `compute_loss()` function. It returns the total loss as a scalar, and it takes `state` as its first argument—a tuple of all the tensors we need gradients for:
[p113_e05] (code)
```
def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
    return loss
```
[p113_e06] (body) Calling `jax.value_and_grad()` on this function gives us a new function, with the same argument as `compute_loss`, which returns both the loss and the gradients of the loss with regard to the elements of `state`:
[p113_e07] (code)
```
grad_fn = jax.value_and_grad(compute_loss)
```
[p113_e08] (body) Next, we can set up our training step function. It looks straightforward. Be mindful that, unlike its TensorFlow and PyTorch equivalents, it needs to be stateless, and so it must return the updated values of the `W` and `b` tensors:
[p113_e09] (code)
```
learning_rate = 0.1
@jax.jit
def training_step(inputs, targets, W, b):
    loss, grads = grad_fn((W, b), inputs, targets)
    grad_wrt_W, grad_wrt_b = grads
    W = W - grad_wrt_W * learning_rate
    b = b - grad_wrt_b * learning_rate
    return loss, W, b
```
[p113_e10] (body) Because we won’t change the `learning_rate` during our example, we can consider it part of the function itself and not our model’s state. If we wanted to modify our learning rate during training, we’d need to pass it through as well.
[p113_e11] (body) Finally, we’re ready to run the full training loop. We initialize `W` and `b`, and we repeatedly update them via stateless calls to `training_step()`:

## Code annotations printed next to listings (callouts, in order)
- We use the jax.jit decorator to take
- advantage of XLA compilation.
- Computes the forward
- pass and backward
- pass in one go
- Updates W and b
- Make sure to return the new values
- of W and b in addition to the loss!

## Equation candidates (formulas written inline in the text)
- [p113_e07] grad_fn = jax.value_and_grad(compute_loss)

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

## Previous page (PDF 112) text, for continuity
1 Edit the loss function to return a tuple where the first entry is the loss, and the

second entry is your extra output. 2 Pass the argument `has_aux=True` to `value_and_grad()`. This tells `value_and_` `grad()` to return not just the gradient but also the “auxiliary” output(s) of `compute_loss()`, like this:

Admittedly, things are starting to be pretty convoluted at this point. Don’t worry, though; this is about as hard as JAX gets! Almost everything else is simpler by comparison.

Making JAX functions fast with @jax.jit

One more thing. As a JAX user, you will frequently use the `@jax.jit` decorator, which behaves identically to the `@tf.function(jit_compile=True)` decorator. It turns any stateless JAX function into an XLA-compiled piece of code, typically delivering a considerable execution speedup:

Be mindful that you can only decorate a stateless function—any tensors that get updated by the function should be part of its return values.

3.5.4

An end-to-end example: A linear classifier in pure JAX

Now you know enough JAX to write the JAX version of our linear classifier example. There are two major differences from the TensorFlow and PyTorch versions you’ve already seen:

All functions we will create will be stateless. That means the state (the arrays `W` and `b`) will be provided as function arguments, and if they get modified by the function, their new value will be returned by the function.

Gradients are computed using the JAX `value_and_grad()` utility.

Let’s get started. The model function and the mean squared error function should look familiar:

## Next page (PDF 114) text, for continuity
That’s it! You’re now able to write a custom training loop in JAX.

3.5.5

What makes the JAX approach unique

The main thing that makes JAX unique among modern machine learning frameworks is its functional, stateless philosophy. While it may seem to cause friction at first, it is what unlocks the power of JAX—its ability to compile to extremely fast code and to scale to arbitrarily large models and arbitrarily many devices.

There’s a lot to like about JAX:

It’s fast. For most models, it is the fastest of all frameworks you’ve seen so far.

Its numerical API is fully consistent with NumPy, making it pleasant to learn.

It’s the best fit for training models on TPUs, as it was developed from the ground up for XLA and TPUs.

Using JAX can also come with some amount of developer friction:

Its use of metaprogramming and compilation can make it significantly harder to debug compared to pure eager execution.

Low-level training loops tend to be more verbose and more difficult to write than in TensorFlow or PyTorch.

At this point, you know the basics of TensorFlow, PyTorch, and JAX, and you can use these frameworks to implement a basic linear classifier from scratch. That’s a solid foundation to build upon. It’s now time to move on to a more productive path to deep learning: the Keras API.

3.6

Introduction to Keras

Keras is a deep learning API for Python that provides a convenient way to define and train any kind of deep learning model. It was released in March 2015, with its v2 in 2017 and its v3 in 2023.

Keras users range from academic researchers, engineers, and data scientists at both startups and large companies to graduate students and hobbyists. Keras is used at Google, Netflix, Uber, YouTube, CERN, NASA, Yelp, Instacart, Square, Waymo, YouTube, and thousands of  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.111 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.110 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.104 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.95 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.102 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.253 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.110 (printed 86) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Computing gradients with JAX
Unlike TensorFlow and PyTorch, JAX takes a metaprogramming approach to gradient computation. Metaprogramming refers to the idea of having functions that return functions—you could call them “meta-functions.” In practice, JAX lets you turn a loss-computation function into a gradient-computation function. So computing gradients in JAX is a three-step process:

1 Define a loss function, `compute_loss()`. 2 Call `grad_fn = jax.grad(compute_loss)` to retrieve a gradient-computation

function. 3 Call `grad_fn` to retrieve the gradient values. The loss function should verify the following properties:

It should return a scalar loss value.

Its first argument (which, in the following example, is also the only argument) should contain the state arrays we need gradients for. This argument is usually named `state`. For instance, this first argument could be a single array, a list of arrays, or a dict of arrays.

Let’s take a look at a simple example. Here’s a loss-computation function that takes a single scalar, `input_var` and returns a scalar loss value—just the square of the input:

### PDF p.111 (printed 87) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Getting gradients for a complex function
def compute_loss(state, x, y):
    ...
    return loss
grad_fn = jax.value_and_grad(compute_loss)
state = (a, b, c)
loss, grads_of_loss_wrt_state = grad_fn(state, x, y)

### PDF p.110 (printed 86) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Computing gradients with JAX
Once you’ve obtained `grad_fn()`, you can call it with the same arguments as `compute_` `loss()`, and it will return gradients arrays corresponding to the first argument of `compute_loss()`. In our case, our first argument was a single array, so `grad_fn()` directly returns the gradient of the loss with respect to that one array:

### PDF p.242 (printed 218) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A JAX training step function
When it comes to low-level training code, JAX tends to be the most complex of the three backends because of its fully stateless nature. Statelessness makes JAX highly performant and scalable, making it amenable to compilation and automatic performance optimizations. However, writing stateless code requires you to jump through some hoops.

Since the gradient function is obtained via metaprogramming, you first need to define the function that returns your loss. Further, this function needs to be stateless, so it needs to take as arguments all the variables it’s going to be using, and it needs to return the value of any variable it has updated. Remember those non-trainable weights that can get modified during the forward pass? Those are the variables we need to return. To make it easier to work with the stateless programming paradigm of JAX, Keras models make available a stateless forward pass method: the `stateless_call()` method. It behaves just like `call`, except that

It takes as input the model’s trainable weights and non-trainable weights, in addition to the `inputs` and `training` arguments.

### PDF p.111 (printed 87) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Getting gradients for a complex function
Now, what if you need gradients for more than a single variable? And what if your `compute_loss()` function has more than one input?

Let’s say your state contains three variables, `a`, `b`, and `c`, and your loss function has two inputs, `x` and `y`. You would simply structure it like this:

### PDF p.327 (printed 303) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3.3 Getting the gradient of the top class: JAX version
Finally, let’s do JAX. We define a separate loss computation function that takes the final layer’s output and returns the activation channel corresponding to the top predicted class. We use this activation value as our loss, allowing us to compute the gradient.

Listing 10.22 Computing the top class gradients with Jax

### PDF p.102 (printed 78) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › An end-to-end example: A linear classifier in pure PyTorch
You now know enough to rewrite our linear classifier in PyTorch. It will stay very similar to the TensorFlow one—the only major difference is how we compute the gradients.

Let’s start by creating our model variables. Don’t forget to pass `requires_grad=True` so we can compute gradients with respect to them:

### PDF p.103 (printed 79) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
1 `loss.backward()` runs backpropagation starting from the `loss` output node and

populates the `tensor.grad` attribute on all tensors that were involved in the computation of `loss`. `tensor.grad` represents the gradient of the loss with regard to that tensor. 2 We use the `.grad` attribute to recover the gradients of the loss with regard to `W` and `b`. 3 We update `W` and `b` using those gradients. Because these updates are not intended to be part of the backward pass, we do them inside a `torch.no_grad()` scope, which skips gradient computation for everything inside it. 4 We reset the contents of the `.grad` property of our `W` and `b` parameters, by setting it to `None`. If we didn’t do this, gradient values would accumulate across multiple calls to `training_step()`, resulting in invalid values:
