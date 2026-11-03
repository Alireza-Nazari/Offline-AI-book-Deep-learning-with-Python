# Generation bundle for page_111  (PDF page 111, printed page 87)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.5 Introduction to JAX
Subsection: 3.5.3 Random number generation in JAX
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_111.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_111.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p111_e03] (code)
```
input_var = jnp.array(3.0)
grad_of_loss_wrt_input_var = grad_fn(input_var)
```
[p111_e04] (heading_subsubsection) JAX gradient-computation best practices
[p111_e05] (body) So far so good! Metaprogramming is a big word, but it turns out to be quite simple. Now, in real-world use cases, there are a few more things you’ll need to take into account. Let’s take a look.
[p111_e06] (heading_subsubsection) Returning the loss value
[p111_e07] (body) It’s usually the case that you don’t just need the gradient array; you also need the loss value. It would be quite inefficient to recompute it independently outside of `grad_` `fn()`, so instead, you can just configure your `grad_fn()` to also return the loss value. This is done by using the JAX utility `jax.value_and_grad()` instead of `jax.grad()`. It works identically, but it returns a tuple of values, where the first entry is the loss value, and the second entry is the gradient(s):
[p111_e08] (code)
```
grad_fn = jax.value_and_grad(compute_loss)
output, grad_of_loss_wrt_input_var = grad_fn(input_var)
```
[p111_e09] (heading_subsubsection) Getting gradients for a complex function
[p111_e10] (body) Now, what if you need gradients for more than a single variable? And what if your `compute_loss()` function has more than one input?
[p111_e11] (body) Let’s say your state contains three variables, `a`, `b`, and `c`, and your loss function has two inputs, `x` and `y`. You would simply structure it like this:
[p111_e12] (code)
```
def compute_loss(state, x, y):
    ...
    return loss
grad_fn = jax.value_and_grad(compute_loss)
state = (a, b, c)
loss, grads_of_loss_wrt_state = grad_fn(state, x, y)
```
[p111_e13] (body) Note that `state` doesn’t have to be a tuple—it could be a dict, a list, or any nested structure of tuples, dicts, and lists. In JAX parlance, such a nested structure is called a tree.
[p111_e14] (heading_subsubsection) Returning auxiliary outputs
[p111_e15] (body) Finally, what if your `compute_loss()` function needs to return more than just the loss? Let’s say you want to return an additional value `output` that’s computed as a by-product of the loss computation. How to get it out?
[p111_e16] (body) You would use the `has_aux` argument:

## Code annotations printed next to listings (callouts, in order)
- state contains a, b, and c. It
- must be the first argument.
- grads_of_loss_wrt_
- state has the same
- structure as state.

## Equation candidates (formulas written inline in the text)
- [p111_e03] input_var = jnp.array(3.0)
grad_of_loss_wrt_input_var = grad_fn(input_var)
- [p111_e08] grad_fn = jax.value_and_grad(compute_loss)
output, grad_of_loss_wrt_input_var = grad_fn(input_var)

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

## Previous page (PDF 110) text, for continuity
Here’s a dense layer:

Computing gradients with JAX

Unlike TensorFlow and PyTorch, JAX takes a metaprogramming approach to gradient computation. Metaprogramming refers to the idea of having functions that return functions—you could call them “meta-functions.” In practice, JAX lets you turn a loss-computation function into a gradient-computation function. So computing gradients in JAX is a three-step process:

1 Define a loss function, `compute_loss()`. 2 Call `grad_fn = jax.grad(compute_loss)` to retrieve a gradient-computation

function. 3 Call `grad_fn` to retrieve the gradient values. The loss function should verify the following properties:

It should return a scalar loss value.

Its first argument (which, in the following example, is also the only argument) should contain the state arrays we need gradients for. This argument is usually named `state`. For instance, this first argument could be a single array, a list of arrays, or a dict of arrays.

Let’s take a look at a simple example. Here’s a loss-computation function that takes a single scalar, `input_var` and returns a scalar loss value—just the square of the input:

We can now call the JAX utility `jax.grad()` on this loss function. It returns a gradient-computation function—a function that takes the same arguments as the original loss function and returns the gradient of the loss with respect to `input_var`:

Once you’ve obtained `grad_fn()`, you can call it with the same arguments as `compute_` `loss()`, and it will return gradients arrays corresponding to the first argument of `compute_loss()`. In our case, our first argument was a single array, so `grad_fn()` directly returns the gradient of the loss with respect to that one array:

## Next page (PDF 112) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.113 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.243 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.102 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.107 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.253 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.249 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.327 (printed 303) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3.3 Getting the gradient of the top class: JAX version
Finally, let’s do JAX. We define a separate loss computation function that takes the final layer’s output and returns the activation channel corresponding to the top predicted class. We use this activation value as our loss, allowing us to compute the gradient.

Listing 10.22 Computing the top class gradients with Jax

### PDF p.319 (printed 295) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2.3 Gradient ascent in JAX
In the case of JAX, we use `jax.grad()` to obtain a function that returns the gradients of the loss with respect to the input image.

Listing 10.13 Loss maximization via stochastic gradient ascent: JAX

### PDF p.113 (printed 89) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
Calling `jax.value_and_grad()` on this function gives us a new function, with the same argument as `compute_loss`, which returns both the loss and the gradients of the loss with regard to the elements of `state`:

### PDF p.242 (printed 218) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A JAX training step function
When it comes to low-level training code, JAX tends to be the most complex of the three backends because of its fully stateless nature. Statelessness makes JAX highly performant and scalable, making it amenable to compilation and automatic performance optimizations. However, writing stateless code requires you to jump through some hoops.

Since the gradient function is obtained via metaprogramming, you first need to define the function that returns your loss. Further, this function needs to be stateless, so it needs to take as arguments all the variables it’s going to be using, and it needs to return the value of any variable it has updated. Remember those non-trainable weights that can get modified during the forward pass? Those are the variables we need to return. To make it easier to work with the stateless programming paradigm of JAX, Keras models make available a stateless forward pass method: the `stateless_call()` method. It behaves just like `call`, except that

It takes as input the model’s trainable weights and non-trainable weights, in addition to the `inputs` and `training` arguments.

### PDF p.243 (printed 219) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Now, there’s just a small problem. Both `jax.grad()` and `jax.value_and_grad()` require `fn` to return a scalar value only. Our `compute_loss_and_updates()` function returns a scalar value as its first output, but it also returns the new value for the non-trainable weights. Remember what you learned in chapter 3? The solution is to pass a `has_aux` argument to `grad()` or `value_and_grad()`, like this:

### PDF p.113 (printed 89) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
To compute gradients, we need to package loss computation in a single `compute_loss()` function. It returns the total loss as a scalar, and it takes `state` as its first argument—a tuple of all the tensors we need gradients for:

### PDF p.319 (printed 295) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2.3 Gradient ascent in JAX
Listing 10.13 Loss maximization via stochastic gradient ascent: JAX
import jax
grad_fn = jax.grad(compute_loss)
@jax.jit
def gradient_ascent_step(image, filter_index, learning_rate):
    grads = grad_fn(image, filter_index)
    grads = ops.normalize(grads)
    image += learning_rate * grads
    return image

### PDF p.243 (printed 219) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
import jax
grad_fn = jax.value_and_grad(fn)
loss, gradients = grad_fn(...)
