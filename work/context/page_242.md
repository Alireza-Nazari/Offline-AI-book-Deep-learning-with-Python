# Generation bundle for page_242  (PDF page 242, printed page 218)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.2 Writing custom training step functions
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_242.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_242.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p242_e03] (body) Let’s chain these steps:
[p242_e04] (code)
```
import torch
model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()
def train_step(inputs, targets):
    predictions = model(inputs, training=True)
    loss = loss_fn(targets, predictions)
    loss.backward()
    gradients = [weight.value.grad for weight in model.trainable_weights]
    with torch.no_grad():
        optimizer.apply(gradients, model.trainable_weights)
    model.zero_grad()
    return loss
```
[p242_e05] (body) Let’s run it for a single step:
[p242_e06] (code)
```
batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)
```
[p242_e07] (body) That wasn’t too difficult! Now, let’s move on to JAX.
[p242_e08] (heading_subsubsection) A JAX training step function
[p242_e09] (body) When it comes to low-level training code, JAX tends to be the most complex of the three backends because of its fully stateless nature. Statelessness makes JAX highly performant and scalable, making it amenable to compilation and automatic performance optimizations. However, writing stateless code requires you to jump through some hoops.
[p242_e10] (body) Since the gradient function is obtained via metaprogramming, you first need to define the function that returns your loss. Further, this function needs to be stateless, so it needs to take as arguments all the variables it’s going to be using, and it needs to return the value of any variable it has updated. Remember those non-trainable weights that can get modified during the forward pass? Those are the variables we need to return. To make it easier to work with the stateless programming paradigm of JAX, Keras models make available a stateless forward pass method: the `stateless_call()` method. It behaves just like `call`, except that
[p242_e11] (bullet) It takes as input the model’s trainable weights and non-trainable weights, in addition to the `inputs` and `training` arguments.

## Code annotations printed next to listings (callouts, in order)
- Runs the backward pass,
- populating gradient values
- Runs the forward pass
- Updates the model’s
- Don’t forget to
- trainable weights based on
- clear the gradients!
- the gradients. This must be
- done in a no_grad() scope.
- Recovers the gradient associated with each
- trainable variable. That weight.value is the
- PyTorch tensor that contains the variable’s value.

## Equation candidates (formulas written inline in the text)
- [p242_e06] batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)

## Chapter outline
Chapter: 7 A deep dive on Keras (PDF pages 214-254)
  - 7.1 A spectrum of workflows (p.215)
  - 7.2 Different ways to build Keras models (p.216)
      - 7.2.1 The Sequential model (p.216)
      - 7.2.2 The Functional API (p.219)
      - 7.2.3 Subclassing the Model class (p.226)
      - 7.2.4 Mixing and matching different components (p.228)
      - 7.2.5 Remember: Use the right tool for the job (p.229)
  - 7.3 Using built-in training and evaluation loops (p.230)
      - 7.3.1 Writing your own metrics (p.231)
      - 7.3.2 Using callbacks (p.232)
      - 7.3.3 Writing your own callbacks (p.234)
      - 7.3.4 Monitoring and visualization with TensorBoard (p.236)
  - 7.4 Writing your own training and evaluation loops (p.238)
      - 7.4.1 Training vs. inference (p.239)
      - 7.4.2 Writing custom training step functions (p.240)
      - 7.4.3 Low-level usage of metrics (p.245)
      - 7.4.4 Using fit() with a custom training loop (p.246)
      - 7.4.5 Handling metrics in a custom train_step() (p.250)

## Previous page (PDF 241) text, for continuity
Let’s run it for a single step:

Easy enough! Let’s do PyTorch next.

A PyTorch training step function

When you use the PyTorch backend, all of your Keras layers and models inherit from the PyTorch `torch.nn.Module` class and expose the native `Module` API. As a result, your model, its trainable weights, and your loss tensor are all aware of each other and interact via three methods: `loss.backward()`, `weight.value.grad`, and `model
` `.zero_grad()`.

As a reminder from chapter 3, the mental model you’ve got to keep in mind is this:

With each forward pass, PyTorch builds up a one-time computation graph that keeps track of the computation that just happened.

Calling `.backward()` on any given scalar node of this graph (like your loss) will run the graph backward starting from that node, automatically populating a `tensor.grad` attribute on all tensors involved (if they satisfy `requires_` `grad=True`), containing the gradient of the output node with respect to that tensor. In particular, it will populate the `grad` attribute of your trainable parameters.

To clear the contents of that `tensor.grad` attribute, you should call `tensor.grad ` `= None` on all your tensors. Because it would be a bit cumbersome to do this on all model variables individually, you can just do it at the model level via `model
` `.zero_grad()`—the `zero_grad()` call will propagate to all variables tracked by the model. Clearing gradients is critical because calls to `backward()` are additive: if you don’t clear the gradients at each step, the gradient values will accumulate and training won’t proceed.

## Next page (PDF 243) text, for continuity
It returns the model’s updated non-trainable weights, in addition to the model’s outputs.

It works like this:

We can use `stateless_call()` to implement our JAX loss function. Since the loss function also computes updates for all non-trainable variables, we name it `compute_loss_and_updates()`:

Once we have this `compute_loss_and_updates()` function, we can pass it to `jax.value_` `and_grad` to obtain the gradient computation:

Now, there’s just a small problem. Both `jax.grad()` and `jax.value_and_grad()` require `fn` to return a scalar value only. Our `compute_loss_and_updates()` function returns a scalar value as its first output, but it also returns the new value for the non-trainable weights. Remember what you learned in chapter 3? The solution is to pass a `has_aux` argument to `grad()` or `value_and_grad()`, like this:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.244 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.240 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.250 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.253 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.112 (printed 88) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Making JAX functions fast with @jax.jit
Be mindful that you can only decorate a stateless function—any tensors that get updated by the function should be part of its return values.

### PDF p.112 (printed 88) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › An end-to-end example: A linear classifier in pure JAX
Now you know enough JAX to write the JAX version of our linear classifier example. There are two major differences from the TensorFlow and PyTorch versions you’ve already seen:

All functions we will create will be stateless. That means the state (the arrays `W` and `b`) will be provided as function arguments, and if they get modified by the function, their new value will be returned by the function.

Gradients are computed using the JAX `value_and_grad()` utility.

Let’s get started. The model function and the mean squared error function should look familiar:

### PDF p.110 (printed 86) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Computing gradients with JAX
Unlike TensorFlow and PyTorch, JAX takes a metaprogramming approach to gradient computation. Metaprogramming refers to the idea of having functions that return functions—you could call them “meta-functions.” In practice, JAX lets you turn a loss-computation function into a gradient-computation function. So computing gradients in JAX is a three-step process:

1 Define a loss function, `compute_loss()`. 2 Call `grad_fn = jax.grad(compute_loss)` to retrieve a gradient-computation

function. 3 Call `grad_fn` to retrieve the gradient values. The loss function should verify the following properties:

It should return a scalar loss value.

Its first argument (which, in the following example, is also the only argument) should contain the state arrays we need gradients for. This argument is usually named `state`. For instance, this first argument could be a single array, a list of arrays, or a dict of arrays.

Let’s take a look at a simple example. Here’s a loss-computation function that takes a single scalar, `input_var` and returns a scalar loss value—just the square of the input:

### PDF p.106 (printed 82) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Introduction to JAX
JAX is an open source library for differentiable computation, primarily developed by Google. After its release in 2018, JAX quickly gained traction in the research community, particularly for its ability to use Google’s TPUs at scale. Today, JAX is in use by most of the top players in the generative AI space—companies like DeepMind, Apple, Midjourney, Anthropic, Cohere, and so on.

JAX embraces a stateless approach to computation, meaning that functions in JAX do not maintain any persistent state. This contrasts with traditional imperative programming, where variables can hold values between function calls. The stateless nature of JAX functions has several advantages. In particular, it enables effective automatic parallelization and distributed computation, as functions can be executed independently without the need for synchronization. The extreme scalability of JAX is essential for handling the very large-scale machine learning problems faced by companies like Google and DeepMind.

### PDF p.113 (printed 89) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
Next, we can set up our training step function. It looks straightforward. Be mindful that, unlike its TensorFlow and PyTorch equivalents, it needs to be stateless, and so it must return the updated values of the `W` and `b` tensors:

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing custom training step functions
This snippet is pseudocode rather than real code because it includes an imaginary function, `get_gradients_of()`. In reality, retrieving gradients is done in a way that is specific to your current backend—JAX, TensorFlow, or PyTorch.

Let’s use what you learned about each framework in chapter 3 to implement a real version of this `train_step()` function. We’ll start with TensorFlow and PyTorch because these two make the job relatively easy, so they’re a good place to start. We’ll end with JAX, which is quite a bit more complex.

### PDF p.249 (printed 225) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Customizing fit() with JAX
Finally, let’s write the JAX version. First we need to define a `compute_loss_and_` `updates()` method, similar to the `compute_loss_and_updates()` function we used in our custom training step example:

### PDF p.114 (printed 90) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › What makes the JAX approach unique
The main thing that makes JAX unique among modern machine learning frameworks is its functional, stateless philosophy. While it may seem to cause friction at first, it is what unlocks the power of JAX—its ability to compile to extremely fast code and to scale to arbitrarily large models and arbitrarily many devices.

There’s a lot to like about JAX:

It’s fast. For most models, it is the fastest of all frameworks you’ve seen so far.

Its numerical API is fully consistent with NumPy, making it pleasant to learn.

It’s the best fit for training models on TPUs, as it was developed from the ground up for XLA and TPUs.

Using JAX can also come with some amount of developer friction:

Its use of metaprogramming and compilation can make it significantly harder to debug compared to pure eager execution.

Low-level training loops tend to be more verbose and more difficult to write than in TensorFlow or PyTorch.

At this point, you know the basics of TensorFlow, PyTorch, and JAX, and you can use these frameworks to implement a basic linear classifier from scratch. That’s a solid foundation to build upon. It’s now time to move on to a more productive path to deep learning: the Keras API.
