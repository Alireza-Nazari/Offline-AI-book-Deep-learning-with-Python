# Generation bundle for page_243  (PDF page 243, printed page 219)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.2 Writing custom training step functions
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_243.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_243.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p243_e03] (bullet) It returns the model’s updated non-trainable weights, in addition to the model’s outputs.
[p243_e04] (body) It works like this:
[p243_e05] (code)
```
outputs, non_trainable_weights = model.stateless_call(
    trainable_weights, non_trainable_weights, inputs
)
```
[p243_e06] (body) We can use `stateless_call()` to implement our JAX loss function. Since the loss function also computes updates for all non-trainable variables, we name it `compute_loss_and_updates()`:
[p243_e07] (code)
```
model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, inputs, targets
):
    outputs, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, inputs, training=True
    )
    loss = loss_fn(targets, outputs)
    return loss, non_trainable_variables
```
[p243_e08] (body) Once we have this `compute_loss_and_updates()` function, we can pass it to `jax.value_` `and_grad` to obtain the gradient computation:
[p243_e09] (code)
```
import jax
grad_fn = jax.value_and_grad(fn)
loss, gradients = grad_fn(...)
```
[p243_e10] (body) Now, there’s just a small problem. Both `jax.grad()` and `jax.value_and_grad()` require `fn` to return a scalar value only. Our `compute_loss_and_updates()` function returns a scalar value as its first output, but it also returns the new value for the non-trainable weights. Remember what you learned in chapter 3? The solution is to pass a `has_aux` argument to `grad()` or `value_and_grad()`, like this:
[p243_e11] (code)
```
import jax
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
```

## Code annotations printed next to listings (callouts, in order)
- Gradients are computed
- for the entries in the first
- argument (trainable_
- variables here)
- Calls stateless_call
- Returns the scalar loss value and
- the updated non-trainable weights

## Equation candidates (formulas written inline in the text)
- [p243_e05] outputs, non_trainable_weights = model.stateless_call(
    trainable_weights, non_trainable_weights, inputs
)

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

## Previous page (PDF 242) text, for continuity
Let’s chain these steps:

Let’s run it for a single step:

That wasn’t too difficult! Now, let’s move on to JAX.

A JAX training step function

When it comes to low-level training code, JAX tends to be the most complex of the three backends because of its fully stateless nature. Statelessness makes JAX highly performant and scalable, making it amenable to compilation and automatic performance optimizations. However, writing stateless code requires you to jump through some hoops.

Since the gradient function is obtained via metaprogramming, you first need to define the function that returns your loss. Further, this function needs to be stateless, so it needs to take as arguments all the variables it’s going to be using, and it needs to return the value of any variable it has updated. Remember those non-trainable weights that can get modified during the forward pass? Those are the variables we need to return. To make it easier to work with the stateless programming paradigm of JAX, Keras models make available a stateless forward pass method: the `stateless_call()` method. It behaves just like `call`, except that

It takes as input the model’s trainable weights and non-trainable weights, in addition to the `inputs` and `training` arguments.

## Next page (PDF 244) text, for continuity
You would use it like this:

Okay, that was a lot of JAXiness. But now we’ve got almost everything we need to assemble our JAX training step. We just need the last piece of the puzzle: `optimizer.apply()`.

When you wrote your first basic training step in TensorFlow at the beginning of chapter 2, you wrote an update step function that looked like this:

This corresponds to what the optimizer `keras.optimizers.SGD` would do. However, every other optimizer in the Keras API is somewhat more complex than that and keeps track of auxiliary variables that help speed up training—in particular, most optimizers use some form of momentum, which you learned about in chapter 2. These extra variables get updated at each step of training, and in the JAX world, that means that you need to get your hands on a stateless function that takes these variables as arguments and returns their new value.

To make this easy, Keras makes available the `stateless_apply()` method on all optimizers. It works like this:

Now, we have enough to assemble an end-to-end training step:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.253 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.111 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.249 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.110 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.113 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.112 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.110 (printed 86) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Computing gradients with JAX
Unlike TensorFlow and PyTorch, JAX takes a metaprogramming approach to gradient computation. Metaprogramming refers to the idea of having functions that return functions—you could call them “meta-functions.” In practice, JAX lets you turn a loss-computation function into a gradient-computation function. So computing gradients in JAX is a three-step process:

1 Define a loss function, `compute_loss()`. 2 Call `grad_fn = jax.grad(compute_loss)` to retrieve a gradient-computation

function. 3 Call `grad_fn` to retrieve the gradient values. The loss function should verify the following properties:

It should return a scalar loss value.

Its first argument (which, in the following example, is also the only argument) should contain the state arrays we need gradients for. This argument is usually named `state`. For instance, this first argument could be a single array, a list of arrays, or a dict of arrays.

Let’s take a look at a simple example. Here’s a loss-computation function that takes a single scalar, `input_var` and returns a scalar loss value—just the square of the input:

### PDF p.110 (printed 86) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Computing gradients with JAX
We can now call the JAX utility `jax.grad()` on this loss function. It returns a gradient-computation function—a function that takes the same arguments as the original loss function and returns the gradient of the loss with respect to `input_var`:

### PDF p.113 (printed 89) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
Calling `jax.value_and_grad()` on this function gives us a new function, with the same argument as `compute_loss`, which returns both the loss and the gradients of the loss with regard to the elements of `state`:

### PDF p.111 (printed 87) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Returning the loss value
It’s usually the case that you don’t just need the gradient array; you also need the loss value. It would be quite inefficient to recompute it independently outside of `grad_` `fn()`, so instead, you can just configure your `grad_fn()` to also return the loss value. This is done by using the JAX utility `jax.value_and_grad()` instead of `jax.grad()`. It works identically, but it returns a tuple of values, where the first entry is the loss value, and the second entry is the gradient(s):

### PDF p.249 (printed 225) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Customizing fit() with JAX
Finally, let’s write the JAX version. First we need to define a `compute_loss_and_` `updates()` method, similar to the `compute_loss_and_updates()` function we used in our custom training step example:

### PDF p.319 (printed 295) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2.3 Gradient ascent in JAX
In the case of JAX, we use `jax.grad()` to obtain a function that returns the gradients of the loss with respect to the input image.

Listing 10.13 Loss maximization via stochastic gradient ascent: JAX

### PDF p.112 (printed 88) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
1 Edit the loss function to return a tuple where the first entry is the loss, and the

second entry is your extra output. 2 Pass the argument `has_aux=True` to `value_and_grad()`. This tells `value_and_` `grad()` to return not just the gradient but also the “auxiliary” output(s) of `compute_loss()`, like this:

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Training vs. inference
In the low-level training loop examples you’ve seen so far, step 1 (the forward pass) was done via `predictions = model(inputs)`, and step 2 (retrieving the gradients computed by the gradient tape) was done via a backend-specific API, such as

`gradients = tape.gradient(loss, model.weights)` in TensorFlow

`loss.backward()` in PyTorch

`jax.value_and_grad()` in JAX

In the general case, there are actually two subtleties you need to take into account.

Some Keras layers, such as the `Dropout` layer, have different behaviors during training and during inference (when you use them to generate predictions). Such layers expose a `training` Boolean argument in their `call()` method. Calling `dropout(inputs, ` `training=True)` will drop some activation entries, while calling `dropout(inputs, ` `training=False)` does nothing. By extension, Functional models and Sequential models also expose this `training` argument in their `call()` methods. Remember to pass `training=True` when you call a Keras model during the forward pass! Our forward pass thus becomes `predictions = model(inputs, training=True)`. In addition, note that when you retrieve the gradients of the weights of your model, you s
