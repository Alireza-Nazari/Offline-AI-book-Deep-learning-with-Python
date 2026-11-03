# Generation bundle for page_319  (PDF page 319, printed page 295)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.2 Visualizing ConvNet filters
Subsection: 10.2.3 Gradient ascent in JAX
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_319.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_319.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p319_e03] (code)
```
        loss = compute_loss(image, filter_index)
    grads = tape.gradient(loss, image)
    grads = ops.normalize(grads)
    image += learning_rate * grads
    return image
```
[p319_e04] (heading_subsection) 10.2.2 Gradient ascent in PyTorch
[p319_e05] (body) In the case of PyTorch, we use `loss.backward()` and `image.grad` to obtain the gradients of the loss with respect to the input image, like this.
[p319_e06] (listing_caption) Listing 10.12 Loss maximization via stochastic gradient ascent: PyTorch
[p319_e07] (code)
```
import torch
def gradient_ascent_step(image, filter_index, learning_rate):
    image = image.clone().detach().requires_grad_(True)
    loss = compute_loss(image, filter_index)
    loss.backward()
    grads = image.grad
    grads = ops.normalize(grads)
    image = image + learning_rate * grads
    return image
```
[p319_e08] (body) No need to reset the gradients since the image tensor is recreated at each iteration.
[p319_e09] (heading_subsection) 10.2.3 Gradient ascent in JAX
[p319_e10] (body) In the case of JAX, we use `jax.grad()` to obtain a function that returns the gradients of the loss with respect to the input image.
[p319_e11] (listing_caption) Listing 10.13 Loss maximization via stochastic gradient ascent: JAX
[p319_e12] (code)
```
import jax
grad_fn = jax.grad(compute_loss)
@jax.jit
def gradient_ascent_step(image, filter_index, learning_rate):
    grads = grad_fn(image, filter_index)
    grads = ops.normalize(grads)
    image += learning_rate * grads
    return image
```

## Code annotations printed next to listings (callouts, in order)
- Computes the loss
- scalar, indicating how
- much the current image
- activates the filter
- Computes the gradients of the
- loss with respect to the image
- Returns the updated image, so we
- can run the step function in a loop
- Applies the “gradient
- normalization trick”
- Moves the image a little bit in a
- direction that activates our
- target filter more strongly
- Creates a copy of
- "image" that we can get
- gradients for

## Listings
- Listing 10.12: Loss maximization via stochastic gradient ascent: PyTorch (code elements p319_e07)
- Listing 10.13: Loss maximization via stochastic gradient ascent: JAX (code elements p319_e12)

## Chapter outline
Chapter: 10 Interpreting what ConvNets learn (PDF pages 308-331)
  - 10.1 Visualizing intermediate activations (p.309)
  - 10.2 Visualizing ConvNet filters (p.315)
      - 10.2.1 Gradient ascent in TensorFlow (p.318)
      - 10.2.2 Gradient ascent in PyTorch (p.319)
      - 10.2.3 Gradient ascent in JAX (p.319)
      - 10.2.4 The filter visualization loop (p.320)
  - 10.3 Visualizing heatmaps of class activation (p.323)
      - 10.3.1 Getting the gradient of the top class: TensorFlow version (p.326)
      - 10.3.2 Getting the gradient of the top class: PyTorch version (p.326)
      - 10.3.3 Getting the gradient of the top class: JAX version (p.327)
      - 10.3.4 Displaying the class activation heatmap (p.328)
  - 10.4 Visualizing the latent space of a ConvNet (p.330)

## Previous page (PDF 318) text, for continuity
(continued)

You should use `model(x)` when you need to retrieve the gradients of the model call. And you should use `predict()` if you just need the output value. In other words, always use `predict()`, unless you’re in the middle of writing a low-level gradient descent loop (as we are now).

A non-obvious trick to help the gradient-ascent process go smoothly is to normalize the gradient tensor by dividing it by its L2 norm (the square root of the sum of the squares of the values in the tensor). This ensures that the magnitude of the updates done to the input image is always within the same range.

Let’s set up the gradient ascent step function. Anything that involves gradients requires calling backend-level APIs, such as `GradientTape` in TensorFlow, `.backward()` in PyTorch, and `jax.grad()` in JAX. Let’s line up all the code snippets for each of the three backends, starting with TensorFlow.

10.2.1 Gradient ascent in TensorFlow

For TensorFlow, we can just open a `GradientTape` scope and compute the loss inside of it to retrieve the gradients we need. We’ll use a `@tf.function` decorator to speed up computation:

Listing 10.11 Loss maximization via stochastic gradient ascent: TensorFlow

## Next page (PDF 320) text, for continuity
10.2.4 The filter visualization loop

Now you have all the pieces. Let’s put them together into a Python function that takes a filter index as input and returns a tensor representing the pattern that maximizes the activation of the specified filter in our target layer.

Listing 10.14 Function to generate filter visualizations

The resulting image tensor is a floating-point array of shape `(200, 200, 3)`, with values that may not be integers within `[0, 255]`. Hence, you need to post-process this tensor to turn it into a displayable image. You do so with the following straightforward utility function.

Listing 10.15 Utility function to convert a tensor into a valid image

Let’s try it (see figure 10.5):

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.317 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.113 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.326 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.327 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.110 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.111 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.315 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.110 (printed 86) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Computing gradients with JAX
We can now call the JAX utility `jax.grad()` on this loss function. It returns a gradient-computation function—a function that takes the same arguments as the original loss function and returns the gradient of the loss with respect to `input_var`:

### PDF p.110 (printed 86) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Computing gradients with JAX
Once you’ve obtained `grad_fn()`, you can call it with the same arguments as `compute_` `loss()`, and it will return gradients arrays corresponding to the first argument of `compute_loss()`. In our case, our first argument was a single array, so `grad_fn()` directly returns the gradient of the loss with respect to that one array:

### PDF p.111 (printed 87) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Returning the loss value
It’s usually the case that you don’t just need the gradient array; you also need the loss value. It would be quite inefficient to recompute it independently outside of `grad_` `fn()`, so instead, you can just configure your `grad_fn()` to also return the loss value. This is done by using the JAX utility `jax.value_and_grad()` instead of `jax.grad()`. It works identically, but it returns a tuple of values, where the first entry is the loss value, and the second entry is the gradient(s):

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Computing gradients with PyTorch
Listing 3.26 Computing a gradient with `.backward()`
>>> input_var = torch.tensor(3.0, requires_grad=True)
>>> result = torch.square(input_var)
>>> result.backward()
>>> gradient = input_var.grad
>>> gradient
tensor(6.)

### PDF p.103 (printed 79) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
1 `loss.backward()` runs backpropagation starting from the `loss` output node and

populates the `tensor.grad` attribute on all tensors that were involved in the computation of `loss`. `tensor.grad` represents the gradient of the loss with regard to that tensor. 2 We use the `.grad` attribute to recover the gradients of the loss with regard to `W` and `b`. 3 We update `W` and `b` using those gradients. Because these updates are not intended to be part of the backward pass, we do them inside a `torch.no_grad()` scope, which skips gradient computation for everything inside it. 4 We reset the contents of the `.grad` property of our `W` and `b` parameters, by setting it to `None`. If we didn’t do this, gradient values would accumulate across multiple calls to `training_step()`, resulting in invalid values:

### PDF p.315 (printed 291) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2 Visualizing ConvNet filters
Another easy way to inspect the filters learned by ConvNets is to display the visual pattern that each filter is meant to respond to. This can be done with gradient ascent in input space, applying gradient descent to the value of the input image of a ConvNet so as to maximize the response of a specific filter, starting from a blank input image. The resulting input image will be one that the chosen filter is maximally responsive to.

Let’s try this with the filters of the Xception model. The process is simple: we’ll build a loss function that maximizes the value of a given filter in a given convolution layer, and then we’ll use stochastic gradient descent to adjust the values of the input image so as to maximize this activation value. This will be your second example of a low-level

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Computing gradients with PyTorch
If you call `backward()` multiple times in a row, the `.grad` attribute will “accumulate” gradients: each new call will sum the new gradient with the preexisting one. For instance,

### PDF p.113 (printed 89) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
Calling `jax.value_and_grad()` on this function gives us a new function, with the same argument as `compute_loss`, which returns both the loss and the gradients of the loss with regard to the elements of `state`:
