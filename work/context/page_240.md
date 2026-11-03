# Generation bundle for page_240  (PDF page 240, printed page 216)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.2 Writing custom training step functions
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_240.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_240.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p240_e03] (body) stored in a non-trainable weight, and at each batch, your layer would increment the counter by one. Among Keras built-in layers, the only layer that features non-trainable weights is the `BatchNormalization` layer, which we will introduce in chapter 9. The `BatchNormalization` layer needs non-trainable weights to track information about the mean and standard deviation of the data that passes through it, so as to perform an online approximation of feature normalization (a concept you’ve learned about in chapters 4 and 6).
[p240_e04] (heading_subsection) 7.4.2
[p240_e05] (heading_subsection) Writing custom training step functions
[p240_e06] (body) Taking into account these two details, a supervised learning training step ends up looking like this in pseudocode:
[p240_e07] (code)
```
def train_step(inputs, targets):
    predictions = model(inputs, training=True)
    loss = loss_fn(targets, predictions)
    gradients = get_gradients_of(loss, wrt=model.trainable_weights)
    optimizer.apply(gradients, model.trainable_weights)
```
[p240_e08] (body) This snippet is pseudocode rather than real code because it includes an imaginary function, `get_gradients_of()`. In reality, retrieving gradients is done in a way that is specific to your current backend—JAX, TensorFlow, or PyTorch.
[p240_e09] (body) Let’s use what you learned about each framework in chapter 3 to implement a real version of this `train_step()` function. We’ll start with TensorFlow and PyTorch because these two make the job relatively easy, so they’re a good place to start. We’ll end with JAX, which is quite a bit more complex.
[p240_e10] (heading_subsubsection) A TensorFlow training step function
[p240_e11] (body) TensorFlow lets you write code that looks pretty much like our pseudocode snippet. The only difference is that your forward pass should take place inside a `GradientTape` scope. You can then use the `tape` object to retrieve the gradients:
[p240_e12] (code)
```
import tensorflow as tf
model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()
```

## Code annotations printed next to listings (callouts, in order)
- Runs the forward pass
- Computes the loss
- for the current batch
- Updates the model’s trainable
- weights based on the gradients
- Retrieves the gradients of the loss with regard to the model’s
- trainable weights. This function doesn’t actually exist!

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

## Previous page (PDF 239) text, for continuity
Whenever you find yourself in a situation where the built-in `fit()` is not enough, you will need to write your own custom training logic. You’ve already seen simple examples of low-level training loops in chapters 2 and 3. As a reminder, the contents of a typical training loop look like this: 1 Run the “forward pass” (compute the model’s output) to obtain a loss value for

the current batch of data. 2 Retrieve the gradients of the loss with regard to the model’s weights. 3 Update the model’s weights so as to lower the loss value on the current batch of data. These steps are repeated for as many batches as necessary. This is essentially what `fit()` does under the hood. In this section, you will learn to reimplement `fit()` from scratch, which will give you all the knowledge you need to write any training algorithm you may come up with. Let’s go over the details. Throughout the next few sections, you’ll work your way up to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.

7.4.1

Training vs. inference

In the low-level training loop examples you’ve seen so far, step 1 (the forward pass) was done via `predictions = model(inputs)`, and step 2 (retrieving the gradients computed by the gradient tape) was done via a backend-specific API, such as

`gradients = tape.gradient(loss, model.weights)` in TensorFlow

`loss.backward()` in PyTorch

`jax.value_and_grad()` in JAX

In the general case, there are actually two subtleties you need to take into account.

Some Keras layers, such as the `Dropout` layer, have different behaviors during training and during inference (when you use them to generate predictions). Such layers expose a `training` Boolean argument in their `call()` method. Calling `dropout(inputs, ` `training=True)` will drop some activat …

## Next page (PDF 241) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.250 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.242 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.247 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.244 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.540 (printed 516) — 17 Image generation › 17.1 Deep learning for image generation › 
Now, let’s create the VAE model itself. This is your first example of a model that isn’t doing supervised learning (an autencoder is an example of self-supervised learning because it uses its inputs as targets). Whenever you depart from classic supervised learning, it’s common to subclass the `Model` class and implement a custom `train_` `step()` to specify the new training logic, a workflow you’ve learned about in chapter 7. We could easily do that here, but a downside of this technique is that the `train_step()` contents must be backend specific—you’d use `GradientTape` with TensorFlow, you’d use `loss.backward()` with PyTorch, and so on. A simpler way to customize your training logic is to just implement the `compute_loss()` method instead and keep the default `train_step()`. `compute_loss()` is the key bit of differentiable logic called by the built-in `train_step()`. Since it doesn’t involve direct manipulation of gradients, it’s easy to keep it backend agnostic.

Its signature is as follows:

### PDF p.113 (printed 89) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
Next, we can set up our training step function. It looks straightforward. Be mindful that, unlike its TensorFlow and PyTorch equivalents, it needs to be stateless, and so it must return the updated values of the `W` and `b` tensors:

### PDF p.214 (printed 190) — 7 A deep dive on Keras ›  › This chapter covers
The different ways to create Keras models: the

`Sequential` class, the Functional API, and model

subclassing

How to use the built-in Keras training and

evaluation loops, including how to use custom

metrics and custom losses

Using Keras callbacks to further customize how

training proceeds

Using TensorBoard for monitoring your training

and evaluation metrics over time

How to write your own training and evaluation

loops from scratch

You’re starting to have some amount of experience with Keras. You’re familiar with the Sequential model, `Dense` layers, and built-in APIs for training, evaluation, and inference—`compile()`, `fit()`, `evaluate()`, and `predict()`. You’ve even learned in chapter 3 how to inherit from the `Layer` class to create custom layers, and how to

### PDF p.242 (printed 218) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A JAX training step function
When it comes to low-level training code, JAX tends to be the most complex of the three backends because of its fully stateless nature. Statelessness makes JAX highly performant and scalable, making it amenable to compilation and automatic performance optimizations. However, writing stateless code requires you to jump through some hoops.

Since the gradient function is obtained via metaprogramming, you first need to define the function that returns your loss. Further, this function needs to be stateless, so it needs to take as arguments all the variables it’s going to be using, and it needs to return the value of any variable it has updated. Remember those non-trainable weights that can get modified during the forward pass? Those are the variables we need to return. To make it easier to work with the stateless programming paradigm of JAX, Keras models make available a stateless forward pass method: the `stateless_call()` method. It behaves just like `call`, except that

It takes as input the model’s trainable weights and non-trainable weights, in addition to the `inputs` and `training` arguments.

### PDF p.318 (printed 294) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
(continued)

You should use `model(x)` when you need to retrieve the gradients of the model call. And you should use `predict()` if you just need the output value. In other words, always use `predict()`, unless you’re in the middle of writing a low-level gradient descent loop (as we are now).

A non-obvious trick to help the gradient-ascent process go smoothly is to normalize the gradient tensor by dividing it by its L2 norm (the square root of the sum of the squares of the values in the tensor). This ensures that the magnitude of the updates done to the input image is always within the same range.

Let’s set up the gradient ascent step function. Anything that involves gradients requires calling backend-level APIs, such as `GradientTape` in TensorFlow, `.backward()` in PyTorch, and `jax.grad()` in JAX. Let’s line up all the code snippets for each of the three backends, starting with TensorFlow.

### PDF p.96 (printed 72) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
For simplicity, we’ll do batch training instead of mini-batch training: we’ll run each training step (gradient computation and weight update) on the entire data, rather than iterate over the data in small batches. On one hand, this means that each training step will take much longer to run, since we compute the forward pass and the gradients for 2,000 samples at once. On the other hand, each gradient update will be much more effective at reducing the loss on the training data, since it will encompass information from all training samples instead of, say, only 128 random samples. As a result, we will need many fewer steps of training, and we should use a larger learning rate than what we would typically use for mini-batch training (we’ll use `learning_rate = 0.1`, as previously defined).

Listing 3.21 The batch training loop

### PDF p.96 (printed 72) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
Now, we move to the training step, which receives some training data and updates the weights `W` and `b` to minimize the loss on the data.

Listing 3.20 The training-step function
