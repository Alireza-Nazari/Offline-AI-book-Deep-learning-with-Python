# Generation bundle for page_241  (PDF page 241, printed page 217)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.2 Writing custom training step functions
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_241.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_241.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p241_e03] (code)
```
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply(gradients, model.trainable_weights)
    return loss
```
[p241_e04] (body) Let’s run it for a single step:
[p241_e05] (code)
```
batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)
```
[p241_e06] (body) Easy enough! Let’s do PyTorch next.
[p241_e07] (heading_subsubsection) A PyTorch training step function
[p241_e08] (body) When you use the PyTorch backend, all of your Keras layers and models inherit from the PyTorch `torch.nn.Module` class and expose the native `Module` API. As a result, your model, its trainable weights, and your loss tensor are all aware of each other and interact via three methods: `loss.backward()`, `weight.value.grad`, and `model
` `.zero_grad()`.
[p241_e09] (body) As a reminder from chapter 3, the mental model you’ve got to keep in mind is this:
[p241_e10] (bullet) With each forward pass, PyTorch builds up a one-time computation graph that keeps track of the computation that just happened.
[p241_e11] (bullet) Calling `.backward()` on any given scalar node of this graph (like your loss) will run the graph backward starting from that node, automatically populating a `tensor.grad` attribute on all tensors involved (if they satisfy `requires_` `grad=True`), containing the gradient of the output node with respect to that tensor. In particular, it will populate the `grad` attribute of your trainable parameters.
[p241_e12] (bullet) To clear the contents of that `tensor.grad` attribute, you should call `tensor.grad ` `= None` on all your tensors. Because it would be a bit cumbersome to do this on all model variables individually, you can just do it at the model level via `model
` `.zero_grad()`—the `zero_grad()` call will propagate to all variables tracked by the model. Clearing gradients is critical because calls to `backward()` are additive: if you don’t clear the gradients at each step, the gradient values will accumulate and training won’t proceed.

## Code annotations printed next to listings (callouts, in order)
- Opens a GradientTape
- Runs the
- forward pass
- Retrieves the
- gradients
- Updates the model’s trainable
- from the tape
- weights based on the gradients

## Equation candidates (formulas written inline in the text)
- [p241_e05] batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)
- [p241_e11] Calling `.backward()` on any given scalar node of this graph (like your loss) will run the graph backward starting from that node, automatically populating a `tensor.grad` attribute on all tensors involved (if they satisfy `requires_` `grad=True`), containing the gradient of the output node with respect to that tensor.

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

## Previous page (PDF 240) text, for continuity
stored in a non-trainable weight, and at each batch, your layer would increment the counter by one. Among Keras built-in layers, the only layer that features non-trainable weights is the `BatchNormalization` layer, which we will introduce in chapter 9. The `BatchNormalization` layer needs non-trainable weights to track information about the mean and standard deviation of the data that passes through it, so as to perform an online approximation of feature normalization (a concept you’ve learned about in chapters 4 and 6).

7.4.2

Writing custom training step functions

Taking into account these two details, a supervised learning training step ends up looking like this in pseudocode:

This snippet is pseudocode rather than real code because it includes an imaginary function, `get_gradients_of()`. In reality, retrieving gradients is done in a way that is specific to your current backend—JAX, TensorFlow, or PyTorch.

Let’s use what you learned about each framework in chapter 3 to implement a real version of this `train_step()` function. We’ll start with TensorFlow and PyTorch because these two make the job relatively easy, so they’re a good place to start. We’ll end with JAX, which is quite a bit more complex.

A TensorFlow training step function

TensorFlow lets you write code that looks pretty much like our pseudocode snippet. The only difference is that your forward pass should take place inside a `GradientTape` scope. You can then use the `tape` object to retrieve the gradients:

## Next page (PDF 242) text, for continuity
Let’s chain these steps:

Let’s run it for a single step:

That wasn’t too difficult! Now, let’s move on to JAX.

A JAX training step function

When it comes to low-level training code, JAX tends to be the most complex of the three backends because of its fully stateless nature. Statelessness makes JAX highly performant and scalable, making it amenable to compilation and automatic performance optimizations. However, writing stateless code requires you to jump through some hoops.

Since the gradient function is obtained via metaprogramming, you first need to define the function that returns your loss. Further, this function needs to be stateless, so it needs to take as arguments all the variables it’s going to be using, and it needs to return the value of any variable it has updated. Remember those non-trainable weights that can get modified during the forward pass? Those are the variables we need to return. To make it easier to work with the stateless programming paradigm of JAX, Keras models make available a stateless forward pass method: the `stateless_call()` method. It behaves just like `call`, except that

It takes as input the model’s trainable weights and non-trainable weights, in addition to the `inputs` and `training` arguments.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.250 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.104 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.239 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.103 (printed 79) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
1 `loss.backward()` runs backpropagation starting from the `loss` output node and

populates the `tensor.grad` attribute on all tensors that were involved in the computation of `loss`. `tensor.grad` represents the gradient of the loss with regard to that tensor. 2 We use the `.grad` attribute to recover the gradients of the loss with regard to `W` and `b`. 3 We update `W` and `b` using those gradients. Because these updates are not intended to be part of the backward pass, we do them inside a `torch.no_grad()` scope, which skips gradient computation for everything inside it. 4 We reset the contents of the `.grad` property of our `W` and `b` parameters, by setting it to `None`. If we didn’t do this, gradient values would accumulate across multiple calls to `training_step()`, resulting in invalid values:

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Training vs. inference
In the low-level training loop examples you’ve seen so far, step 1 (the forward pass) was done via `predictions = model(inputs)`, and step 2 (retrieving the gradients computed by the gradient tape) was done via a backend-specific API, such as

`gradients = tape.gradient(loss, model.weights)` in TensorFlow

`loss.backward()` in PyTorch

`jax.value_and_grad()` in JAX

In the general case, there are actually two subtleties you need to take into account.

Some Keras layers, such as the `Dropout` layer, have different behaviors during training and during inference (when you use them to generate predictions). Such layers expose a `training` Boolean argument in their `call()` method. Calling `dropout(inputs, ` `training=True)` will drop some activation entries, while calling `dropout(inputs, ` `training=False)` does nothing. By extension, Functional models and Sequential models also expose this `training` argument in their `call()` methods. Remember to pass `training=True` when you call a Keras model during the forward pass! Our forward pass thus becomes `predictions = model(inputs, training=True)`. In addition, note that when you retrieve the gradients of the weights of your model, you s

### PDF p.105 (printed 81) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
Similarly, previously we needed to reset parameter gradients by hand by doing `tensor.grad = None` on each one. Now we can just do `model.zero_grad()`. Overall, this may feel a bit confusing—somehow the loss tensor, the optimizer, and the `Module` instance all seem to be aware of each other through some hidden background mechanism. They’re all interacting with one another via spooky action at a distance. Don’t worry though—you can just treat this sequence of steps (`loss.backward()` - `optimizer.step()` - `model.zero_grad()`) as a magic incantation to be recited any time you need to write a training step function. Just make sure not to forget `model.zero_` `grad()`. That would be a major bug (and it is unfortunately quite common)!

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Computing gradients with PyTorch
If you call `backward()` multiple times in a row, the `.grad` attribute will “accumulate” gradients: each new call will sum the new gradient with the preexisting one. For instance,

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Computing gradients with PyTorch
Listing 3.26 Computing a gradient with `.backward()`
>>> input_var = torch.tensor(3.0, requires_grad=True)
>>> result = torch.square(input_var)
>>> result.backward()
>>> gradient = input_var.grad
>>> gradient
tensor(6.)

### PDF p.103 (printed 79) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Packaging state and computation with the Module class
PyTorch also has a higher-level, object-oriented API for performing backpropagation, which requires relying on two new classes: the `torch.nn.Module` class and an optimizer class from the `torch.optim` module, such as `torch.optim.SGD` (the equivalent of `keras
` `.optimizers.SGD`).

The general idea is to define a subclass of `torch.nn.Module`, which will

Hold some `Parameters`, to store state variables. Those are defined in the `__init__()`method.

Implement the forward pass computation in the `forward()` method.

It should look just like the following.

Listing 3.27 Defining a `torch.nn.Module`

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Whenever you find yourself in a situation where the built-in `fit()` is not enough, you will need to write your own custom training logic. You’ve already seen simple examples of low-level training loops in chapters 2 and 3. As a reminder, the contents of a typical training loop look like this: 1 Run the “forward pass” (compute the model’s output) to obtain a loss value for

the current batch of data. 2 Retrieve the gradients of the loss with regard to the model’s weights. 3 Update the model’s weights so as to lower the loss value on the current batch of data. These steps are repeated for as many batches as necessary. This is essentially what `fit()` does under the hood. In this section, you will learn to reimplement `fit()` from scratch, which will give you all the knowledge you need to write any training algorithm you may come up with. Let’s go over the details. Throughout the next few sections, you’ll work your way up to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.

### PDF p.251 (printed 227) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › train_step() metrics handling with PyTorch
import keras
from keras import layers
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        predictions = self(inputs, training=True)
        loss = self.compute_loss(y=targets, y_pred=predictions)
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
