# Generation bundle for page_078  (PDF page 78, printed page 54)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.5 Looking back at our first example
Subsection: 2.5.1 Reimplementing our first example from scratch
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_078.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_078.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p078_e03] (code)
```
    @property
    def weights(self):
        return [self.W, self.b]
```
[p078_e04] (heading_subsubsection) A simple Sequential class
[p078_e05] (body) Now, let’s create a `NaiveSequential` class to chain these layers. It wraps a list of layers and exposes a `__call__()` method that simply calls the underlying layers on the inputs, in order. It also features a `weights` property to easily keep track of the layers’ parameters:
[p078_e06] (code)
```
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
```
[p078_e07] (body) Using this `NaiveDense` class and this `NaiveSequential` class, we can create a mock Keras model:
[p078_e08] (code)
```
model = NaiveSequential(
    [
        NaiveDense(input_size=28 * 28, output_size=512, activation=ops.relu),
        NaiveDense(input_size=512, output_size=10, activation=ops.softmax),
    ]
)
assert len(model.weights) == 4
```
[p078_e09] (heading_subsubsection) A batch generator
[p078_e10] (body) Next, we need a way to iterate over the MNIST data in mini-batches. This is easy:
[p078_e11] (code)
```
import math
class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
```

## Code annotations printed next to listings (callouts, in order)
- The convenience method for
- retrieving the layer’s weights

## Chapter outline
Chapter: 2 The mathematical building blocks of neural networks (PDF pages 40-83)
  - 2.1 A first look at a neural network (p.41)
  - 2.2 Data representations for neural networks (p.45)
      - 2.2.1 Scalars (rank-0 tensors) (p.46)
      - 2.2.2 Vectors (rank-1 tensors) (p.46)
      - 2.2.3 Matrices (rank-2 tensors) (p.46)
      - 2.2.4 Rank-3 tensors and higher-rank tensors (p.47)
      - 2.2.5 Key attributes (p.47)
      - 2.2.6 Manipulating tensors in NumPy (p.49)
      - 2.2.7 The notion of data batches (p.49)
      - 2.2.8 Real-world examples of data tensors (p.50)
  - 2.3 The gears of neural networks: Tensor operations (p.52)
      - 2.3.1 Element-wise operations (p.53)
      - 2.3.2 Broadcasting (p.54)
      - 2.3.3 Tensor product (p.56)
      - 2.3.4 Tensor reshaping (p.58)
      - 2.3.5 Geometric interpretation of tensor operations (p.59)
      - 2.3.6 A geometric interpretation of deep learning (p.62)
  - 2.4 The engine of neural networks: Gradient-based optimization (p.63)
      - 2.4.1 What’s a derivative? (p.65)
      - 2.4.2 Derivative of a tensor operation: The gradient (p.66)
      - 2.4.3 Stochastic gradient descent (p.67)
      - 2.4.4 Chaining derivatives: The Backpropagation algorithm (p.70)
  - 2.5 Looking back at our first example (p.75)
      - 2.5.1 Reimplementing our first example from scratch (p.77)
      - 2.5.2 Running one training step (p.79)
      - 2.5.3 The full training loop (p.81)
      - 2.5.4 Evaluating the model (p.82)

## Previous page (PDF 77) text, for continuity
At this point, you already know most of what there is to know about neural networks. Let’s prove it by reimplementing a simplified version of that first example step by step, using only low-level operations.

2.5.1

Reimplementing our first example from scratch

What’s better to demonstrate full, unambiguous understanding than to implement everything from scratch? Of course, what “from scratch” means here is relative: we won’t reimplement basic tensor operations, and we won’t implement backpropagation. But we’ll go to such a low level that each computation step will be made explicit.

Don’t worry if you don’t understand every little detail in this example just yet. The next chapter will dive in more detail into the Keras API. For now, just try to follow the gist of what’s going on—the intent of this example is to help crystallize your understanding of the mathematics of deep learning using a concrete implementation. Let’s go!

A simple Dense class

You’ve learned earlier that the `Dense` layer implements the following input transformation, where `W` and `b` are model parameters, and `activation` is an element-wise function (usually `relu`):

Let’s implement a simple Python class `NaiveDense` that creates two Keras variables `W` and `b`, and exposes a `__call__()` method that applies the previous transformation:

## Next page (PDF 79) text, for continuity
2.5.2

Running one training step

The most difficult part of the process is the “training step”: updating the weights of the model after running it on one batch of data. We need to

Compute the predictions of the model for the images in the batch

Compute the loss value for these predictions given the actual labels

Compute the gradient of the loss with regard to the model’s weights

Move the weights by a small amount in the direction opposite to the gradient

Listing 2.9 A single step of training

The weight update step

As you already know, the purpose of the “weight update” step (represented by the `update_weights()` function) is to move the weights by “a bit” in a direction that will reduce the loss on this batch. The magnitude of the move is determined by the

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.119 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.80 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.117 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.55 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.81 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.217 (printed 193) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
You’ve seen in chapter 3 that layers only get built (which is to say, create their weights) when they are called for the first time. That’s because the shape of the layers’ weights depends on the shape of their input: until the input shape is known, they can’t be created.

As such, the previous Sequential model does not have any weights until you actually call it on some data, or call its `build()` method with an input shape.

Listing 7.3 Models that aren’t yet built have no weights

### PDF p.119 (printed 95) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
It would be even worse when the rules used by a layer to produce its output shape are complex. For instance, what if our layer returned outputs of shape `(batch,` `input_size ` `* 2 if input_size % 2 == 0 else input_size * 3)?`

If we were to reimplement our `NaiveDense` layer as a Keras layer capable of automatic shape inference, it would look like the `SimpleDense` layer, with its `build()` and `call()` methods. In the Keras `SimpleDense`, we no longer create weights in the constructor like in the previous example. Instead, we create them in a dedicated state-creation method `build()`, which receives as argument the first input shape seen by the layer. The `build()` method is called automatically the first time the layer is called (via its `__call__()` method). In fact, that’s why we defined the computation in a separate `call()` method rather than in the `__call__()` method directly! The `__call__()` method of the base layer schematically looks like this:

### PDF p.226 (printed 202) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Subclassing the Model class
The last model-building pattern you should know about is the most advanced one: `Model` subclassing. You’ve already learned in chapter 3 how to subclass the `Layer` class to create custom layers. Subclassing `Model` is pretty similar:

In the `__init__()` method, define the layers the model will use.

In the `call` method, define the forward pass of the model, reusing the layers previously created.

Instantiate your subclass and call it on data to create its weights.

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

### PDF p.216 (printed 192) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › The Sequential model
Listing 7.1 The `Sequential` class
import keras
from keras import layers
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

### PDF p.117 (printed 93) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › The base Layer class in Keras
A simple API should have a single abstraction around which everything is centered. In Keras, that’s the `Layer` class. Everything in Keras is either a `Layer` or something that closely interacts with a `Layer`.

A `Layer` is an object that encapsulates some state (weights) and some computation (a forward pass). The weights are typically defined in a `build()` (although they could also be created in the constructor `__init__()`), and the computation is defined in the `call()` method. In the previous chapter, we implemented a `NaiveDense` class that contained two weights `W` and `b` and applied the computation `output = activation(matmul(input, W) + ` `b)`. The following is what the same layer would look like in Keras.

Listing 3.33 A simple dense layer from scratch in Keras

### PDF p.120 (printed 96) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Note that automatic shape inference is not the only thing that the `Layer` class’s `__call__()` method handles. It takes care of many more things, in particular routing between eager and graph execution, and input masking (which we cover in chapter 14). For now, just remember: when implementing your own layers, put the forward pass in the `call()` method.

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Training vs. inference
In the low-level training loop examples you’ve seen so far, step 1 (the forward pass) was done via `predictions = model(inputs)`, and step 2 (retrieving the gradients computed by the gradient tape) was done via a backend-specific API, such as

`gradients = tape.gradient(loss, model.weights)` in TensorFlow

`loss.backward()` in PyTorch

`jax.value_and_grad()` in JAX

In the general case, there are actually two subtleties you need to take into account.

Some Keras layers, such as the `Dropout` layer, have different behaviors during training and during inference (when you use them to generate predictions). Such layers expose a `training` Boolean argument in their `call()` method. Calling `dropout(inputs, ` `training=True)` will drop some activation entries, while calling `dropout(inputs, ` `training=False)` does nothing. By extension, Functional models and Sequential models also expose this `training` argument in their `call()` methods. Remember to pass `training=True` when you call a Keras model during the forward pass! Our forward pass thus becomes `predictions = model(inputs, training=True)`. In addition, note that when you retrieve the gradients of the weights of your model, you s
