# Generation bundle for page_100  (PDF page 100, printed page 76)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.4 Introduction to PyTorch
Subsection: 3.4.1 First steps with PyTorch
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_100.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_100.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p100_e03] (listing_caption) Listing 3.23 Random tensors
[p100_e04] (code)
```
>>> torch.normal(
... mean=torch.zeros(size=(3, 1)),
... std=torch.ones(size=(3, 1)))
tensor([[-0.9613],
        [-2.0169],
        [ 0.2088]])
```
[p100_e05] (body) As for creating a random uniform tensor, you’d do that via `torch.rand`. Unlike `np.random
` `.uniform` or `tf.random.uniform`, the output shape should be provided as independent arguments for each dimension, like this:
[p100_e06] (code)
```
>>> torch.rand(3, 1)
```
[p100_e07] (heading_subsubsection) Tensor assignment and the Parameter class
[p100_e08] (body) Like NumPy arrays, but unlike TensorFlow tensors, PyTorch tensors are assignable. You can do operations like this:
[p100_e09] (code)
```
>>> x = torch.zeros(size=(2, 1))
>>> x[0, 0] = 1.
>>> x
tensor([[1.],
        [0.]])
```
[p100_e10] (body) While you can just use a regular `torch.Tensor` to store the trainable state of a model, PyTorch does provide a specialized tensor subclass for that purpose, the `torch
` `.nn.parameter.Parameter` class. Compared to a regular tensor, it provides semantic clarity—if you see a `Parameter`, you’ll know it’s a piece of trainable state, whereas a `Tensor` could be anything. As a result, it enables PyTorch to automatically track and retrieve the `Parameters` you assign to PyTorch models—similar to what Keras does with Keras `Variable` instances.
[p100_e11] (body) Here’s a `Parameter`.
[p100_e12] (listing_caption) Listing 3.24 Creating a PyTorch parameter
[p100_e13] (code)
```
>>> x = torch.zeros(size=(2, 1))
>>> p = torch.nn.parameter.Parameter(data=x)
```

## Code annotations printed next to listings (callouts, in order)
- Equivalent to tf.random.normal(shape=(3, 1),
- mean=0., stddev=1.)
- Equivalent to tf.random.uniform(shape=(3, 1),
- minval=0., maxval=1.)
- A Parameter can only be
- created using a torch.
- Tensor value—no NumPy
- arrays allowed.

## Listings
- Listing 3.23: Random tensors (code elements p100_e04)
- Listing 3.24: Creating a PyTorch parameter (code elements p100_e13)

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

## Previous page (PDF 99) text, for continuity
The PyTorch API is higher level than that of TensorFlow and JAX: it includes layers and optimizers, like Keras. These layers and optimizers are compatible with Keras workflows when you use Keras with the PyTorch backend.

3.4.1

First steps with PyTorch

Over the next paragraphs, you’ll get familiar with all the basics of PyTorch. We’ll cover the following key concepts:

Tensors and parameters

Numerical operations in PyTorch

Computing gradients with the `backward()` method

Packaging computation with the `Module` class

Speeding up PyTorch by using compilation

We’ll conclude the introduction by reimplementing our linear regression end-to-end example in pure PyTorch.

Tensors and parameters in PyTorch

A first gotcha about PyTorch is that the package isn’t named `pytorch`. It’s actually named `torch`. You’d install it via `pip install torch` and you’d import it via `import torch`.

Like in NumPy and TensorFlow, the object at the heart of the framework is the tensor. First, let’s get our hands on some PyTorch tensors.

Constant tensors

Here are some constant tensors.

Listing 3.22 All-ones or all-zeros tensors

Random tensors

Random tensor creation is similar to NumPy and TensorFlow, but with divergent syntax. Consider the function `normal`: it doesn’t take a shape argument. Instead, the mean and standard deviation should be provided as PyTorch tensors with the expected output shape.

## Next page (PDF 101) text, for continuity
Tensor operations: Doing math in PyTorch

Math in PyTorch works just the same as math in NumPy or TensorFlow, although much like TensorFlow, the PyTorch API often diverges in subtle ways from the NumPy API.

Listing 3.25 A few basic math operations in PyTorch

Here’s a dense layer:

Computing gradients with PyTorch

There’s no explicit “gradient tape” in PyTorch. A similar mechanism does exist: when you run any computation in PyTorch, the framework creates a one-time computation graph (a “tape”) that records what just happened. However, that tape is hidden from the user. The public API for using it is at the level of tensors themselves: you can call `tensor.backward()` to run backpropagation through all operations previously executed that led to that tensor. Doing this will populate the `.grad` attribute of all tensors that are tracking gradients.

Listing 3.26 Computing a gradient with `.backward()`

If you call `backward()` multiple times in a row, the `.grad` attribute will “accumulate” gradients: each new call will sum the new gradient with the preexisting one. For instance,

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.89 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.102 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.90 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.107 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.109 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.113 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.88 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.89 (printed 65) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Tensor assignment and the Variable class
To train a model, we’ll need to update its state, which is a set of tensors. If tensors aren’t assignable, how do we do it, then? That’s where variables come in. `tf.Variable` is the class meant to manage modifiable state in TensorFlow.

To create a variable, you need to provide some initial value, such as a random tensor.

Listing 3.5 Creating a `tf.Variable`

### PDF p.89 (printed 65) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Tensor assignment and the Variable class
Listing 3.5 Creating a `tf.Variable`
>>> v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
>>> print(v)

### PDF p.241 (printed 217) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A PyTorch training step function
When you use the PyTorch backend, all of your Keras layers and models inherit from the PyTorch `torch.nn.Module` class and expose the native `Module` API. As a result, your model, its trainable weights, and your loss tensor are all aware of each other and interact via three methods: `loss.backward()`, `weight.value.grad`, and `model
` `.zero_grad()`.

As a reminder from chapter 3, the mental model you’ve got to keep in mind is this:

With each forward pass, PyTorch builds up a one-time computation graph that keeps track of the computation that just happened.

Calling `.backward()` on any given scalar node of this graph (like your loss) will run the graph backward starting from that node, automatically populating a `tensor.grad` attribute on all tensors involved (if they satisfy `requires_` `grad=True`), containing the gradient of the output node with respect to that tensor. In particular, it will populate the `grad` attribute of your trainable parameters.

To clear the contents of that `tensor.grad` attribute, you should call `tensor.grad ` `= None` on all your tensors. Because it would be a bit cumbersome to do this on all model variables individually, you can just do it at the mo

### PDF p.103 (printed 79) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Packaging state and computation with the Module class
PyTorch also has a higher-level, object-oriented API for performing backpropagation, which requires relying on two new classes: the `torch.nn.Module` class and an optimizer class from the `torch.optim` module, such as `torch.optim.SGD` (the equivalent of `keras
` `.optimizers.SGD`).

The general idea is to define a subclass of `torch.nn.Module`, which will

Hold some `Parameters`, to store state variables. Those are defined in the `__init__()`method.

Implement the forward pass computation in the `forward()` method.

It should look just like the following.

Listing 3.27 Defining a `torch.nn.Module`

### PDF p.106 (printed 82) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
The popular pretrained model-sharing platform Hugging Face has first-class support for PyTorch, which means that any model you’d like to use is likely available in PyTorch. This is the primary drive behind PyTorch adoption today.

Meanwhile, there are also some downsides to using PyTorch:

Like with TensorFlow, the PyTorch API is inconsistent with NumPy. Further, it’s also internally inconsistent. For instance, the commonly used keyword `axis` is occasionally named `dim` instead, depending on the function. Some pseudo-random number generation operations take a `seed` argument; others don’t. And so on. This can make PyTorch frustrating to learn, especially when coming from NumPy.

Due to its focus on eager execution, PyTorch is quite slow—it’s the slowest of all the major frameworks by a large margin. For most models, you may see a 20% or 30% speedup with JAX. For some models—especially large ones—you may even see a 3× or a 5× speedup with JAX, even after using `torch.compile()`.

While it is possible to make PyTorch code faster via `torch.compile()`, the PyTorch Dynamo compiler remains at this time (in 2025) quite ineffective and full of trapdoors. As a result, only a very small pe

### PDF p.88 (printed 64) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Constant tensors
Tensors need to be created with some initial value, so common ways to create tensors are via `tf.ones` (equivalent to `np.ones`) and `tf.zeros` (equivalent to `np.zeros`). You can also create a tensor from Python or NumPy values using `tf.constant`.

Listing 3.1 All-ones or all-zeros tensors

### PDF p.88 (printed 64) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Random tensors
You can also create tensors filled with random values via one of the methods of the `tf.random` submodule (equivalent to the `np.random` submodule).

### PDF p.104 (printed 80) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
Now, let’s get our hands on a PyTorch optimizer. To instantiate it, you will need to provide the list of parameters that the optimizer is intended to update. You can retrieve it from our `Module` instance via `.parameters()`:
