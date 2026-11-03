# Generation bundle for page_099  (PDF page 99, printed page 75)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.4 Introduction to PyTorch
Subsection: 3.4.1 First steps with PyTorch
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_099.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_099.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p099_e03] (body) The PyTorch API is higher level than that of TensorFlow and JAX: it includes layers and optimizers, like Keras. These layers and optimizers are compatible with Keras workflows when you use Keras with the PyTorch backend.
[p099_e04] (heading_subsection) 3.4.1
[p099_e05] (heading_subsection) First steps with PyTorch
[p099_e06] (body) Over the next paragraphs, you’ll get familiar with all the basics of PyTorch. We’ll cover the following key concepts:
[p099_e07] (bullet) Tensors and parameters
[p099_e08] (bullet) Numerical operations in PyTorch
[p099_e09] (bullet) Computing gradients with the `backward()` method
[p099_e10] (bullet) Packaging computation with the `Module` class
[p099_e11] (bullet) Speeding up PyTorch by using compilation
[p099_e12] (body) We’ll conclude the introduction by reimplementing our linear regression end-to-end example in pure PyTorch.
[p099_e13] (heading_subsubsection) Tensors and parameters in PyTorch
[p099_e14] (body) A first gotcha about PyTorch is that the package isn’t named `pytorch`. It’s actually named `torch`. You’d install it via `pip install torch` and you’d import it via `import torch`.
[p099_e15] (body) Like in NumPy and TensorFlow, the object at the heart of the framework is the tensor. First, let’s get our hands on some PyTorch tensors.
[p099_e16] (heading_subsubsection) Constant tensors
[p099_e17] (body) Here are some constant tensors.
[p099_e18] (listing_caption) Listing 3.22 All-ones or all-zeros tensors
[p099_e19] (code)
```
>>> import torch
>>> torch.ones(size=(2, 1))
tensor([[1.], [1.]])
>>> torch.zeros(size=(2, 1))
tensor([[0.], [0.]])
>>> torch.tensor([1, 2, 3], dtype=torch.float32)
tensor([1., 2., 3.])
```
[p099_e20] (heading_subsubsection) Random tensors
[p099_e21] (body) Random tensor creation is similar to NumPy and TensorFlow, but with divergent syntax. Consider the function `normal`: it doesn’t take a shape argument. Instead, the mean and standard deviation should be provided as PyTorch tensors with the expected output shape.

## Code annotations printed next to listings (callouts, in order)
- Unlike in other frameworks, the shape
- argument is named “size” rather than “shape.”
- Unlike in other frameworks, you
- cannot pass dtype="float32" as
- a string. The dtype argument
- must be a torch dtype instance.

## Listings
- Listing 3.22: All-ones or all-zeros tensors (code elements p099_e19)

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

## Previous page (PDF 98) text, for continuity
This is really what a linear classifier is all about: finding the parameters of a line (or, in higher-dimensional spaces, a hyperplane) neatly separating two classes of data.

3.3.3

What makes the TensorFlow approach unique

You’re now familiar with all the basic APIs that underlie TensorFlow-based workflows, and you’re about to dive into more frameworks—in particular, PyTorch and JAX. What makes working with TensorFlow different from working with any other framework? When should you use TensorFlow, and when could you use something else?

If you ask us, here are the main benefits of TensorFlow:

Thanks to graph mode and XLA compilation, it’s fast. It’s usually significantly faster than PyTorch and NumPy, though JAX is often even faster.

It is extremely feature complete. Unique among all frameworks, it has support for string tensors as well as “ragged tensors” (tensors where different entries may have different dimensions—very useful for handling sequences without requiring to pad them to a shared length). It also has outstanding support for data preprocessing, via the highly performant `tf.data` API. `tf.data` is so good that even JAX recommends it for data preprocessing. Whatever you need to do, TensorFlow has a solution for it.

Its ecosystem for production deployment is the most mature among all frameworks, especially when it comes to deploying on mobile or in the browser.

However, TensorFlow also has some noticeable flaws:

It has a sprawling API—the flipside of being very feature complete. TensorFlow includes thousands of different operations.

Its numerical API is occasionally inconsistent with the NumPy API, making it a bit harder to approach if you’re already familiar with NumPy.

The popular pretrained model-sharing platform Hugging Face has less support for …

## Next page (PDF 100) text, for continuity
Listing 3.23 Random tensors

As for creating a random uniform tensor, you’d do that via `torch.rand`. Unlike `np.random
` `.uniform` or `tf.random.uniform`, the output shape should be provided as independent arguments for each dimension, like this:

Tensor assignment and the Parameter class

Like NumPy arrays, but unlike TensorFlow tensors, PyTorch tensors are assignable. You can do operations like this:

While you can just use a regular `torch.Tensor` to store the trainable state of a model, PyTorch does provide a specialized tensor subclass for that purpose, the `torch
` `.nn.parameter.Parameter` class. Compared to a regular tensor, it provides semantic clarity—if you see a `Parameter`, you’ll know it’s a piece of trainable state, whereas a `Tensor` could be anything. As a result, it enables PyTorch to automatically track and retrieve the `Parameters` you assign to PyTorch models—similar to what Keras does with Keras `Variable` instances.

Here’s a `Parameter`.

Listing 3.24 Creating a PyTorch parameter

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.88 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.116 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.101 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.87 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.105 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.102 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.106 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.127 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.106 (printed 82) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
The popular pretrained model-sharing platform Hugging Face has first-class support for PyTorch, which means that any model you’d like to use is likely available in PyTorch. This is the primary drive behind PyTorch adoption today.

Meanwhile, there are also some downsides to using PyTorch:

Like with TensorFlow, the PyTorch API is inconsistent with NumPy. Further, it’s also internally inconsistent. For instance, the commonly used keyword `axis` is occasionally named `dim` instead, depending on the function. Some pseudo-random number generation operations take a `seed` argument; others don’t. And so on. This can make PyTorch frustrating to learn, especially when coming from NumPy.

Due to its focus on eager execution, PyTorch is quite slow—it’s the slowest of all the major frameworks by a large margin. For most models, you may see a 20% or 30% speedup with JAX. For some models—especially large ones—you may even see a 3× or a 5× speedup with JAX, even after using `torch.compile()`.

While it is possible to make PyTorch code faster via `torch.compile()`, the PyTorch Dynamo compiler remains at this time (in 2025) quite ineffective and full of trapdoors. As a result, only a very small pe

### PDF p.116 (printed 92) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
NOTE When configuring the Keras backend, you should use the string `"torch"` to refer to the PyTorch backend, rather than the string `"pytorch"`, which would be invalid. This is because the PyTorch package name is `torch` (as in `import ` `torch` or `pip` `install` `torch`). Now, you may ask, which backend should I be picking? It’s really your own choice: all Keras code examples in the rest of the book will be compatible with all three backends. If the need for backend-specific code arises (as in chapter 7, for instance), we will show you all three versions—TensorFlow, PyTorch, JAX. If you have no particular backend preference, our personal recommendation is JAX. It’s usually the most performant backend. Once your backend is configured, you can start actually building and training Keras models. Let’s take a look.

### PDF p.87 (printed 63) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Introduction to TensorFlow
TensorFlow is a Python-based open source machine learning framework developed primarily by Google. Its initial release was in November 2015, followed by a v1 release in February 2017, and a v2 release in October 2019. TensorFlow is heavily used in production-grade machine learning applications across the industry.

2 Note that PyTorch is a bit of an intermediate case: while it is mainly a lower-level framework, it also includes its own layers and its own optimizers. However, if you use PyTorch in conjunction with Keras, then you will only interact with low-level PyTorch APIs such as tensor operations.

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Tensor operations: Doing math in PyTorch
Math in PyTorch works just the same as math in NumPy or TensorFlow, although much like TensorFlow, the PyTorch API often diverges in subtle ways from the NumPy API.

Listing 3.25 A few basic math operations in PyTorch

### PDF p.88 (printed 64) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › First steps with TensorFlow
Over the next paragraphs, you’ll get familiar with all the basics of TensorFlow. We’ll cover the following key concepts:

Tensors and variables

Numerical operations in TensorFlow

Computing gradients with a `GradientTape`

Making TensorFlow functions fast by using just-in-time compilation

We’ll then conclude the introduction with an end-to-end example: a pure-TensorFlow implementation of linear regression.

Let’s get those tensors flowing.

### PDF p.87 (printed 63) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › How these frameworks relate to each other
Keras, TensorFlow, PyTorch, and JAX don’t all have the same feature set and aren’t interchangeable. They have some overlap, but to a large extent, they serve different roles for different use cases. The biggest difference is between Keras and the three others. Keras is a high-level framework, while the others are lower level. Imagine building a house. Keras is like a prefabricated building kit: it provides a streamlined interface for setting up and training neural networks. In contrast, TensorFlow, PyTorch, and JAX are like the raw materials used in construction.

As you saw in the previous chapters, training a neural network revolves around the following concepts:

First, low-level tensor manipulation—The infrastructure that underlies all modern machine learning. This translates to low-level APIs found in TensorFlow, PyTorch,2 and JAX: – Tensors, including special tensors that store the network’s state (variables) – Tensor operations such as addition, `relu`, or `matmul` – Backpropagation, a way to compute the gradient of mathematical expressions

Second, high-level deep learning concepts—This translates to Keras APIs: – Layers, which are combined into a model – A loss function, w

### PDF p.104 (printed 80) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
Now, let’s get our hands on a PyTorch optimizer. To instantiate it, you will need to provide the list of parameters that the optimizer is intended to update. You can retrieve it from our `Module` instance via `.parameters()`:

### PDF p.102 (printed 78) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › An end-to-end example: A linear classifier in pure PyTorch
You now know enough to rewrite our linear classifier in PyTorch. It will stay very similar to the TensorFlow one—the only major difference is how we compute the gradients.

Let’s start by creating our model variables. Don’t forget to pass `requires_grad=True` so we can compute gradients with respect to them:
