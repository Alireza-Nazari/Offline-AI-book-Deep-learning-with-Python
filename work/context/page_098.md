# Generation bundle for page_098  (PDF page 98, printed page 74)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.4 Introduction to PyTorch
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_098.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_098.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p098_e03] (body) This is really what a linear classifier is all about: finding the parameters of a line (or, in higher-dimensional spaces, a hyperplane) neatly separating two classes of data.
[p098_e04] (heading_subsection) 3.3.3
[p098_e05] (heading_subsection) What makes the TensorFlow approach unique
[p098_e06] (body) You’re now familiar with all the basic APIs that underlie TensorFlow-based workflows, and you’re about to dive into more frameworks—in particular, PyTorch and JAX. What makes working with TensorFlow different from working with any other framework? When should you use TensorFlow, and when could you use something else?
[p098_e07] (body) If you ask us, here are the main benefits of TensorFlow:
[p098_e08] (bullet) Thanks to graph mode and XLA compilation, it’s fast. It’s usually significantly faster than PyTorch and NumPy, though JAX is often even faster.
[p098_e09] (bullet) It is extremely feature complete. Unique among all frameworks, it has support for string tensors as well as “ragged tensors” (tensors where different entries may have different dimensions—very useful for handling sequences without requiring to pad them to a shared length). It also has outstanding support for data preprocessing, via the highly performant `tf.data` API. `tf.data` is so good that even JAX recommends it for data preprocessing. Whatever you need to do, TensorFlow has a solution for it.
[p098_e10] (bullet) Its ecosystem for production deployment is the most mature among all frameworks, especially when it comes to deploying on mobile or in the browser.
[p098_e11] (body) However, TensorFlow also has some noticeable flaws:
[p098_e12] (bullet) It has a sprawling API—the flipside of being very feature complete. TensorFlow includes thousands of different operations.
[p098_e13] (bullet) Its numerical API is occasionally inconsistent with the NumPy API, making it a bit harder to approach if you’re already familiar with NumPy.
[p098_e14] (bullet) The popular pretrained model-sharing platform Hugging Face has less support for TensorFlow, which means that the latest generative AI models may not always be available in TensorFlow.
[p098_e15] (body) Now, let’s move on to PyTorch.
[p098_e16] (heading_section) 3.4
[p098_e17] (heading_section) Introduction to PyTorch
[p098_e18] (body) PyTorch is a Python-based open source machine learning framework developed primarily by Meta (formerly Facebook). It was originally released in September 2016 (as a response to the release of TensorFlow), with its 1.0 version launched in 2018, and its 2.0 version launched in 2023. PyTorch inherits its programming style from the now-defunct Chainer framework, which was itself inspired by NumPy-Autograd. PyTorch is used extensively in the machine learning research community.
[p098_e19] (body) Like TensorFlow, PyTorch is at the center of a large ecosystem of related packages, such as `torchvision`, `torchaudio`, or the popular model-sharing platform Hugging Face.

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

## Previous page (PDF 97) text, for continuity
Figure 3.2 Our model’s predictions on the training inputs: pretty similar to the training targets

Recall that the prediction value for a given point `[x, y]` is simply `prediction == [[w1], ` `[w2]] • [x, y] + b == w1 * x + w2 * y + b`. Thus, class “0” is defined as `w1 * x + w2 * ` `y + b < 0.5` and class “1” is defined as `w1 * x + w2 * y + b > 0.5`. You’ll notice that what you’re looking at is really the equation of a line in the 2D plane: `w1 * x + w2 * y + b ` `= 0.5`. Class 1 is above the line; class 0 is below the line. You may be used to seeing line equations in the format `y = a * x + b`; in the same format, our line becomes `y = - w1 ` `/ w2 * x + (0.5 - b) / w2`.

Let’s plot this line, as shown in figure 3.3:

Figure 3.3 Our model, visualized as a line

## Next page (PDF 99) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.87 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.106 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.86 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.1 A brief history of deep learning frameworks
- PDF p.105 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.101 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.127 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.115 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.106 (printed 82) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
The popular pretrained model-sharing platform Hugging Face has first-class support for PyTorch, which means that any model you’d like to use is likely available in PyTorch. This is the primary drive behind PyTorch adoption today.

Meanwhile, there are also some downsides to using PyTorch:

Like with TensorFlow, the PyTorch API is inconsistent with NumPy. Further, it’s also internally inconsistent. For instance, the commonly used keyword `axis` is occasionally named `dim` instead, depending on the function. Some pseudo-random number generation operations take a `seed` argument; others don’t. And so on. This can make PyTorch frustrating to learn, especially when coming from NumPy.

Due to its focus on eager execution, PyTorch is quite slow—it’s the slowest of all the major frameworks by a large margin. For most models, you may see a 20% or 30% speedup with JAX. For some models—especially large ones—you may even see a 3× or a 5× speedup with JAX, even after using `torch.compile()`.

While it is possible to make PyTorch code faster via `torch.compile()`, the PyTorch Dynamo compiler remains at this time (in 2025) quite ineffective and full of trapdoors. As a result, only a very small pe

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

### PDF p.87 (printed 63) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Introduction to TensorFlow
TensorFlow is a Python-based open source machine learning framework developed primarily by Google. Its initial release was in November 2015, followed by a v1 release in February 2017, and a v2 release in October 2019. TensorFlow is heavily used in production-grade machine learning applications across the industry.

2 Note that PyTorch is a bit of an intermediate case: while it is mainly a lower-level framework, it also includes its own layers and its own optimizers. However, if you use PyTorch in conjunction with Keras, then you will only interact with low-level PyTorch APIs such as tensor operations.

### PDF p.87 (printed 63) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › How these frameworks relate to each other
Keras, TensorFlow, PyTorch, and JAX don’t all have the same feature set and aren’t interchangeable. They have some overlap, but to a large extent, they serve different roles for different use cases. The biggest difference is between Keras and the three others. Keras is a high-level framework, while the others are lower level. Imagine building a house. Keras is like a prefabricated building kit: it provides a streamlined interface for setting up and training neural networks. In contrast, TensorFlow, PyTorch, and JAX are like the raw materials used in construction.

As you saw in the previous chapters, training a neural network revolves around the following concepts:

First, low-level tensor manipulation—The infrastructure that underlies all modern machine learning. This translates to low-level APIs found in TensorFlow, PyTorch,2 and JAX: – Tensors, including special tensors that store the network’s state (variables) – Tensor operations such as addition, `relu`, or `matmul` – Backpropagation, a way to compute the gradient of mathematical expressions

Second, high-level deep learning concepts—This translates to Keras APIs: – Layers, which are combined into a model – A loss function, w

### PDF p.105 (printed 81) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › What makes the PyTorch approach unique
Compared to TensorFlow and JAX, which we will cover next, what makes PyTorch stand out? Why should you use it or not use it?

Here are PyTorch’s two key strengths:

PyTorch code executes eagerly by default, making it easy to debug. Note that this is also the case for TensorFlow code and JAX code, but a big difference is that PyTorch is generally intended to be run eagerly at all times, whereas any serious TensorFlow or JAX project will inevitably need compilation at some point, which can significantly hurt the debugging experience.

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Tensor operations: Doing math in PyTorch
Math in PyTorch works just the same as math in NumPy or TensorFlow, although much like TensorFlow, the PyTorch API often diverges in subtle ways from the NumPy API.

Listing 3.25 A few basic math operations in PyTorch

### PDF p.127 (printed 103) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Summary
TensorFlow, PyTorch, and JAX are three popular low-level frameworks for numerical computation and autodifferentiation. They all have their own way of doing things and their own strengths and weaknesses.

Keras is a high-level API for building and training neural networks. It can be used with either TensorFlow, PyTorch, or JAX—just pick the backend you like best.

The central class of Keras is the `Layer`. A layer encapsulates some weights and some computation. Layers are assembled into models.

Before you start training a model, you need to pick an optimizer, a loss, and some metrics, which you specify via the `model.compile()` method.

To train a model, you can use the `fit()` method, which runs mini-batch gradient descent for you. You can also use it to monitor your loss and metrics on validation data, a set of inputs that the model doesn’t see during training.

Once your model is trained, you can use the `model.predict()` method to generate predictions on new inputs.

### PDF p.86 (printed 62) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.1 A brief history of deep learning frameworks › 
world—in particular, Torch 7 (Lua-based) and Caffe (C++-based). Keras launched in early 2015 as a higher-level, easier-to-use deep learning library powered by Theano, and it quickly gained traction with the few thousands of people who were into deep learning at the time.

Then in late 2015, Google launched TensorFlow, which took many of the key ideas from Theano and added support for large-scale distributed computation. The release of TensorFlow was a watershed moment that precipitated deep learning in the mainstream developer zeitgeist. Keras immediately added support for TensorFlow. By mid-2016, over half of all TensorFlow users were using it through Keras. In response to TensorFlow, Meta (named Facebook at the time) launched PyTorch about one year later, taking ideas from Chainer (a niche but innovative framework launched in mid-2015, now long dead) and NumPy-Autograd, a CPU-only autodifferentiation library for NumPy released by Maclaurin et al. in 2014. Meanwhile, Google released TPUs as an alternative to GPUs, alongside XLA, a high-performance compiler developed to enable TensorFlow to run on TPUs. A few years later, at Google, Matthew Johnson—one of the developers who worked 
