# Generation bundle for page_087  (PDF page 87, printed page 63)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.3 Introduction to TensorFlow
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_087.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_087.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p087_e03] (heading_section) 3.2
[p087_e04] (heading_section) How these frameworks relate to each other
[p087_e05] (body) Keras, TensorFlow, PyTorch, and JAX don’t all have the same feature set and aren’t interchangeable. They have some overlap, but to a large extent, they serve different roles for different use cases. The biggest difference is between Keras and the three others. Keras is a high-level framework, while the others are lower level. Imagine building a house. Keras is like a prefabricated building kit: it provides a streamlined interface for setting up and training neural networks. In contrast, TensorFlow, PyTorch, and JAX are like the raw materials used in construction.
[p087_e06] (body) As you saw in the previous chapters, training a neural network revolves around the following concepts:
[p087_e07] (bullet) First, low-level tensor manipulation—The infrastructure that underlies all modern machine learning. This translates to low-level APIs found in TensorFlow, PyTorch,2 and JAX: – Tensors, including special tensors that store the network’s state (variables) – Tensor operations such as addition, `relu`, or `matmul` – Backpropagation, a way to compute the gradient of mathematical expressions
[p087_e08] (bullet) Second, high-level deep learning concepts—This translates to Keras APIs: – Layers, which are combined into a model – A loss function, which defines the feedback signal used for learning – An optimizer, which determines how learning proceeds – Metrics to evaluate model performance, such as accuracy – A training loop that performs mini-batch stochastic gradient descent
[p087_e09] (body) Further, Keras is unique in that it isn’t a fully standalone framework. It needs a backend engine to run, much like a prefabricated house-building kit needs to source building materials from somewhere. TensorFlow, PyTorch, and JAX can all be used as Keras backends. In addition, Keras can run on NumPy, but since NumPy does not provide an API for gradients, Keras workflows on NumPy are restricted to making predictions from a model—training is impossible.
[p087_e10] (body) Now that you have a clearer understanding of how all these frameworks came to be and how they relate to each other, let’s dive into what it’s like to work with them. We’ll cover them in chronological order: TensorFlow first, then PyTorch, and finally JAX.
[p087_e11] (heading_section) 3.3
[p087_e12] (heading_section) Introduction to TensorFlow
[p087_e13] (body) TensorFlow is a Python-based open source machine learning framework developed primarily by Google. Its initial release was in November 2015, followed by a v1 release in February 2017, and a v2 release in October 2019. TensorFlow is heavily used in production-grade machine learning applications across the industry.
[p087_e14] (footnote) 2 Note that PyTorch is a bit of an intermediate case: while it is mainly a lower-level framework, it also includes its own layers and its own optimizers. However, if you use PyTorch in conjunction with Keras, then you will only interact with low-level PyTorch APIs such as tensor operations.

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

## Previous page (PDF 86) text, for continuity
world—in particular, Torch 7 (Lua-based) and Caffe (C++-based). Keras launched in early 2015 as a higher-level, easier-to-use deep learning library powered by Theano, and it quickly gained traction with the few thousands of people who were into deep learning at the time.

Then in late 2015, Google launched TensorFlow, which took many of the key ideas from Theano and added support for large-scale distributed computation. The release of TensorFlow was a watershed moment that precipitated deep learning in the mainstream developer zeitgeist. Keras immediately added support for TensorFlow. By mid-2016, over half of all TensorFlow users were using it through Keras. In response to TensorFlow, Meta (named Facebook at the time) launched PyTorch about one year later, taking ideas from Chainer (a niche but innovative framework launched in mid-2015, now long dead) and NumPy-Autograd, a CPU-only autodifferentiation library for NumPy released by Maclaurin et al. in 2014. Meanwhile, Google released TPUs as an alternative to GPUs, alongside XLA, a high-performance compiler developed to enable TensorFlow to run on TPUs. A few years later, at Google, Matthew Johnson—one of the developers who worked on NumPy-Autograd—released JAX as an alternative way to use autodifferentiation with XLA. JAX quickly gained traction with researchers thanks to its minimalistic API and high scalability. Today, Keras, TensorFlow, PyTorch, and JAX are the top frameworks in the deep learning world. Looking back on this chaotic history, we can ask, What’s next? Will a new framework arise tomorrow? Will we switch to a new programming language or a new hardware platform? If you ask me, three things today are certain:

Python has won. Its machine learning and data science ecosystem simply has too much momentum at t …

## Next page (PDF 88) text, for continuity
It’s important to keep in mind that TensorFlow is more than a single library. It’s really a platform, home to a vast ecosystem of components, some developed by Google, some developed by third parties. For instance, there’s TFX for industry-strength machine learning workflow management, TF-Serving for production deployment, the TF Optimization Toolkit for model quantization and pruning, and TFLite and MediaPipe for mobile application deployment. Together, these components cover a very wide range of use cases, from cutting-edge research to large-scale production applications.

3.3.1

First steps with TensorFlow

Over the next paragraphs, you’ll get familiar with all the basics of TensorFlow. We’ll cover the following key concepts:

Tensors and variables

Numerical operations in TensorFlow

Computing gradients with a `GradientTape`

Making TensorFlow functions fast by using just-in-time compilation

We’ll then conclude the introduction with an end-to-end example: a pure-TensorFlow implementation of linear regression.

Let’s get those tensors flowing.

Tensors and variables in TensorFlow

To do anything in TensorFlow, we’re going to need some tensors. There are a few different ways you can create them.

Constant tensors

Tensors need to be created with some initial value, so common ways to create tensors are via `tf.ones` (equivalent to `np.ones`) and `tf.zeros` (equivalent to `np.zeros`). You can also create a tensor from Python or NumPy values using `tf.constant`.

Listing 3.1 All-ones or all-zeros tensors

Random tensors

You can also create tensors filled with random values via one of the methods of the `tf.random` submodule (equivalent to the `np.random` submodule).

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.115 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.98 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.127 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.99 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.84 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 
- PDF p.116 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.85 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.1 A brief history of deep learning frameworks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.127 (printed 103) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Summary
TensorFlow, PyTorch, and JAX are three popular low-level frameworks for numerical computation and autodifferentiation. They all have their own way of doing things and their own strengths and weaknesses.

Keras is a high-level API for building and training neural networks. It can be used with either TensorFlow, PyTorch, or JAX—just pick the backend you like best.

The central class of Keras is the `Layer`. A layer encapsulates some weights and some computation. Layers are assembled into models.

Before you start training a model, you need to pick an optimizer, a loss, and some metrics, which you specify via the `model.compile()` method.

To train a model, you can use the `fit()` method, which runs mini-batch gradient descent for you. You can also use it to monitor your loss and metrics on validation data, a set of inputs that the model doesn’t see during training.

Once your model is trained, you can use the `model.predict()` method to generate predictions on new inputs.

### PDF p.115 (printed 91) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Picking a backend framework
Keras can be used together with JAX, TensorFlow, or PyTorch. They’re the “backend frameworks” of Keras. Through these backend frameworks, Keras can run on top of different types of hardware (see figure 3.4)—GPU, TPU, or plain CPU—can be seamlessly scaled to thousands of machines, and can be deployed to a variety of platforms.

Backend frameworks are pluggable: you can switch to a different backend framework after you’ve written some Keras code. You aren’t locked into a single framework and a single ecosystem—you can move your models from JAX to TensorFlow to PyTorch depending on your current needs. For instance, when you develop a Keras model, you could debug it with PyTorch, train it on TPU with JAX for maximum efficiency, and finally run inference with the excellent tooling from the TensorFlow ecosystem.

Figure 3.4 Keras and its backends. A backend is a low-level tensor-computing platform; Keras is a high-level deep learning API.

The default backend for Keras right now is TensorFlow, so if you run `import` `keras` in a fresh environment, without having configured anything, you will be running on top of TensorFlow. There are two ways to pick a different backend:

Set the environ

### PDF p.99 (printed 75) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
The PyTorch API is higher level than that of TensorFlow and JAX: it includes layers and optimizers, like Keras. These layers and optimizers are compatible with Keras workflows when you use Keras with the PyTorch backend.

### PDF p.84 (printed 60) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras ›  › This chapter covers
A closer look at all major deep learning

frameworks and their relationships

An overview of how core deep learning concepts

translate to code across all frameworks

This chapter is meant to give you everything you need to start doing deep learning in practice. First, you’ll get familiar with three popular deep learning frameworks that can be used with Keras:

TensorFlow (https://tensorflow.org)

PyTorch (https://pytorch.org/)

JAX (https://jax.readthedocs.io/)

Then, building on top of the first contact you’ve had with Keras in chapter 2, we’ll review the core components of neural networks and how they translate to Keras APIs.

By the end of this chapter, you’ll be ready to move on to practical, real-world applications—which will start with chapter 4.

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

### PDF p.85 (printed 61) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.1 A brief history of deep learning frameworks › A brief history of deep learning frameworks
In the real world, you’re not going to be writing low-level code from scratch like we did at the end of chapter 2. Instead, you’re going to use a framework. Besides Keras, the main deep learning frameworks today are JAX, TensorFlow, and PyTorch. This book will teach you about all four.

If you’re just getting started with deep learning, it may seem like all these frameworks have been here forever. In reality, they’re all quite recent, with Keras being the oldest among the four (launched in March 2015). The ideas behind these frameworks, however, have a long history—the first paper about automatic differentiation was published in 1964.1 All these frameworks combine three key features:

A way to compute gradients for arbitrary differentiable functions (automatic differentiation)

A way to run tensor computations on CPUs and GPUs (and possibly even on other specialized deep learning hardware)

A way to distribute computation across multiple devices or multiple computers, such as multiple GPUs on one computer, or even multiple GPUs across multiple separate computers

Together, these three simple features unlock all modern deep learning.

### PDF p.98 (printed 74) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › What makes the TensorFlow approach unique
You’re now familiar with all the basic APIs that underlie TensorFlow-based workflows, and you’re about to dive into more frameworks—in particular, PyTorch and JAX. What makes working with TensorFlow different from working with any other framework? When should you use TensorFlow, and when could you use something else?

If you ask us, here are the main benefits of TensorFlow:

Thanks to graph mode and XLA compilation, it’s fast. It’s usually significantly faster than PyTorch and NumPy, though JAX is often even faster.

It is extremely feature complete. Unique among all frameworks, it has support for string tensors as well as “ragged tensors” (tensors where different entries may have different dimensions—very useful for handling sequences without requiring to pad them to a shared length). It also has outstanding support for data preprocessing, via the highly performant `tf.data` API. `tf.data` is so good that even JAX recommends it for data preprocessing. Whatever you need to do, TensorFlow has a solution for it.

Its ecosystem for production deployment is the most mature among all frameworks, especially when it comes to deploying on mobile or in the browser.

However, TensorFlow also

### PDF p.116 (printed 92) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
NOTE When configuring the Keras backend, you should use the string `"torch"` to refer to the PyTorch backend, rather than the string `"pytorch"`, which would be invalid. This is because the PyTorch package name is `torch` (as in `import ` `torch` or `pip` `install` `torch`). Now, you may ask, which backend should I be picking? It’s really your own choice: all Keras code examples in the rest of the book will be compatible with all three backends. If the need for backend-specific code arises (as in chapter 7, for instance), we will show you all three versions—TensorFlow, PyTorch, JAX. If you have no particular backend preference, our personal recommendation is JAX. It’s usually the most performant backend. Once your backend is configured, you can start actually building and training Keras models. Let’s take a look.
