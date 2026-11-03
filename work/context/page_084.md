# Generation bundle for page_084  (PDF page 84, printed page 60)

Page kind: chapter_opener
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_084.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_084.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p084_e01] (chapter_number) 3
[p084_e02] (chapter_title) Introduction to
[p084_e03] (chapter_title) TensorFlow, PyTorch,
[p084_e04] (chapter_title) JAX, and Keras
[p084_e05] (chapter_covers) This chapter covers
[p084_e06] (bullet) A closer look at all major deep learning
[p084_e07] (bullet) frameworks and their relationships
[p084_e08] (bullet) An overview of how core deep learning concepts
[p084_e09] (bullet) translate to code across all frameworks
[p084_e10] (body) This chapter is meant to give you everything you need to start doing deep learning in practice. First, you’ll get familiar with three popular deep learning frameworks that can be used with Keras:
[p084_e11] (bullet) TensorFlow (https://tensorflow.org)
[p084_e12] (bullet) PyTorch (https://pytorch.org/)
[p084_e13] (bullet) JAX (https://jax.readthedocs.io/)
[p084_e14] (body) Then, building on top of the first contact you’ve had with Keras in chapter 2, we’ll review the core components of neural networks and how they translate to Keras APIs.
[p084_e15] (body) By the end of this chapter, you’ll be ready to move on to practical, real-world applications—which will start with chapter 4.

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

## Previous page (PDF 83) text, for continuity
move is defined by the learning rate) in the opposite direction from the gradient.

This is called mini-batch gradient descent.

The entire learning process is made possible by the fact that all tensor operations in neural networks are differentiable, and thus it’s possible to apply the chain rule of derivation to find the gradient function mapping the current parameters and current batch of data to a gradient value. This is called backpropagation.

Two key concepts you’ll see frequently in future chapters are loss and optimizers. These are the two things you need to define before you begin feeding data into a model: – The loss is the quantity you’ll attempt to minimize during training, so it should represent a measure of success for the task you’re trying to solve. – The optimizer specifies the exact way in which the gradient of the loss will be used to update parameters: for instance, it could be the RMSProp optimizer, SGD with momentum, and so on.

## Next page (PDF 85) text, for continuity
3.1

A brief history of deep learning frameworks

In the real world, you’re not going to be writing low-level code from scratch like we did at the end of chapter 2. Instead, you’re going to use a framework. Besides Keras, the main deep learning frameworks today are JAX, TensorFlow, and PyTorch. This book will teach you about all four.

If you’re just getting started with deep learning, it may seem like all these frameworks have been here forever. In reality, they’re all quite recent, with Keras being the oldest among the four (launched in March 2015). The ideas behind these frameworks, however, have a long history—the first paper about automatic differentiation was published in 1964.1 All these frameworks combine three key features:

A way to compute gradients for arbitrary differentiable functions (automatic differentiation)

A way to run tensor computations on CPUs and GPUs (and possibly even on other specialized deep learning hardware)

A way to distribute computation across multiple devices or multiple computers, such as multiple GPUs on one computer, or even multiple GPUs across multiple separate computers

Together, these three simple features unlock all modern deep learning.

It took a long time for the field to develop robust solutions for all three problems and package those solutions in a reusable form. Since its inception in the 1960s and until the 2000s, autodifferentiation had no practical applications in machine learning—folks who worked with neural networks simply wrote their own gradient logic by hand, usually in a language like C++. Meanwhile, GPU programming was all but impossible. Things started to slowly change in the late 2000s. First, Python and its ecosystem were slowly rising in popularity in the scientific community, gaining traction over MATLAB …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.87 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.115 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.215 — 7 A deep dive on Keras › 7.1 A spectrum of workflows
- PDF p.127 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.116 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.86 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.1 A brief history of deep learning frameworks
- PDF p.41 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.87 (printed 63) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › How these frameworks relate to each other
Keras, TensorFlow, PyTorch, and JAX don’t all have the same feature set and aren’t interchangeable. They have some overlap, but to a large extent, they serve different roles for different use cases. The biggest difference is between Keras and the three others. Keras is a high-level framework, while the others are lower level. Imagine building a house. Keras is like a prefabricated building kit: it provides a streamlined interface for setting up and training neural networks. In contrast, TensorFlow, PyTorch, and JAX are like the raw materials used in construction.

As you saw in the previous chapters, training a neural network revolves around the following concepts:

First, low-level tensor manipulation—The infrastructure that underlies all modern machine learning. This translates to low-level APIs found in TensorFlow, PyTorch,2 and JAX: – Tensors, including special tensors that store the network’s state (variables) – Tensor operations such as addition, `relu`, or `matmul` – Backpropagation, a way to compute the gradient of mathematical expressions

Second, high-level deep learning concepts—This translates to Keras APIs: – Layers, which are combined into a model – A loss function, w

### PDF p.215 (printed 191) — 7 A deep dive on Keras › 7.1 A spectrum of workflows › 
use the gradient APIs in TensorFlow, JAX, and PyTorch to implement a step-by-step training loop.

In the coming chapters, we’ll dig into computer vision, timeseries forecasting, natural language processing, and generative deep learning. These complex applications will require much more than a `Sequential` architecture and the default `fit()` loop. So let’s first turn you into a Keras expert! In this chapter, you’ll get a complete overview of the key ways to work with Keras APIs: everything you’re going to need to handle the advanced deep learning use cases you’ll encounter next.

### PDF p.115 (printed 91) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Picking a backend framework
Keras can be used together with JAX, TensorFlow, or PyTorch. They’re the “backend frameworks” of Keras. Through these backend frameworks, Keras can run on top of different types of hardware (see figure 3.4)—GPU, TPU, or plain CPU—can be seamlessly scaled to thousands of machines, and can be deployed to a variety of platforms.

Backend frameworks are pluggable: you can switch to a different backend framework after you’ve written some Keras code. You aren’t locked into a single framework and a single ecosystem—you can move your models from JAX to TensorFlow to PyTorch depending on your current needs. For instance, when you develop a Keras model, you could debug it with PyTorch, train it on TPU with JAX for maximum efficiency, and finally run inference with the excellent tooling from the TensorFlow ecosystem.

Figure 3.4 Keras and its backends. A backend is a low-level tensor-computing platform; Keras is a high-level deep learning API.

The default backend for Keras right now is TensorFlow, so if you run `import` `keras` in a fresh environment, without having configured anything, you will be running on top of TensorFlow. There are two ways to pick a different backend:

Set the environ

### PDF p.127 (printed 103) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Summary
TensorFlow, PyTorch, and JAX are three popular low-level frameworks for numerical computation and autodifferentiation. They all have their own way of doing things and their own strengths and weaknesses.

Keras is a high-level API for building and training neural networks. It can be used with either TensorFlow, PyTorch, or JAX—just pick the backend you like best.

The central class of Keras is the `Layer`. A layer encapsulates some weights and some computation. Layers are assembled into models.

Before you start training a model, you need to pick an optimizer, a loss, and some metrics, which you specify via the `model.compile()` method.

To train a model, you can use the `fit()` method, which runs mini-batch gradient descent for you. You can also use it to monitor your loss and metrics on validation data, a set of inputs that the model doesn’t see during training.

Once your model is trained, you can use the `model.predict()` method to generate predictions on new inputs.

### PDF p.127 (printed 103) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
For now, this is all you need to know about Keras models. At this point, you are ready to move on to solving real-world machine problems with Keras, in the next chapter.

### PDF p.87 (printed 63) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › How these frameworks relate to each other
Now that you have a clearer understanding of how all these frameworks came to be and how they relate to each other, let’s dive into what it’s like to work with them. We’ll cover them in chronological order: TensorFlow first, then PyTorch, and finally JAX.

### PDF p.116 (printed 92) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
NOTE When configuring the Keras backend, you should use the string `"torch"` to refer to the PyTorch backend, rather than the string `"pytorch"`, which would be invalid. This is because the PyTorch package name is `torch` (as in `import ` `torch` or `pip` `install` `torch`). Now, you may ask, which backend should I be picking? It’s really your own choice: all Keras code examples in the rest of the book will be compatible with all three backends. If the need for backend-specific code arises (as in chapter 7, for instance), we will show you all three versions—TensorFlow, PyTorch, JAX. If you have no particular backend preference, our personal recommendation is JAX. It’s usually the most performant backend. Once your backend is configured, you can start actually building and training Keras models. Let’s take a look.

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
