# Generation bundle for page_085  (PDF page 85, printed page 61)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.1 A brief history of deep learning frameworks
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_085.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_085.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p085_e03] (heading_section) 3.1
[p085_e04] (heading_section) A brief history of deep learning frameworks
[p085_e05] (body) In the real world, you’re not going to be writing low-level code from scratch like we did at the end of chapter 2. Instead, you’re going to use a framework. Besides Keras, the main deep learning frameworks today are JAX, TensorFlow, and PyTorch. This book will teach you about all four.
[p085_e06] (body) If you’re just getting started with deep learning, it may seem like all these frameworks have been here forever. In reality, they’re all quite recent, with Keras being the oldest among the four (launched in March 2015). The ideas behind these frameworks, however, have a long history—the first paper about automatic differentiation was published in 1964.1 All these frameworks combine three key features:
[p085_e07] (bullet) A way to compute gradients for arbitrary differentiable functions (automatic differentiation)
[p085_e08] (bullet) A way to run tensor computations on CPUs and GPUs (and possibly even on other specialized deep learning hardware)
[p085_e09] (bullet) A way to distribute computation across multiple devices or multiple computers, such as multiple GPUs on one computer, or even multiple GPUs across multiple separate computers
[p085_e10] (body) Together, these three simple features unlock all modern deep learning.
[p085_e11] (body) It took a long time for the field to develop robust solutions for all three problems and package those solutions in a reusable form. Since its inception in the 1960s and until the 2000s, autodifferentiation had no practical applications in machine learning—folks who worked with neural networks simply wrote their own gradient logic by hand, usually in a language like C++. Meanwhile, GPU programming was all but impossible. Things started to slowly change in the late 2000s. First, Python and its ecosystem were slowly rising in popularity in the scientific community, gaining traction over MATLAB and C++. Second, NVIDIA released CUDA in 2006, unlocking the possibility of building neural networks that could run on consumer GPUs. The initial focus on CUDA was on physics simulation rather than machine learning, but that didn’t stop machine learning researchers from starting to implement CUDA-based neural networks from 2009 onward. They were typically one-off implementations that ran on a single GPU without any autodifferentiation. The first framework to enable autodifferentiation and GPU computation to train deep learning models was Theano, circa 2009. Theano is the conceptual ancestor of all modern deep learning tools. It started getting good traction in the machine learning research community in 2013–2014, after the results of the ImageNet 2012 competition ignited the world’s interest in deep learning. Around the same time, a few other GPU-enabled deep learning libraries started gaining popularity in the computer vision
[p085_e12] (footnote) 1 R. E. Wengert, “A Simple Automatic Derivative Evaluation Program,” Communications of the ACM, 7 no. 8 (1964).

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

## Previous page (PDF 84) text, for continuity
Introduction to

TensorFlow, PyTorch,

JAX, and Keras

This chapter covers

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

## Next page (PDF 86) text, for continuity
world—in particular, Torch 7 (Lua-based) and Caffe (C++-based). Keras launched in early 2015 as a higher-level, easier-to-use deep learning library powered by Theano, and it quickly gained traction with the few thousands of people who were into deep learning at the time.

Then in late 2015, Google launched TensorFlow, which took many of the key ideas from Theano and added support for large-scale distributed computation. The release of TensorFlow was a watershed moment that precipitated deep learning in the mainstream developer zeitgeist. Keras immediately added support for TensorFlow. By mid-2016, over half of all TensorFlow users were using it through Keras. In response to TensorFlow, Meta (named Facebook at the time) launched PyTorch about one year later, taking ideas from Chainer (a niche but innovative framework launched in mid-2015, now long dead) and NumPy-Autograd, a CPU-only autodifferentiation library for NumPy released by Maclaurin et al. in 2014. Meanwhile, Google released TPUs as an alternative to GPUs, alongside XLA, a high-performance compiler developed to enable TensorFlow to run on TPUs. A few years later, at Google, Matthew Johnson—one of the developers who worked on NumPy-Autograd—released JAX as an alternative way to use autodifferentiation with XLA. JAX quickly gained traction with researchers thanks to its minimalistic API and high scalability. Today, Keras, TensorFlow, PyTorch, and JAX are the top frameworks in the deep learning world. Looking back on this chaotic history, we can ask, What’s next? Will a new framework arise tomorrow? Will we switch to a new programming language or a new hardware platform? If you ask me, three things today are certain:

Python has won. Its machine learning and data science ecosystem simply has too much momentum at t …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.87 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.9 — contents › 
- PDF p.623 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.21 — about this book › How this book is organized: A road map
- PDF p.115 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.215 — 7 A deep dive on Keras › 7.1 A spectrum of workflows
- PDF p.98 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.623 (printed 599) — 20 Conclusions › 20.1 Key concepts in review › 
Moore’s law applied to storage media. Today, state-of-the-art language models are trained on a large fraction of the entire internet.

The availability of fast, highly parallel computation hardware at a low price, especially the GPUs produced by NVIDIA—first gaming GPUs and then chips designed from the ground up for deep learning. Early on, NVIDIA CEO Jensen Huang took note of the deep learning boom and decided to bet the company’s future on it, which paid off in a big way.

A complex stack of software layers that makes this computational power available to humans: the CUDA language, frameworks like TensorFlow, JAX, and PyTorch that do automatic differentiation, and Keras, which makes deep learning accessible to most people.

In the future, deep learning will not be used only by specialists such as researchers, graduate students, and engineers with an academic profile; it will be a tool in the toolbox of every developer, much like web technology today. Everyone needs to build intelligent apps: just as every business today needs a website, every product will need to intelligently make sense of user-generated data. Bringing about this future will require us to build tools that make d

### PDF p.34 (printed 10) — 1 What is deep learning? › 1.7 What makes deep learning different › What makes deep learning different
Is there anything special about deep neural networks that makes them the “right” approach for companies to invest in and for researchers to flock to? Will we still be using deep neural networks in 20 years?

Deep learning has several properties that justify its status as an AI revolution, and it’s here to stay. We may not be using neural networks many decades from now, but whatever we use will directly inherit from modern deep learning and its core concepts. These important properties can be broadly sorted into three categories:

Simplicity—Deep learning makes problem solving much easier, because it automates what used to be the most crucial step in a machine learning workflow: feature engineering. Previous machine learning techniques—shallow learning—only involved transforming the input data into one or two successive representation spaces, which wasn’t expressive enough for most problems. As such, humans had to go to great lengths to make the initial input data more amenable to processing by these methods: they had to manually engineer good representations for their data. This is called feature engineering. Deep learning, on the other hand, completely automates this step: with de

### PDF p.87 (printed 63) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › How these frameworks relate to each other
Keras, TensorFlow, PyTorch, and JAX don’t all have the same feature set and aren’t interchangeable. They have some overlap, but to a large extent, they serve different roles for different use cases. The biggest difference is between Keras and the three others. Keras is a high-level framework, while the others are lower level. Imagine building a house. Keras is like a prefabricated building kit: it provides a streamlined interface for setting up and training neural networks. In contrast, TensorFlow, PyTorch, and JAX are like the raw materials used in construction.

As you saw in the previous chapters, training a neural network revolves around the following concepts:

First, low-level tensor manipulation—The infrastructure that underlies all modern machine learning. This translates to low-level APIs found in TensorFlow, PyTorch,2 and JAX: – Tensors, including special tensors that store the network’s state (variables) – Tensor operations such as addition, `relu`, or `matmul` – Backpropagation, a way to compute the gradient of mathematical expressions

Second, high-level deep learning concepts—This translates to Keras APIs: – Layers, which are combined into a model – A loss function, w

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › 
If you’re a deep learning researcher or practitioner looking to get started with the Keras framework, you’ll find this book to be the ideal Keras crash course.

If you’re a graduate student studying deep learning in a formal setting, you’ll find this book to be a practical complement to your education, helping you build intuition around the behavior of deep neural networks and familiarizing you with key best practices.

Even technically minded people who don’t code regularly will find this book useful as an introduction to both basic and advanced deep learning concepts.

To understand the code examples, you’ll need reasonable Python proficiency. You don’t need previous experience with machine learning or deep learning: this book covers, from scratch, all the necessary basics. You don’t need an advanced mathematics background either—high-school-level mathematics should suffice to follow along.

### PDF p.215 (printed 191) — 7 A deep dive on Keras › 7.1 A spectrum of workflows › A spectrum of workflows
As such, there’s not a single “true” way of using Keras. Rather, Keras offers a spectrum of workflows, from the very simple to the very flexible. There are different ways to build Keras models, and different ways to train them, answering different needs. For instance, you have a range of ways to build models and an array of ways to train them, each representing a certain tradeoff between usability and flexibility. You could be using Keras like you would use scikit-learn—just calling `fit()` and letting the framework do its thing—or you could be using it like NumPy—taking full control of every little detail. Because all these workflows are based on shared APIs, such as `Layer` and `Model`, components from any workflow can be used in any other workflow: they can all talk to each other. This means that everything you’re learning now as you’re getting started will still be relevant once you’ve become an expert. You can get started easily and then gradually dive into workflows where you’re writing more and more logic from scratch. You won’t have to switch to an entirely different framework as you go from student to researcher, or from data scientist to deep learning engineer. This philo

### PDF p.18 (printed xviii) — preface ›  › 
of Keras, and over the last decade, the Keras developer community has made remarkable progress in this area. We’ve put deep learning into the hands of millions of people, who, in turn, are using it to solve problems that were, until recently, thought to be unsolvable.

The book you’re holding is another step on the way toward broadening access to the field. We aim to make the concepts behind deep learning and their implementation as approachable as possible. Doing so doesn’t require watering anything down; we believe that there are no difficult ideas in deep learning. This book will start with the very basics of the field and, layer by layer, build up to the cutting-edge generative AI models being deployed today. This is the third edition of Deep Learning with Python. In an effort to make this content as broadly available as possible, we are making this edition available for free, online at https://deeplearningwithpython.io/. We hope you will consider purchasing a physical copy to support the project and read what we consider to be the best presentation of the content. This third edition amounts to a complete rewrite of the entire book, with a broader introduction to today’s popula

### PDF p.115 (printed 91) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Picking a backend framework
Keras can be used together with JAX, TensorFlow, or PyTorch. They’re the “backend frameworks” of Keras. Through these backend frameworks, Keras can run on top of different types of hardware (see figure 3.4)—GPU, TPU, or plain CPU—can be seamlessly scaled to thousands of machines, and can be deployed to a variety of platforms.

Backend frameworks are pluggable: you can switch to a different backend framework after you’ve written some Keras code. You aren’t locked into a single framework and a single ecosystem—you can move your models from JAX to TensorFlow to PyTorch depending on your current needs. For instance, when you develop a Keras model, you could debug it with PyTorch, train it on TPU with JAX for maximum efficiency, and finally run inference with the excellent tooling from the TensorFlow ecosystem.

Figure 3.4 Keras and its backends. A backend is a low-level tensor-computing platform; Keras is a high-level deep learning API.

The default backend for Keras right now is TensorFlow, so if you run `import` `keras` in a fresh environment, without having configured anything, you will be running on top of TensorFlow. There are two ways to pick a different backend:

Set the environ

### PDF p.9 (printed ix) — contents ›  › 
2

The mathematical building blocks of neural networks 16 2.1

A first look at a neural network 17 2.2 Data representations for neural networks 21

Scalars (rank-0 tensors) 22 ■ Vectors (rank-1 tensors) 22 Matrices (rank-2 tensors) 22 ■ Rank-3 tensors and higher-rank tensors 23 ■ Key attributes 23 ■ Manipulating tensors in NumPy 25 ■ The notion of data batches 25 Real-world examples of data tensors 26

2.3

The gears of neural networks: Tensor operations 28

Element-wise operations 29 ■ Broadcasting 30 Tensor product 32 ■ Tensor reshaping 34 ■ Geometric interpretation of tensor operations 35 ■ A geometric interpretation of deep learning 38

2.4

The engine of neural networks: Gradient-based optimization 39

What’s a derivative? 41 ■ Derivative of a tensor operation: The gradient 42 ■ Stochastic gradient descent 43 ■ Chaining derivatives: The Backpropagation algorithm 46

2.5

Looking back at our first example 51

Reimplementing our first example from scratch 53 Running one training step 55 ■ The full training loop 57 Evaluating the model 58

3

Introduction to TensorFlow, PyTorch, JAX, and Keras 60 3.1

A brief history of deep learning frameworks 61 3.2 How these frameworks relate 
