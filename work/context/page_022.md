# Generation bundle for page_022  (PDF page 22, printed page xxii)

Page kind: front_matter
Chapter: about this book
Section: liveBook discussion forum
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_022.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_022.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p022_e04] (body) and identifying new research directions, and then review the concepts presented across the book.
[p022_e05] (heading_section) About the code
[p022_e06] (body) This book is an example of literate programming—it is simultaneously a prose explanation of deep learning concepts and a runnable Python program. Starting with chapter 2, each chapter will be interspersed with code listings that can be run as a Jupyter notebook, an interactive notebook for running Python code and visualizing data. We strongly suggest running, modifying, and playing with the examples in this book as you read to build your own intuitions on the concepts presented.
[p022_e07] (body) All code is written in the Python language with the deep learning library Keras. Keras can be run on top of TensorFlow, PyTorch, and JAX, the most popular low-level deep learning frameworks as of 2025. All code can be run on a local machine or directly in the browser using Google Colab, a hosted environment for Jupyter notebooks. You can get executable snippets of code from the liveBook version of this book, included with your print or eBook purchase, at https://livebook.manning.com/book/ deep-learning-with-python-third-edition. The complete code for the examples in the book is available for download from the Manning website at https://www.manning .com/books/deep-learning-with-python-third-edition and on GitHub, at https:// github.com/fchollet/deep-learning-with-python-notebooks, along with instructions for running the code locally. You can run the code directly through the web version of this book available at https://deeplearningwithpython.io. A Colab link appears at the top of each chapter allowing you to run the chapter’s code in the browser on free, hosted hardware.
[p022_e08] (heading_section) liveBook discussion forum
[p022_e09] (body) Purchase of Deep Learning with Python, Third Edition, includes free access to liveBook, Manning’s online reading platform. Using liveBook’s exclusive discussion features, you can attach comments to the book globally or to specific sections or paragraphs. It’s a snap to make notes for yourself, ask and answer technical questions, and receive help from the authors and other users. To access the forum, go to https://livebook .manning.com/book/deep-learning-with-python-third-edition/discussion.
[p022_e10] (body) Manning’s commitment to our readers is to provide a venue where a meaningful dialogue between individual readers and between readers and the authors can take place. It is not a commitment to any specific amount of participation on the part of the authors, whose contribution to the forum remains voluntary (and unpaid). We suggest you try asking the authors some challenging questions lest their interest stray! The forum and the archives of previous discussions will be accessible from the publisher’s website as long as the book is in print.

## Chapter outline
Chapter: about this book (PDF pages 20-22)
  - Who should read this book (p.20)
  - How this book is organized: A road map (p.21)
  - About the code (p.22)
  - liveBook discussion forum (p.22)

## Previous page (PDF 21) text, for continuity
If you’re a deep learning researcher or practitioner looking to get started with the Keras framework, you’ll find this book to be the ideal Keras crash course.

If you’re a graduate student studying deep learning in a formal setting, you’ll find this book to be a practical complement to your education, helping you build intuition around the behavior of deep neural networks and familiarizing you with key best practices.

Even technically minded people who don’t code regularly will find this book useful as an introduction to both basic and advanced deep learning concepts.

To understand the code examples, you’ll need reasonable Python proficiency. You don’t need previous experience with machine learning or deep learning: this book covers, from scratch, all the necessary basics. You don’t need an advanced mathematics background either—high-school-level mathematics should suffice to follow along.

How this book is organized: A road map

This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter cov …

## Next page (PDF 23) text, for continuity
about the authors

François Chollet has been working with deep learning since it started getting traction in academia in 2012. François is the author of Keras, one of the most popular libraries for deep learning. Keras is used in university classrooms; at companies like Google, Netflix, and Spotify; and in scientific organizations like CERN and NASA. Francois is the co-founder of the Ndea research lab for frontier AI systems and created the ARC-AGI challenge for measuring machine intelligence.

Matthew Watson has been working on machine learning across Google since 2018, including the Gemini model and Google’s open source deep learning ecosystem. He is a core maintainer of Keras, focusing on Keras’s tools for natural language processing. He completed his master’s in computer science at Stanford University, researching procedural modeling techniques at the Stanford Graphics Lab.

xxi

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.41 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.20 — about this book › Who should read this book
- PDF p.18 — preface › 
- PDF p.647 — index › 
- PDF p.650 — index › 
- PDF p.495 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.3 — Praise for the Second Edition › 
- PDF p.632 — 20 Conclusions › 20.5 Final words

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.41 (printed 17) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › Running the code in this book
This book is full of runnable Python code. Each chapter is paired with a Jupyter notebook that contains all of the code from the chapter. A Jupyter notebook is a live Python scratch pad of sorts, where you can interactively run code, graph data, view images, and a lot more. You will gain a lot more practical knowledge from this book if you run and experiment with the code as you read. By far the easiest way to set up a deep learning environment to run these notebooks is Google Colaboratory (or Colab for short), a hosted environment for Jupyter notebooks that has become the industry standard for ML practitioners. With Colab, you can run the code for this book interactively in the browser, connecting to cloud runtimes with configurable hardware. By default, the notebooks in this book will run on Colab’s free GPU runtime. If you would like, you can also run these notebooks locally on your own machine. A GPU is recommended, especially as you get to the larger and more compute-intensive models later in this book. Instructions for running locally and on Colab, along with the code, can be found at https://github.com/fchollet/deep-learning-with-python-notebooks.

### PDF p.18 (printed xviii) — preface ›  › 
of Keras, and over the last decade, the Keras developer community has made remarkable progress in this area. We’ve put deep learning into the hands of millions of people, who, in turn, are using it to solve problems that were, until recently, thought to be unsolvable.

The book you’re holding is another step on the way toward broadening access to the field. We aim to make the concepts behind deep learning and their implementation as approachable as possible. Doing so doesn’t require watering anything down; we believe that there are no difficult ideas in deep learning. This book will start with the very basics of the field and, layer by layer, build up to the cutting-edge generative AI models being deployed today. This is the third edition of Deep Learning with Python. In an effort to make this content as broadly available as possible, we are making this edition available for free, online at https://deeplearningwithpython.io/. We hope you will consider purchasing a physical copy to support the project and read what we consider to be the best presentation of the content. This third edition amounts to a complete rewrite of the entire book, with a broader introduction to today’s popula

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

### PDF p.20 (printed xx) — about this book › Who should read this book › Who should read this book
This book is written for people with some Python programming experience who want to get started with machine learning and deep learning. But this book can also be valuable to many different types of readers:

If you’re a data scientist familiar with machine learning, this book will provide you with a solid, practical introduction to deep learning, the fastest-growing and most significant subfield of machine learning.

xviii

### PDF p.650 (printed 626) — index ›  › 
“

Perfect for anyone interested

DEEP LEARNING with Python Third Edition

in learning by doing from one

of the industry greats.”

Chollet ● Watson

—Anthony Goldbloom

I

n less than a decade, deep learning has changed the world

Founder of Kaggle

—twice. First, Python-based libraries like Keras, Tensor-

Flow, and PyTorch elevated neural networks from lab

“

A sharp, deeply practical

experiments to high-performance production systems deployed

guide that teaches you how

at scale. And now, through Large Language Models and other

to think from fi rst

generative AI tools, deep learning is again transforming busi-

principles to build models

ness and society. In this new edition, Keras creator François

that actually work.”

Chollet invites you into this amazing subject in the fl uid,

mentoring style of a true insider.

—Santiago Valdarrama

Founder of ml.school

Deep Learning with Python, Third Edition makes the concepts

behind deep learning and generative AI understandable and

“

Th e most up-to-date and

approachable. Th is complete rewrite of the bestselling original

includes fresh chapters on transformers, building your own

complete guide to deep

learning you’ll fi

### PDF p.20 (printed xx) — about this book › Who should read this book › about this book
This book was written for anyone who wishes to explore deep learning from scratch or broaden their understanding of deep learning. Whether you’re a practicing machine learning engineer, a software developer, or a college student, you’ll find value in these pages.

You’ll explore deep learning in an approachable way—starting simply and working up to state-of-the-art techniques. We hope you’ll find that this book strikes a balance between intuition, theory, and hands-on practice. It avoids mathematical notation, preferring instead to explain the core ideas of deep learning via functioning code paired with explanations of the underlying principles. You’ll train machine learning models from scratch in a number of different problem domains and learn practical recommendations for writing deep learning programs and deploying them in the real world. After reading this book, you’ll have a solid understanding of what deep learning is, when it’s applicable, and what its limitations are. You’ll be familiar with the standard workflow for approaching and solving machine learning problems, and you’ll know how to address commonly encountered issues.

### PDF p.495 (printed 471) — 16 Text generation › 16.2 Training a mini-GPT › Running the code in this chapter
Generative language models are big and take a lot of compute to run. While we’ve taken pains to make the code in this chapter accessible, this is still the most compute-intensive chapter in this book. If you’d like, you can run everything on the free Colab GPU runtime (a T4 GPU as of this writing), but be prepared to wait! This mini-GPT example will take about 6 hours to train, and you will need to restart your Colab runtime in the middle of the notebook to free up GPU memory before loading a larger pretrained model. A larger GPU will make quicker work of these examples; we developed this example on an A100, which can run this chapter’s code end-to-end in a little over an hour. You can always read through the expensive `fit()` calls and edit down the number of training steps for quick experimentation. And if you are running this a few years down the line, there is a good chance these examples are mere child’s play to modern hardware!

We have 50 shards of text data, each with about 75 MB of raw text. Each line contains a document in the crawl with newlines escaped. Let’s look at a document in our first shard:

### PDF p.215 (printed 191) — 7 A deep dive on Keras › 7.1 A spectrum of workflows › A spectrum of workflows
As such, there’s not a single “true” way of using Keras. Rather, Keras offers a spectrum of workflows, from the very simple to the very flexible. There are different ways to build Keras models, and different ways to train them, answering different needs. For instance, you have a range of ways to build models and an array of ways to train them, each representing a certain tradeoff between usability and flexibility. You could be using Keras like you would use scikit-learn—just calling `fit()` and letting the framework do its thing—or you could be using it like NumPy—taking full control of every little detail. Because all these workflows are based on shared APIs, such as `Layer` and `Model`, components from any workflow can be used in any other workflow: they can all talk to each other. This means that everything you’re learning now as you’re getting started will still be relevant once you’ve become an expert. You can get started easily and then gradually dive into workflows where you’re writing more and more logic from scratch. You won’t have to switch to an entirely different framework as you go from student to researcher, or from data scientist to deep learning engineer. This philo
