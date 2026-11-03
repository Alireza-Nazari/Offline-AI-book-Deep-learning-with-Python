# Generation bundle for page_021  (PDF page 21, printed page xxi)

Page kind: front_matter
Chapter: about this book
Section: How this book is organized: A road map
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_021.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_021.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p021_e04] (bullet) If you’re a deep learning researcher or practitioner looking to get started with the Keras framework, you’ll find this book to be the ideal Keras crash course.
[p021_e05] (bullet) If you’re a graduate student studying deep learning in a formal setting, you’ll find this book to be a practical complement to your education, helping you build intuition around the behavior of deep neural networks and familiarizing you with key best practices.
[p021_e06] (body) Even technically minded people who don’t code regularly will find this book useful as an introduction to both basic and advanced deep learning concepts.
[p021_e07] (body) To understand the code examples, you’ll need reasonable Python proficiency. You don’t need previous experience with machine learning or deep learning: this book covers, from scratch, all the necessary basics. You don’t need an advanced mathematics background either—high-school-level mathematics should suffice to follow along.
[p021_e08] (heading_section) How this book is organized: A road map
[p021_e09] (body) This book contains 20 chapters, organized as follows:
[p021_e10] (bullet) Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.
[p021_e11] (bullet) Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.
[p021_e12] (bullet) Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.
[p021_e13] (bullet) Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an English-to-Spanish translation model. We pay special attention to the Transformer, a popular model architecture particularly well suited to text modeling.
[p021_e14] (bullet) Chapters 16–17: Generative modeling—These chapters are all about using deep learning to create new content. We work with models that can produce new images and text as output. We discuss chatbots like ChatGPT and image generation systems. We both train generative models from scratch and use larger pretrained models trained on large amounts of data over weeks or months.
[p021_e15] (bullet) Chapters 18–20: Conclusions—The final chapters of this book tie together what we have learned. We explore practical engineering concerns with deep learning systems. We dedicate a chapter to exploring potential futures for the field

## Chapter outline
Chapter: about this book (PDF pages 20-22)
  - Who should read this book (p.20)
  - How this book is organized: A road map (p.21)
  - About the code (p.22)
  - liveBook discussion forum (p.22)

## Previous page (PDF 20) text, for continuity
about this book

This book was written for anyone who wishes to explore deep learning from scratch or broaden their understanding of deep learning. Whether you’re a practicing machine learning engineer, a software developer, or a college student, you’ll find value in these pages.

You’ll explore deep learning in an approachable way—starting simply and working up to state-of-the-art techniques. We hope you’ll find that this book strikes a balance between intuition, theory, and hands-on practice. It avoids mathematical notation, preferring instead to explain the core ideas of deep learning via functioning code paired with explanations of the underlying principles. You’ll train machine learning models from scratch in a number of different problem domains and learn practical recommendations for writing deep learning programs and deploying them in the real world. After reading this book, you’ll have a solid understanding of what deep learning is, when it’s applicable, and what its limitations are. You’ll be familiar with the standard workflow for approaching and solving machine learning problems, and you’ll know how to address commonly encountered issues.

Who should read this book

This book is written for people with some Python programming experience who want to get started with machine learning and deep learning. But this book can also be valuable to many different types of readers:

If you’re a data scientist familiar with machine learning, this book will provide you with a solid, practical introduction to deep learning, the fastest-growing and most significant subfield of machine learning.

xviii

## Next page (PDF 22) text, for continuity
and identifying new research directions, and then review the concepts presented across the book.

About the code

This book is an example of literate programming—it is simultaneously a prose explanation of deep learning concepts and a runnable Python program. Starting with chapter 2, each chapter will be interspersed with code listings that can be run as a Jupyter notebook, an interactive notebook for running Python code and visualizing data. We strongly suggest running, modifying, and playing with the examples in this book as you read to build your own intuitions on the concepts presented.

All code is written in the Python language with the deep learning library Keras. Keras can be run on top of TensorFlow, PyTorch, and JAX, the most popular low-level deep learning frameworks as of 2025. All code can be run on a local machine or directly in the browser using Google Colab, a hosted environment for Jupyter notebooks. You can get executable snippets of code from the liveBook version of this book, included with your print or eBook purchase, at https://livebook.manning.com/book/ deep-learning-with-python-third-edition. The complete code for the examples in the book is available for download from the Manning website at https://www.manning .com/books/deep-learning-with-python-third-edition and on GitHub, at https:// github.com/fchollet/deep-learning-with-python-notebooks, along with instructions for running the code locally. You can run the code directly through the web version of this book available at https://deeplearningwithpython.io. A Colab link appears at the top of each chapter allowing you to run the chapter’s code in the browser on free, hosted hardware.

liveBook discussion forum

Purchase of Deep Learning with Python, Third Edition, includes free access to liveBoo …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.7 — brief contents › 
- PDF p.18 — preface › 
- PDF p.8 — contents › 
- PDF p.41 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.650 — index › 
- PDF p.215 — 7 A deep dive on Keras › 7.1 A spectrum of workflows
- PDF p.623 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.631 — 20 Conclusions › 20.4 Staying up to date in a fast-moving field

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.18 (printed xviii) — preface ›  › 
of Keras, and over the last decade, the Keras developer community has made remarkable progress in this area. We’ve put deep learning into the hands of millions of people, who, in turn, are using it to solve problems that were, until recently, thought to be unsolvable.

The book you’re holding is another step on the way toward broadening access to the field. We aim to make the concepts behind deep learning and their implementation as approachable as possible. Doing so doesn’t require watering anything down; we believe that there are no difficult ideas in deep learning. This book will start with the very basics of the field and, layer by layer, build up to the cutting-edge generative AI models being deployed today. This is the third edition of Deep Learning with Python. In an effort to make this content as broadly available as possible, we are making this edition available for free, online at https://deeplearningwithpython.io/. We hope you will consider purchasing a physical copy to support the project and read what we consider to be the best presentation of the content. This third edition amounts to a complete rewrite of the entire book, with a broader introduction to today’s popula

### PDF p.7 (printed vii) — brief contents ›  › brief contents
1

■ What is deep learning? 1

2

■ The mathematical building blocks of neural networks 16

3

■ Introduction to TensorFlow, PyTorch, JAX, and Keras 60

4

■ Classification and regression 104

5

■ Fundamentals of machine learning 136

6

■ The universal workflow of machine learning 171

7

■ A deep dive on Keras 190

8

■ Image classification 231

9

■ ConvNet architecture patterns 268

10

■ Interpreting what ConvNets learn 284

11

■ Image segmentation 308

12

■ Object detection 329

13

■ Timeseries forecasting 351

14

■ Text classification 381

15

■ Language models and the Transformer 421

16

■ Text generation 466

17

■ Image generation 508

18

■ Best practices for the real world 538

19

■ The future of AI 564

20

■ Conclusions 595

v

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

### PDF p.215 (printed 191) — 7 A deep dive on Keras › 7.1 A spectrum of workflows › 
use the gradient APIs in TensorFlow, JAX, and PyTorch to implement a step-by-step training loop.

In the coming chapters, we’ll dig into computer vision, timeseries forecasting, natural language processing, and generative deep learning. These complex applications will require much more than a `Sequential` architecture and the default `fit()` loop. So let’s first turn you into a Keras expert! In this chapter, you’ll get a complete overview of the key ways to work with Keras APIs: everything you’re going to need to handle the advanced deep learning use cases you’ll encounter next.

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

### PDF p.85 (printed 61) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.1 A brief history of deep learning frameworks › A brief history of deep learning frameworks
In the real world, you’re not going to be writing low-level code from scratch like we did at the end of chapter 2. Instead, you’re going to use a framework. Besides Keras, the main deep learning frameworks today are JAX, TensorFlow, and PyTorch. This book will teach you about all four.

If you’re just getting started with deep learning, it may seem like all these frameworks have been here forever. In reality, they’re all quite recent, with Keras being the oldest among the four (launched in March 2015). The ideas behind these frameworks, however, have a long history—the first paper about automatic differentiation was published in 1964.1 All these frameworks combine three key features:

A way to compute gradients for arbitrary differentiable functions (automatic differentiation)

A way to run tensor computations on CPUs and GPUs (and possibly even on other specialized deep learning hardware)

A way to distribute computation across multiple devices or multiple computers, such as multiple GPUs on one computer, or even multiple GPUs across multiple separate computers

Together, these three simple features unlock all modern deep learning.

### PDF p.8 (printed viii) — contents ›  › contents
preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

### PDF p.171 (printed 147) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Figure 5.10 Going from a random model to an overfit model and achieving a robust fit as an intermediate state

Deep learning models tend to be structured in a way that mirrors the “shape” of the information in their training data (via architecture priors). This is the case in particular for image-processing models (see chapters 8–12) and sequence-processing models (see chapter 13). More generally, deep neural networks structure their learned representations in a hierarchical and modular way, which echoes the way natural data is organized.
