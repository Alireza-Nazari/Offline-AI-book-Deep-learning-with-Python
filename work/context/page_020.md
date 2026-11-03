# Generation bundle for page_020  (PDF page 20, printed page xx)

Page kind: front_matter
Chapter: about this book
Section: Who should read this book
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_020.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_020.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p020_e01] (chapter_title) about this book
[p020_e02] (body) This book was written for anyone who wishes to explore deep learning from scratch or broaden their understanding of deep learning. Whether you’re a practicing machine learning engineer, a software developer, or a college student, you’ll find value in these pages.
[p020_e03] (body) You’ll explore deep learning in an approachable way—starting simply and working up to state-of-the-art techniques. We hope you’ll find that this book strikes a balance between intuition, theory, and hands-on practice. It avoids mathematical notation, preferring instead to explain the core ideas of deep learning via functioning code paired with explanations of the underlying principles. You’ll train machine learning models from scratch in a number of different problem domains and learn practical recommendations for writing deep learning programs and deploying them in the real world. After reading this book, you’ll have a solid understanding of what deep learning is, when it’s applicable, and what its limitations are. You’ll be familiar with the standard workflow for approaching and solving machine learning problems, and you’ll know how to address commonly encountered issues.
[p020_e04] (heading_section) Who should read this book
[p020_e05] (body) This book is written for people with some Python programming experience who want to get started with machine learning and deep learning. But this book can also be valuable to many different types of readers:
[p020_e06] (bullet) If you’re a data scientist familiar with machine learning, this book will provide you with a solid, practical introduction to deep learning, the fastest-growing and most significant subfield of machine learning.
[p020_e07] (body) xviii

## Chapter outline
Chapter: about this book (PDF pages 20-22)
  - Who should read this book (p.20)
  - How this book is organized: A road map (p.21)
  - About the code (p.22)
  - liveBook discussion forum (p.22)

## Previous page (PDF 19) text, for continuity
acknowledgments

First, we’d like to thank the Keras community for making this book possible. Over the past decade, Keras has grown to have thousands of open source contributors and more than 2 million users. Their contributions and feedback have turned Keras into what it is today.

On a personal note, François would like to thank his wife for her endless support during the development of Keras and the writing of this book. Matthew would like to thank his partner Kate, his family, and all the friends who have supported him along the way. We thank the people at Manning who made this book possible: publisher Marjan Bace and everyone on the editorial and production teams, including Michael Stephens, Aleksandar Dragosavljević, and many others who worked behind the scenes. Many thanks go to the peer reviewers: Aakash Nain, Abheesht Sharma, Abhishek Shivanna, Aritra Roy Gosthipaty, Avinash Tiwari, Brandon Friar, Christopher Kardell, Srivathsan Srinivasagopalan, Edmon Begoli, Guillaume Alleon, Ian Stirk, Jacqueline Nolis, Kishore Reddy, Levi McClenny, Margaret Maynard-Reid, Nilson Chapagain, Prashanth Josyula, Preetish Kakkar, Sai Srinivas Somarouthu, Samuel Marks, Srivathsan Srinivasagopalan, Thiago Britto Borges, Todd Cook, and Varun Chawla—and all the other people who sent us feedback on the draft of the book. On the coding side, special thanks go to Tomasz Kalinowski, who contributed to code examples in this book; Ian Hough, who served as the book’s development editor; and Gabriel Rasskin, who served as the book’s technical proofreader.

xvii

## Next page (PDF 21) text, for continuity
If you’re a deep learning researcher or practitioner looking to get started with the Keras framework, you’ll find this book to be the ideal Keras crash course.

If you’re a graduate student studying deep learning in a formal setting, you’ll find this book to be a practical complement to your education, helping you build intuition around the behavior of deep neural networks and familiarizing you with key best practices.

Even technically minded people who don’t code regularly will find this book useful as an introduction to both basic and advanced deep learning concepts.

To understand the code examples, you’ll need reasonable Python proficiency. You don’t need previous experience with machine learning or deep learning: this book covers, from scratch, all the necessary basics. You don’t need an advanced mathematics background either—high-school-level mathematics should suffice to follow along.

How this book is organized: A road map

This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter cov …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.22 — about this book › liveBook discussion forum
- PDF p.8 — contents › 
- PDF p.631 — 20 Conclusions › 20.4 Staying up to date in a fast-moving field
- PDF p.18 — preface › 
- PDF p.41 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.3 — Praise for the Second Edition › 
- PDF p.26 — 1 What is deep learning? › 1.2 Artificial intelligence
- PDF p.34 — 1 What is deep learning? › 1.7 What makes deep learning different

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.8 (printed viii) — contents ›  › contents
preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

### PDF p.22 (printed xxii) — about this book › liveBook discussion forum › About the code
This book is an example of literate programming—it is simultaneously a prose explanation of deep learning concepts and a runnable Python program. Starting with chapter 2, each chapter will be interspersed with code listings that can be run as a Jupyter notebook, an interactive notebook for running Python code and visualizing data. We strongly suggest running, modifying, and playing with the examples in this book as you read to build your own intuitions on the concepts presented.

All code is written in the Python language with the deep learning library Keras. Keras can be run on top of TensorFlow, PyTorch, and JAX, the most popular low-level deep learning frameworks as of 2025. All code can be run on a local machine or directly in the browser using Google Colab, a hosted environment for Jupyter notebooks. You can get executable snippets of code from the liveBook version of this book, included with your print or eBook purchase, at https://livebook.manning.com/book/ deep-learning-with-python-third-edition. The complete code for the examples in the book is available for download from the Manning website at https://www.manning .com/books/deep-learning-with-python-third-edition and on G

### PDF p.3 (printed iii) — Praise for the Second Edition ›  › Praise for the Second Edition
Chollet is a master of pedagogy and explains complex concepts with minimal fuss, cutting through the math with practical Python code. He is also an experienced ML researcher, and his insights on various model architectures or training tips are a joy to read.

—Martin Görner, Google

Immerse yourself in this exciting introduction to the topic with lots of real-world examples. A must-read for every deep learning practitioner.

—Sayak Paul, Carted

The modern classic just got better.

—Edmon Begoli, Oak Ridge National Laboratory

Truly the bible of deep learning.

—Yiannis Paraskevopoulos, University of West Attica

One of the best books on deep learning with Python.

—Raushan Jha, Microsoft

The book is full of insights, useful both for the novice and the more experienced machine learning professional.

—Viton Vitanis, Viseca Payment Services

Deep learning well explained, from A to Z.

—Todd Cook, Appen

This book really implements the democratization of AI: “AI to the people.”

—Kjell Jansson, GubboIT

### PDF p.18 (printed xviii) — preface ›  › 
of Keras, and over the last decade, the Keras developer community has made remarkable progress in this area. We’ve put deep learning into the hands of millions of people, who, in turn, are using it to solve problems that were, until recently, thought to be unsolvable.

The book you’re holding is another step on the way toward broadening access to the field. We aim to make the concepts behind deep learning and their implementation as approachable as possible. Doing so doesn’t require watering anything down; we believe that there are no difficult ideas in deep learning. This book will start with the very basics of the field and, layer by layer, build up to the cutting-edge generative AI models being deployed today. This is the third edition of Deep Learning with Python. In an effort to make this content as broadly available as possible, we are making this edition available for free, online at https://deeplearningwithpython.io/. We hope you will consider purchasing a physical copy to support the project and read what we consider to be the best presentation of the content. This third edition amounts to a complete rewrite of the entire book, with a broader introduction to today’s popula

### PDF p.25 (printed 1) — 1 What is deep learning? ›  › This chapter covers
High-level definitions of fundamental concepts

A soft introduction to the principles behind

machine learning

Deep learning’s rising popularity and future

potential

Over the past decade, artificial intelligence (AI) has been a subject of intense media hype. Machine learning, deep learning, and AI come up in countless articles, often outside of technology-minded publications. We’re promised a future of intelligent chatbots, self-driving cars, and virtual assistants—a future sometimes painted in a grim light and other times as utopian, where human jobs will be scarce and most economic activity will be handled by robots or AI agents. For a practitioner of machine learning, it’s important to be able to recognize the signal amid the noise, so that you can tell world-changing developments from overhyped press releases. Our future is at stake, and it’s one in which you have an active role to play: after reading this book, you’ll be one of those who can develop these AI systems. So let’s

### PDF p.34 (printed 10) — 1 What is deep learning? › 1.7 What makes deep learning different › What makes deep learning different
Is there anything special about deep neural networks that makes them the “right” approach for companies to invest in and for researchers to flock to? Will we still be using deep neural networks in 20 years?

Deep learning has several properties that justify its status as an AI revolution, and it’s here to stay. We may not be using neural networks many decades from now, but whatever we use will directly inherit from modern deep learning and its core concepts. These important properties can be broadly sorted into three categories:

Simplicity—Deep learning makes problem solving much easier, because it automates what used to be the most crucial step in a machine learning workflow: feature engineering. Previous machine learning techniques—shallow learning—only involved transforming the input data into one or two successive representation spaces, which wasn’t expressive enough for most problems. As such, humans had to go to great lengths to make the initial input data more amenable to processing by these methods: they had to manually engineer good representations for their data. This is called feature engineering. Deep learning, on the other hand, completely automates this step: with de

### PDF p.17 (printed xvii) — preface ›  › preface
If you’ve picked up this book, you’re probably aware of the extraordinary progress that deep learning has brought to the field of artificial intelligence. In just a handful of years, we went from near-unusable computer vision and natural language processing to highly performant systems deployed at scale in products you use every day. The consequences of this sudden progress extend to almost every industry. Deep learning is applied to a diverse range of important problems across domains as different as medical imaging, agriculture, autonomous driving, education, disaster prevention, and manufacturing. Digital assistants are becoming pervasive on nearly every consumer computing device.

Yet, we believe deep learning is still in its early days. AI is going to represent a much greater wave of disruption than anything that came before it. The technology will continue to make its way to every problem where it can provide some utility—a transformation that will take decades to play out. This is a transformation with profound societal implications, on a global scale, spanning industries and cultures. We strongly believe the only way to ensure such a transformation is beneficial for the peo

### PDF p.41 (printed 17) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › Running the code in this book
This book is full of runnable Python code. Each chapter is paired with a Jupyter notebook that contains all of the code from the chapter. A Jupyter notebook is a live Python scratch pad of sorts, where you can interactively run code, graph data, view images, and a lot more. You will gain a lot more practical knowledge from this book if you run and experiment with the code as you read. By far the easiest way to set up a deep learning environment to run these notebooks is Google Colaboratory (or Colab for short), a hosted environment for Jupyter notebooks that has become the industry standard for ML practitioners. With Colab, you can run the code for this book interactively in the browser, connecting to cloud runtimes with configurable hardware. By default, the notebooks in this book will run on Colab’s free GPU runtime. If you would like, you can also run these notebooks locally on your own machine. A GPU is recommended, especially as you get to the larger and more compute-intensive models later in this book. Instructions for running locally and on Colab, along with the code, can be found at https://github.com/fchollet/deep-learning-with-python-notebooks.
