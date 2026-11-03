# Generation bundle for page_619  (PDF page 619, printed page 595)

Page kind: chapter_opener
Chapter: 20 Conclusions
Section: 20.1 Key concepts in review
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_619.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_619.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p619_e01] (chapter_number) 20
[p619_e02] (chapter_title) Conclusions
[p619_e03] (chapter_covers) This chapter covers
[p619_e04] (bullet) Important takeaways from this book
[p619_e05] (bullet) Resources for learning further and applying your
[p619_e06] (bullet) skills in practice
[p619_e07] (body) We’ll start with a bird’s-eye view of what you should take away from this book. This should refresh your memory regarding some of the concepts you’ve learned. Then, we’ll give you a short list of resources and strategies for learning further about machine learning and staying up to date with new advances.
[p619_e08] (body) Becoming an effective AI practitioner is a journey, and finishing this book is merely your first step on it. I want to make sure you realize this and are properly equipped to take the next steps of this journey on your own.
[p619_e09] (heading_section) 20.1 Key concepts in review
[p619_e10] (body) This section briefly synthesizes key takeaways from this book. If you ever need a quick refresher to help you recall what you’ve learned, you can read these few pages.

## Chapter outline
Chapter: 20 Conclusions (PDF pages 619-632)
  - 20.1 Key concepts in review (p.619)
      - 20.1.1 Various approaches to artificial intelligence (p.620)
      - 20.1.2 What makes deep learning special within the field of machine learning (p.620)
      - 20.1.3 How to think about deep learning (p.621)
      - 20.1.4 Key enabling technologies (p.622)
      - 20.1.5 The universal machine learning workflow (p.623)
      - 20.1.6 Key network architectures (p.624)
  - 20.2 Limitations of deep learning (p.629)
  - 20.3 What might lie ahead (p.630)
  - 20.4 Staying up to date in a fast-moving field (p.631)
      - 20.4.1 Practice on real-world problems using Kaggle (p.631)
      - 20.4.2 Read about the latest developments on arXiv (p.631)
      - 20.4.3 Explore the Keras ecosystem (p.632)
  - 20.5 Final words (p.632)

## Previous page (PDF 618) text, for continuity
that humans maintain about their surroundings and themselves, and they will be capable of stronger generalization due to their rich algorithmic nature.

In particular, models will blend algorithmic modules providing formal reasoning, search, and abstraction capabilities with geometric modules providing informal intuition and pattern-recognition capabilities. This will achieve a blend of value-centric and program-centric abstraction. AlphaGo or self-driving cars (systems that required a lot of manual software engineering and human-made design decisions) provide an early example of what such a blend of symbolic and geometric AI could look like.

Such models will be grown automatically rather than hardcoded by human engineers, using modular parts stored in a global library of reusable subroutines—a library evolved by learning high-performing models on thousands of previous tasks and datasets. As frequent problem-solving patterns are identified by the meta-learning system, they will be turned into reusable subroutines—much like functions and classes in software engineering—and added to the global library.

The process that searches over possible combinations of subroutines to grow new models will be a discrete search process (program synthesis), but it will be heavily guided by a form of program-space intuition provided by deep learning.

This global subroutine library and associated model-growing system will be able to achieve some form of human-like extreme generalization: given a new task or situation, the system will be able to assemble a new working model appropriate for the task using very little data, thanks to rich program-like primitives that generalize well, and extensive experience with similar tasks. In the same way, humans can quickly learn to play a complex ne …

## Next page (PDF 620) text, for continuity
20.1.1 Various approaches to artificial intelligence

First, deep learning isn’t synonymous with artificial intelligence (AI), or even with machine learning:

Artificial intelligence (AI) is an ancient, broad field that can generally be understood as “all attempts to automate human cognitive processes.” This can range from the very basic, such as an Excel spreadsheet, to the very advanced, like a humanoid robot that can walk and talk.

Machine learning is a specific subfield of AI that aims at automatically developing programs (called models) purely from exposure to training data. This process of turning data into a program is called learning. Although machine learning has been around for a long time, it only started to take off in the 1990s, before becoming the dominant form of AI in the 2000s.

Deep learning is one of many branches of machine learning, where the models are long chains of geometric transformations, applied one after the other. These operations are structured into modules called layers: deep learning models are typically stacks of layers—or, more generally, graphs of layers. These layers are parameterized by weights, which are the parameters learned during training. The knowledge of a model is stored in its weights, and the process of learning consists of finding “good values” for these weights—values that minimize a loss function. Because the chain of geometric transformations considered is differentiable, updating the weights to minimize the loss function is done efficiently via gradient

descent.

Generative AI is a specific subset of deep learning, where models are capable of generating text, images, videos, or sound. These models tend to be very large— billions of parameters. They’re trained in a self-supervised manner; that is, they’re trained to  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.631 — 20 Conclusions › 20.4 Staying up to date in a fast-moving field
- PDF p.632 — 20 Conclusions › 20.5 Final words
- PDF p.21 — about this book › How this book is organized: A road map
- PDF p.20 — about this book › Who should read this book
- PDF p.22 — about this book › liveBook discussion forum
- PDF p.41 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.621 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.17 — preface › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.631 (printed 607) — 20 Conclusions › 20.4 Staying up to date in a fast-moving field › 20.4 Staying up to date in a fast-moving field
As final parting words, I want to give you some pointers about how to keep learning and updating your knowledge and skills after you’ve turned the last page of this book. The field of modern deep learning, as we know it today, is only a few years old, despite a long, slow prehistory stretching back decades. With an exponential increase in financial resources and research headcount since 2013, the field as a whole is now moving at a frenetic pace. What you’ve learned in this book won’t stay relevant forever, and it isn’t all you’ll need for the rest of your career.

Fortunately, there are plenty of free online resources that you can use to stay up to date and expand your horizons. Here are a few.

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › 
If you’re a deep learning researcher or practitioner looking to get started with the Keras framework, you’ll find this book to be the ideal Keras crash course.

If you’re a graduate student studying deep learning in a formal setting, you’ll find this book to be a practical complement to your education, helping you build intuition around the behavior of deep neural networks and familiarizing you with key best practices.

Even technically minded people who don’t code regularly will find this book useful as an introduction to both basic and advanced deep learning concepts.

To understand the code examples, you’ll need reasonable Python proficiency. You don’t need previous experience with machine learning or deep learning: this book covers, from scratch, all the necessary basics. You don’t need an advanced mathematics background either—high-school-level mathematics should suffice to follow along.

### PDF p.632 (printed 608) — 20 Conclusions › 20.5 Final words › 20.5 Final words
This is the end of Deep Learning with Python! I hope you’ve learned a thing or two about machine learning, deep learning, Keras, and maybe even cognition in general. Learning is a lifelong journey, especially in the field of AI, where we have far more unknowns on our hands than certitudes. So please go on learning, questioning, and researching. Never stop. Because even given the progress made so far, most of the fundamental questions in AI remain unanswered. Many haven’t even been properly asked yet.

### PDF p.25 (printed 1) — 1 What is deep learning? ›  › This chapter covers
High-level definitions of fundamental concepts

A soft introduction to the principles behind

machine learning

Deep learning’s rising popularity and future

potential

Over the past decade, artificial intelligence (AI) has been a subject of intense media hype. Machine learning, deep learning, and AI come up in countless articles, often outside of technology-minded publications. We’re promised a future of intelligent chatbots, self-driving cars, and virtual assistants—a future sometimes painted in a grim light and other times as utopian, where human jobs will be scarce and most economic activity will be handled by robots or AI agents. For a practitioner of machine learning, it’s important to be able to recognize the signal amid the noise, so that you can tell world-changing developments from overhyped press releases. Our future is at stake, and it’s one in which you have an active role to play: after reading this book, you’ll be one of those who can develop these AI systems. So let’s

### PDF p.20 (printed xx) — about this book › Who should read this book › Who should read this book
This book is written for people with some Python programming experience who want to get started with machine learning and deep learning. But this book can also be valuable to many different types of readers:

If you’re a data scientist familiar with machine learning, this book will provide you with a solid, practical introduction to deep learning, the fastest-growing and most significant subfield of machine learning.

xviii

### PDF p.20 (printed xx) — about this book › Who should read this book › about this book
This book was written for anyone who wishes to explore deep learning from scratch or broaden their understanding of deep learning. Whether you’re a practicing machine learning engineer, a software developer, or a college student, you’ll find value in these pages.

You’ll explore deep learning in an approachable way—starting simply and working up to state-of-the-art techniques. We hope you’ll find that this book strikes a balance between intuition, theory, and hands-on practice. It avoids mathematical notation, preferring instead to explain the core ideas of deep learning via functioning code paired with explanations of the underlying principles. You’ll train machine learning models from scratch in a number of different problem domains and learn practical recommendations for writing deep learning programs and deploying them in the real world. After reading this book, you’ll have a solid understanding of what deep learning is, when it’s applicable, and what its limitations are. You’ll be familiar with the standard workflow for approaching and solving machine learning problems, and you’ll know how to address commonly encountered issues.

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

### PDF p.588 (printed 564) — 19 The future of AI › 19.1 The limitations of deep learning › This chapter covers
The limitations of deep learning

The nature of intelligence

What’s missing from current approaches

What the future might look like

To use a tool appropriately, you should not only understand what it can do but also be aware of what it can’t do. I’m going to present an overview of some key limitations of deep learning. Then, I’ll offer some speculative thoughts about the future evolution of AI and what it would take to get to human-level general intelligence. This should be especially interesting to you if you’d like to get into fundamental research.
