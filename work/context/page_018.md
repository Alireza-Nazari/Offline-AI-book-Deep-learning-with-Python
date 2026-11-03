# Generation bundle for page_018  (PDF page 18, printed page xviii)

Page kind: front_matter
Chapter: preface
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_018.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_018.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p018_e04] (body) of Keras, and over the last decade, the Keras developer community has made remarkable progress in this area. We’ve put deep learning into the hands of millions of people, who, in turn, are using it to solve problems that were, until recently, thought to be unsolvable.
[p018_e05] (body) The book you’re holding is another step on the way toward broadening access to the field. We aim to make the concepts behind deep learning and their implementation as approachable as possible. Doing so doesn’t require watering anything down; we believe that there are no difficult ideas in deep learning. This book will start with the very basics of the field and, layer by layer, build up to the cutting-edge generative AI models being deployed today. This is the third edition of Deep Learning with Python. In an effort to make this content as broadly available as possible, we are making this edition available for free, online at https://deeplearningwithpython.io/. We hope you will consider purchasing a physical copy to support the project and read what we consider to be the best presentation of the content. This third edition amounts to a complete rewrite of the entire book, with a broader introduction to today’s popular deep learning frameworks and greatly expanded content on large generative AI models. We hope this book proves valuable and helps you solve the problems that matter to you.

## Chapter outline
Chapter: preface (PDF pages 17-18)

## Previous page (PDF 17) text, for continuity
preface

If you’ve picked up this book, you’re probably aware of the extraordinary progress that deep learning has brought to the field of artificial intelligence. In just a handful of years, we went from near-unusable computer vision and natural language processing to highly performant systems deployed at scale in products you use every day. The consequences of this sudden progress extend to almost every industry. Deep learning is applied to a diverse range of important problems across domains as different as medical imaging, agriculture, autonomous driving, education, disaster prevention, and manufacturing. Digital assistants are becoming pervasive on nearly every consumer computing device.

Yet, we believe deep learning is still in its early days. AI is going to represent a much greater wave of disruption than anything that came before it. The technology will continue to make its way to every problem where it can provide some utility—a transformation that will take decades to play out. This is a transformation with profound societal implications, on a global scale, spanning industries and cultures. We strongly believe the only way to ensure such a transformation is beneficial for the people it will affect— all of us—is to radically democratize access to the underlying technology. We need to put understanding of deep learning—how it works, where it fails, and how to apply it—in the hands of as many people as possible, including people who aren’t researchers in the field. When I wrote the Keras deep learning framework in March 2015, the democratization of AI wasn’t what I had in mind. I had been doing research in machine learning for several years and had built Keras to help with my own experiments. But since then, as newcomers have entered the field of deep learning,  …

## Next page (PDF 19) text, for continuity
acknowledgments

First, we’d like to thank the Keras community for making this book possible. Over the past decade, Keras has grown to have thousands of open source contributors and more than 2 million users. Their contributions and feedback have turned Keras into what it is today.

On a personal note, François would like to thank his wife for her endless support during the development of Keras and the writing of this book. Matthew would like to thank his partner Kate, his family, and all the friends who have supported him along the way. We thank the people at Manning who made this book possible: publisher Marjan Bace and everyone on the editorial and production teams, including Michael Stephens, Aleksandar Dragosavljević, and many others who worked behind the scenes. Many thanks go to the peer reviewers: Aakash Nain, Abheesht Sharma, Abhishek Shivanna, Aritra Roy Gosthipaty, Avinash Tiwari, Brandon Friar, Christopher Kardell, Srivathsan Srinivasagopalan, Edmon Begoli, Guillaume Alleon, Ian Stirk, Jacqueline Nolis, Kishore Reddy, Levi McClenny, Margaret Maynard-Reid, Nilson Chapagain, Prashanth Josyula, Preetish Kakkar, Sai Srinivas Somarouthu, Samuel Marks, Srivathsan Srinivasagopalan, Thiago Britto Borges, Todd Cook, and Varun Chawla—and all the other people who sent us feedback on the draft of the book. On the coding side, special thanks go to Tomasz Kalinowski, who contributed to code examples in this book; Ian Hough, who served as the book’s development editor; and Gabriel Rasskin, who served as the book’s technical proofreader.

xvii

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.21 — about this book › How this book is organized: A road map
- PDF p.650 — index › 
- PDF p.8 — contents › 
- PDF p.20 — about this book › Who should read this book
- PDF p.22 — about this book › liveBook discussion forum
- PDF p.632 — 20 Conclusions › 20.5 Final words
- PDF p.629 — 20 Conclusions › 20.2 Limitations of deep learning
- PDF p.7 — brief contents › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

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

### PDF p.8 (printed viii) — contents ›  › contents
preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › 
If you’re a deep learning researcher or practitioner looking to get started with the Keras framework, you’ll find this book to be the ideal Keras crash course.

If you’re a graduate student studying deep learning in a formal setting, you’ll find this book to be a practical complement to your education, helping you build intuition around the behavior of deep neural networks and familiarizing you with key best practices.

Even technically minded people who don’t code regularly will find this book useful as an introduction to both basic and advanced deep learning concepts.

To understand the code examples, you’ll need reasonable Python proficiency. You don’t need previous experience with machine learning or deep learning: this book covers, from scratch, all the necessary basics. You don’t need an advanced mathematics background either—high-school-level mathematics should suffice to follow along.

### PDF p.22 (printed xxii) — about this book › liveBook discussion forum › About the code
This book is an example of literate programming—it is simultaneously a prose explanation of deep learning concepts and a runnable Python program. Starting with chapter 2, each chapter will be interspersed with code listings that can be run as a Jupyter notebook, an interactive notebook for running Python code and visualizing data. We strongly suggest running, modifying, and playing with the examples in this book as you read to build your own intuitions on the concepts presented.

All code is written in the Python language with the deep learning library Keras. Keras can be run on top of TensorFlow, PyTorch, and JAX, the most popular low-level deep learning frameworks as of 2025. All code can be run on a local machine or directly in the browser using Google Colab, a hosted environment for Jupyter notebooks. You can get executable snippets of code from the liveBook version of this book, included with your print or eBook purchase, at https://livebook.manning.com/book/ deep-learning-with-python-third-edition. The complete code for the examples in the book is available for download from the Manning website at https://www.manning .com/books/deep-learning-with-python-third-edition and on G

### PDF p.632 (printed 608) — 20 Conclusions › 20.5 Final words › 20.5 Final words
This is the end of Deep Learning with Python! I hope you’ve learned a thing or two about machine learning, deep learning, Keras, and maybe even cognition in general. Learning is a lifelong journey, especially in the field of AI, where we have far more unknowns on our hands than certitudes. So please go on learning, questioning, and researching. Never stop. Because even given the progress made so far, most of the fundamental questions in AI remain unanswered. Many haven’t even been properly asked yet.

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
