# Generation bundle for page_016  (PDF page 16, printed page xvi)

Page kind: front_matter
Chapter: contents
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_016.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_016.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p016_e04] (body) 19.3
[p016_e05] (body) How to build intelligence 576
[p016_e06] (quote) The kaleidoscope hypothesis 577 ■ The essence of intelligence: Abstraction acquisition and recombination 578 The importance of setting the right target 578 ■ A new target: On-the-fly adaptation 580 ■ ARC Prize 581 ■ The test-time adaptation era 582 ■ ARC-AGI 2 583
[p016_e07] (body) 19.4
[p016_e08] (body) The missing ingredients: Search and symbols 584
[p016_e09] (quote) The two poles of abstraction 585 ■ Cognition as a combination of both kinds of abstraction 587 ■ Why deep learning isn’t a complete answer to abstraction generation 588 ■ An alternative approach to AI: Program synthesis 589 ■ Blending deep learning and program synthesis 590 ■ Modular component recombination and lifelong learning 592 ■ The long-term vision 593
[p016_e10] (body) 20
[p016_e11] (body) Conclusions 595 20.1
[p016_e12] (body) Key concepts in review 595
[p016_e13] (quote) Various approaches to artificial intelligence 596 ■ What makes deep learning special within the field of machine learning 596 How to think about deep learning 597 ■ Key enabling technologies 598 ■ The universal machine learning workflow 599 ■ Key network architectures 600
[p016_e14] (body) 20.2
[p016_e15] (body) Limitations of deep learning 605 20.3 What might lie ahead 606 20.4 Staying up to date in a fast-moving field 607
[p016_e16] (quote) Practice on real-world problems using Kaggle 607 ■ Read about the latest developments on arXiv 607 ■ Explore the Keras ecosystem 608
[p016_e17] (body) 20.5
[p016_e18] (body) Final words 608
[p016_e19] (quote) index 609

## Chapter outline
Chapter: contents (PDF pages 8-16)

## Previous page (PDF 15) text, for continuity
17

Image generation 508 17.1

Deep learning for image generation 508

Sampling from latent spaces of images 509 ■ Variational autoencoders 510 ■ Implementing a VAE with Keras 513

17.2

Diffusion models 518

The Oxford Flowers dataset 520 ■ A U-Net denoising autoencoder 521 ■ The concepts of diffusion time and diffusion schedule 523 ■ The training process 525 ■ The generation process 527 ■ Visualizing results with a custom callback 528 It’s go time! 529

17.3

Text-to-image models 531

Exploring the latent space of a text-to-image model 533

18

Best practices for the real world 538 18.1

Getting the most out of your models 539

Hyperparameter optimization 539 ■ Model ensembling 546

18.2

Scaling up model training with multiple devices 548

Multi-GPU training 548 ■ Distributed training in practice 550 TPU training 555

18.3

Speeding up training and inference with lower-precision computation 556

Understanding floating-point precision 556 ■ Float16 inference 558 ■ Mixed-precision training 559 ■ Using loss scaling with mixed precision 559 ■ Beyond mixed precision: float8 training 560 ■ Faster inference with quantization 561

19

The future of AI 564 19.1

The limitations of deep learning 564

Deep learning models struggle to adapt to novelty 565 Deep learning models are highly sensitive to phrasing and other distractors 567 ■ Deep learning models struggle to learn generalizable programs 569 ■ The risk of anthropomorphizing machine-learning models 569

19.2

Scale isn’t all you need 570

Automatons vs. intelligent agents 571 ■ Local generalization vs. extreme generalization 573 ■ The purpose of intelligence 575 Climbing the spectrum of generalization 575

## Next page (PDF 17) text, for continuity
preface

If you’ve picked up this book, you’re probably aware of the extraordinary progress that deep learning has brought to the field of artificial intelligence. In just a handful of years, we went from near-unusable computer vision and natural language processing to highly performant systems deployed at scale in products you use every day. The consequences of this sudden progress extend to almost every industry. Deep learning is applied to a diverse range of important problems across domains as different as medical imaging, agriculture, autonomous driving, education, disaster prevention, and manufacturing. Digital assistants are becoming pervasive on nearly every consumer computing device.

Yet, we believe deep learning is still in its early days. AI is going to represent a much greater wave of disruption than anything that came before it. The technology will continue to make its way to every problem where it can provide some utility—a transformation that will take decades to play out. This is a transformation with profound societal implications, on a global scale, spanning industries and cultures. We strongly believe the only way to ensure such a transformation is beneficial for the people it will affect— all of us—is to radically democratize access to the underlying technology. We need to put understanding of deep learning—how it works, where it fails, and how to apply it—in the hands of as many people as possible, including people who aren’t researchers in the field. When I wrote the Keras deep learning framework in March 2015, the democratization of AI wasn’t what I had in mind. I had been doing research in machine learning for several years and had built Keras to help with my own experiments. But since then, as newcomers have entered the field of deep learning,  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.630 — 20 Conclusions › 20.3 What might lie ahead
- PDF p.612 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.602 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.609 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.614 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.8 — contents › 
- PDF p.7 — brief contents › 
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 20.3 What might lie ahead
Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.

Future key developments may include

Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, co

### PDF p.602 (printed 578) — 19 The future of AI › 19.3 How to build intelligence › 19.3.2 The essence of intelligence: Abstraction acquisition and recombination
So how can we move beyond these limitations?

### PDF p.8 (printed viii) — contents ›  › contents
preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

### PDF p.609 (printed 585) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
I’ve said that intelligence consists of two components: abstraction acquisition and abstraction recombination. They are tightly coupled—what kind of abstractions you manipulate determines how, and how well, you can recombine them. Deep learning models only manipulate abstractions stored via parametric curves, fitted via gradient descent. Could there be a better way?

### PDF p.602 (printed 578) — 19 The future of AI › 19.3 How to build intelligence › 19.3.2 The essence of intelligence: Abstraction acquisition and recombination
Intelligence is the ability to mine your experience to identify these atoms of meaning that can seemingly be reused across many different situations—the core beads of the kaleidoscope. Once extracted, they’re called abstractions. Whenever you encounter a new situation, you make sense of it by recombining on the fly abstractions from your collections, to weave a brand new “model” adapted to the situation.

This process consists of two key parts:

Abstraction acquisition—Efficiently extracting compact, reusable abstractions from a stream of experience or data. This involves identifying underlying structures, principles, or invariants.

On-the-fly recombination—Efficiently selecting and recombining these abstractions in novel ways to model new problems and situations, even ones far removed from past experience.

The emphasis on efficiency is crucial. How intelligent you are is determined by how efficiently you can acquire good abstractions from limited experience and how efficiently you can recombine them to navigate uncertainty and novelty. If you need hundreds of thousands of hours of practice to acquire a skill, you are not very intelligent. If you need to enumerate every possible 

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

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

### PDF p.616 (printed 592) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
Deep learning can help program synthesis do the same: although each specific program we’d like to generate might be a fundamentally discrete object that performs non-interpolative data manipulation, evidence so far indicates that the space of all useful programs may look a lot like a continuous manifold. That means that a deep learning model that has been trained on millions of successful program-generation episodes might start to develop solid intuition about the path through program space that the search process should take to go from a specification to the corresponding program—just like a software engineer might have immediate intuition about the overall architecture of the script they’re about to write and about the intermediate functions and classes they should use as stepping stones on the way to the goal. Remember that human reasoning is heavily guided by value-centric abstraction— that is, by pattern recognition and intuition. The same should be true of program synthesis. I expect the general approach of guiding program search via learned heuristics to see increasing research interest over the next 10 to 20 years.
