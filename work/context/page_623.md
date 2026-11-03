# Generation bundle for page_623  (PDF page 623, printed page 599)

Page kind: content
Chapter: 20 Conclusions
Section: 20.1 Key concepts in review
Subsection: 20.1.5 The universal machine learning workflow
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_623.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_623.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p623_e03] (body) Moore’s law applied to storage media. Today, state-of-the-art language models are trained on a large fraction of the entire internet.
[p623_e04] (bullet) The availability of fast, highly parallel computation hardware at a low price, especially the GPUs produced by NVIDIA—first gaming GPUs and then chips designed from the ground up for deep learning. Early on, NVIDIA CEO Jensen Huang took note of the deep learning boom and decided to bet the company’s future on it, which paid off in a big way.
[p623_e05] (bullet) A complex stack of software layers that makes this computational power available to humans: the CUDA language, frameworks like TensorFlow, JAX, and PyTorch that do automatic differentiation, and Keras, which makes deep learning accessible to most people.
[p623_e06] (body) In the future, deep learning will not be used only by specialists such as researchers, graduate students, and engineers with an academic profile; it will be a tool in the toolbox of every developer, much like web technology today. Everyone needs to build intelligent apps: just as every business today needs a website, every product will need to intelligently make sense of user-generated data. Bringing about this future will require us to build tools that make deep learning radically easy to use and accessible to anyone with basic coding abilities. Keras has been the first major step in that direction.
[p623_e07] (heading_subsection) 20.1.5 The universal machine learning workflow
[p623_e08] (body) Having access to an extremely powerful tool for creating models that map any input space to any target space is great, but the difficult part of the machine learning workflow is often everything that comes before designing and training such models (and, for production models, what comes after, as well). Understanding the problem domain to be able to determine what to attempt to predict, given what data, and how to measure success is a prerequisite for any successful application of machine learning, and it isn’t something that advanced tools like Keras and TensorFlow can help you with. As a reminder, here’s a quick summary of the typical machine learning workflow as described in chapter 6:
[p623_e09] (bullet) Define the problem. What data is available, and what are you trying to predict? Will you need to collect more data or hire people to manually label a dataset?
[p623_e10] (bullet) Identify a way to reliably measure success on your goal. For simple tasks, this may be prediction accuracy, but in many cases, it will require sophisticated, domain-specific metrics.
[p623_e11] (bullet) Prepare the validation process that you’ll use to evaluate your models. In particular, you should define a training set, a validation set, and a test set. The validationset and test-set labels shouldn’t leak into the training data: for instance, with temporal prediction, the validation and test data should be posterior to the training data.
[p623_e12] (bullet) Vectorize the data by turning it into vectors and preprocessing it in a way that makes it more
[p623_e13] (quote) easily approachable by a neural network (normalization and so on).

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

## Previous page (PDF 622) text, for continuity
transformation is that it must be differentiable, which is required for us to be able to learn its parameters via gradient descent. Intuitively, this means the geometric morphing from inputs to outputs must be smooth and continuous—a significant constraint.

The entire process of applying this complex geometric transformation to the input data can be visualized in 3D by imagining a person trying to uncrumple a paper ball: the crumpled paper ball is the manifold of the input data that the model starts with. Each movement operated by the person on the paper ball is similar to a simple geometric transformation operated by one layer. The full uncrumpling gesture sequence is the complex transformation of the entire model. Deep learning models are mathematical machines for uncrumpling complicated manifolds of high-dimensional data. That’s the magic of deep learning—turning meaning into vectors, into geometric spaces, and then incrementally learning complex geometric transformations that map one space to another. All you need are spaces of sufficiently high dimensionality to capture the full scope of the relationships found in the original data. The whole thing hinges on two core ideas: that meaning is derived from the pairwise relationship between things (between words in a language, between pixels in an image, and so on) and that these relationships can be captured by a distance function. But note that whether the brain implements meaning via geometric spaces is an entirely separate question. Vector spaces are efficient to work with from a computational standpoint, but different data structures for intelligence can easily be envisioned—in particular, graphs. Neural networks initially emerged from the idea of using graphs as a way to encode meaning, which is why they’re named …

## Next page (PDF 624) text, for continuity
Develop a first model that beats a trivial common-sense baseline, thus demonstrating that

machine learning can work on your problem. This may not always be the case!

Gradually refine your model architecture by tuning hyperparameters and adding regularization. Make changes based on performance on the validation data only, not the test data or the training data. Remember that you should get your model to overfit (thus identifying a model capacity level that’s greater than you need) and only then begin to add regularization or downsize your model. Beware of validation-set overfitting when tuning hyperparameters—the fact that your hyperparameters may end up being overspecialized to the validation set. Avoiding this is the purpose of having a separate test set!

Deploy your final model in production—as a web API, as part of a JavaScript or C++ application, on an embedded device, etc. Keep monitoring its performance on real-world data and use your findings to refine the next iteration of the model!

20.1.6 Key network architectures

The families of network architectures that you should be familiar with after reading this book are densely connected networks, convolutional networks, recurrent networks, Diffusion Models, and Transformers. Each type of model is meant for specific data modalities: a network architecture encodes assumptions about the structure of the data—a hypothesis space within which the search for a good model will proceed. Whether a given architecture will work on a given problem depends entirely on the match between the structure of the data and the assumptions of the network architecture.

These different network types can easily be combined to achieve larger multimodal models, much as you combine LEGO bricks. In a way, deep learning layers are LEGO bricks …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.213 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.21 — about this book › How this book is organized: A road map
- PDF p.196 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.215 — 7 A deep dive on Keras › 7.1 A spectrum of workflows
- PDF p.193 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.34 — 1 What is deep learning? › 1.7 What makes deep learning different
- PDF p.205 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.34 (printed 10) — 1 What is deep learning? › 1.7 What makes deep learning different › What makes deep learning different
Is there anything special about deep neural networks that makes them the “right” approach for companies to invest in and for researchers to flock to? Will we still be using deep neural networks in 20 years?

Deep learning has several properties that justify its status as an AI revolution, and it’s here to stay. We may not be using neural networks many decades from now, but whatever we use will directly inherit from modern deep learning and its core concepts. These important properties can be broadly sorted into three categories:

Simplicity—Deep learning makes problem solving much easier, because it automates what used to be the most crucial step in a machine learning workflow: feature engineering. Previous machine learning techniques—shallow learning—only involved transforming the input data into one or two successive representation spaces, which wasn’t expressive enough for most problems. As such, humans had to go to great lengths to make the initial input data more amenable to processing by these methods: they had to manually engineer good representations for their data. This is called feature engineering. Deep learning, on the other hand, completely automates this step: with de

### PDF p.196 (printed 172) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
Building a music recommendation system for users of an online radio.

Detecting credit card fraud for an e-commerce website.

Predicting display ad click-through rate to decide which ad to serve to a given user at a given time.

Flagging anomalous cookies on the conveyor belt of a cookie-manufacturing line.

Using satellite images to predict the location of as-yet unknown archaeological sites.

It would be very convenient if you could import the correct dataset from `keras
` `.datasets` and start fitting some deep learning models. Unfortunately, in the real world, you’ll have to start from scratch.

In this chapter, you’ll learn about the universal step-by-step blueprint that you can use to approach and solve any machine learning problem, like those previously listed. This template will bring together and consolidate everything you’ve learned in chapters 4 and 5 and give you the wider context that should anchor what you will learn in the next chapters. The universal workflow of machine learning is broadly structured in three parts:

Define the task—Understand the problem domain and the business logic underlying what the customer asked. Collect a dataset, understand what the data re

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

### PDF p.85 (printed 61) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.1 A brief history of deep learning frameworks › A brief history of deep learning frameworks
In the real world, you’re not going to be writing low-level code from scratch like we did at the end of chapter 2. Instead, you’re going to use a framework. Besides Keras, the main deep learning frameworks today are JAX, TensorFlow, and PyTorch. This book will teach you about all four.

If you’re just getting started with deep learning, it may seem like all these frameworks have been here forever. In reality, they’re all quite recent, with Keras being the oldest among the four (launched in March 2015). The ideas behind these frameworks, however, have a long history—the first paper about automatic differentiation was published in 1964.1 All these frameworks combine three key features:

A way to compute gradients for arbitrary differentiable functions (automatic differentiation)

A way to run tensor computations on CPUs and GPUs (and possibly even on other specialized deep learning hardware)

A way to distribute computation across multiple devices or multiple computers, such as multiple GPUs on one computer, or even multiple GPUs across multiple separate computers

Together, these three simple features unlock all modern deep learning.

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

### PDF p.215 (printed 191) — 7 A deep dive on Keras › 7.1 A spectrum of workflows › A spectrum of workflows
As such, there’s not a single “true” way of using Keras. Rather, Keras offers a spectrum of workflows, from the very simple to the very flexible. There are different ways to build Keras models, and different ways to train them, answering different needs. For instance, you have a range of ways to build models and an array of ways to train them, each representing a certain tradeoff between usability and flexibility. You could be using Keras like you would use scikit-learn—just calling `fit()` and letting the framework do its thing—or you could be using it like NumPy—taking full control of every little detail. Because all these workflows are based on shared APIs, such as `Layer` and `Model`, components from any workflow can be used in any other workflow: they can all talk to each other. This means that everything you’re learning now as you’re getting started will still be relevant once you’ve become an expert. You can get started easily and then gradually dive into workflows where you’re writing more and more logic from scratch. You won’t have to switch to an entirely different framework as you go from student to researcher, or from data scientist to deep learning engineer. This philo

### PDF p.17 (printed xvii) — preface ›  › preface
If you’ve picked up this book, you’re probably aware of the extraordinary progress that deep learning has brought to the field of artificial intelligence. In just a handful of years, we went from near-unusable computer vision and natural language processing to highly performant systems deployed at scale in products you use every day. The consequences of this sudden progress extend to almost every industry. Deep learning is applied to a diverse range of important problems across domains as different as medical imaging, agriculture, autonomous driving, education, disaster prevention, and manufacturing. Digital assistants are becoming pervasive on nearly every consumer computing device.

Yet, we believe deep learning is still in its early days. AI is going to represent a much greater wave of disruption than anything that came before it. The technology will continue to make its way to every problem where it can provide some utility—a transformation that will take decades to play out. This is a transformation with profound societal implications, on a global scale, spanning industries and cultures. We strongly believe the only way to ensure such a transformation is beneficial for the peo

### PDF p.85 (printed 61) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.1 A brief history of deep learning frameworks › A brief history of deep learning frameworks
It took a long time for the field to develop robust solutions for all three problems and package those solutions in a reusable form. Since its inception in the 1960s and until the 2000s, autodifferentiation had no practical applications in machine learning—folks who worked with neural networks simply wrote their own gradient logic by hand, usually in a language like C++. Meanwhile, GPU programming was all but impossible. Things started to slowly change in the late 2000s. First, Python and its ecosystem were slowly rising in popularity in the scientific community, gaining traction over MATLAB and C++. Second, NVIDIA released CUDA in 2006, unlocking the possibility of building neural networks that could run on consumer GPUs. The initial focus on CUDA was on physics simulation rather than machine learning, but that didn’t stop machine learning researchers from starting to implement CUDA-based neural networks from 2009 onward. They were typically one-off implementations that ran on a single GPU without any autodifferentiation. The first framework to enable autodifferentiation and GPU computation to train deep learning models was Theano, circa 2009. Theano is the conceptual ancestor of 
