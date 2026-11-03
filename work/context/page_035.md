# Generation bundle for page_035  (PDF page 35, printed page 11)

Page kind: content
Chapter: 1 What is deep learning?
Section: 1.9 What deep learning has achieved so far
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_035.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_035.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p035_e03] (body) law. In addition, deep learning models are trained by iterating over small batches of data, allowing them to be trained on datasets of arbitrary size. (The only bottleneck is the amount of parallel computational power available, which, thanks to Moore’s law, is a fast-moving barrier.)
[p035_e04] (bullet) Versatility and reusability—Unlike many prior machine learning approaches, deep learning models can be trained on additional data without restarting from scratch, making them viable for continuous online learning—an important property for very large production models. Furthermore, trained deep learning models are repurposable and thus reusable: this is the big idea behind “foundation models”—large models trained on humongous amounts of data, which can be used across many new tasks with little retraining, or even none at all.
[p035_e05] (heading_section) 1.8
[p035_e06] (heading_section) The age of generative AI
[p035_e07] (body) Perhaps the most well-known example of deep learning today is the recent wave of generative AI applications—chatbot assistants like ChatGPT, Gemini, and Claude, as well as image generation services like Midjourney. These applications have captured the public imagination with their ability to produce informative or even creative content in response to simple prompts, blurring the lines between human and machine creativity.
[p035_e08] (body) Generative AI is powered by very large “foundation models” that learn to reconstruct the text and image content fed into them—reconstruct a sharp image from a noisy version, predict the next word in a sentence, and so on. This means that the targets from figure 1.8 are taken from the input itself. This is referred to as self-supervised learning, and it enables those models to use vast amounts of unlabeled data. Doing away with the manual data annotations that bottlenecked previous brands of machine learning has unlocked a level of scale never seen before—some of these foundation models have hundreds of billions of parameters and are trained on over 1 petabyte of data, at the cost of tens of millions of dollars. These foundation models operate as a kind of fuzzy database of human knowledge, making them amenable to a very wide range of applications without needing special-purpose programming or retraining. Because they’ve already memorized so much, they can solve new problems merely via prompting—querying the knowledge representations they’ve learned and returning the output most likely to be associated with your prompt. Generative AI only rose to mainstream awareness in 2022, but it has a long history— the earliest experiments with text generation date back to the 1990s. The first edition of this book, released in 2017, already had a hefty chapter titled “Generative AI” that explored the text generation and image generation techniques of the time, while promising the then-outlandish notion that, “soon,” much of the cultural content we consume would be created with the help of AI.
[p035_e09] (heading_section) 1.9
[p035_e10] (heading_section) What deep learning has achieved so far
[p035_e11] (body) Over the past decade, deep learning has achieved nothing short of a technological revolution, starting with remarkable results on perceptual tasks from 2013 to 2017,

## Chapter outline
Chapter: 1 What is deep learning? (PDF pages 25-39)
  - 1.1 Artificial intelligence, machine learning, and deep learning (p.26)
  - 1.2 Artificial intelligence (p.26)
  - 1.3 Machine learning (p.27)
  - 1.4 Learning rules and representations from data (p.28)
  - 1.5 The “deep” in “deep learning” (p.31)
  - 1.6 Understanding how deep learning works, in three figures (p.32)
  - 1.7 What makes deep learning different (p.34)
  - 1.8 The age of generative AI (p.35)
  - 1.9 What deep learning has achieved so far (p.35)
  - 1.10 Beware of the short-term hype (p.36)
  - 1.11 Summer can turn to winter (p.38)
  - 1.12 The promise of AI (p.38)

## Previous page (PDF 34) text, for continuity
Figure 1.9 The loss score is used as a feedback signal to adjust the weights.

number of times (typically tens of passes over thousands of examples), yields weight values that minimize the loss function. A network with a minimal loss is one for which the outputs are as close as they can be to the targets: a trained network. Once again, it’s a simple mechanism that, once scaled, ends up looking like magic.

1.7

What makes deep learning different

Is there anything special about deep neural networks that makes them the “right” approach for companies to invest in and for researchers to flock to? Will we still be using deep neural networks in 20 years?

Deep learning has several properties that justify its status as an AI revolution, and it’s here to stay. We may not be using neural networks many decades from now, but whatever we use will directly inherit from modern deep learning and its core concepts. These important properties can be broadly sorted into three categories:

Simplicity—Deep learning makes problem solving much easier, because it automates what used to be the most crucial step in a machine learning workflow: feature engineering. Previous machine learning techniques—shallow learning—only involved transforming the input data into one or two successive representation spaces, which wasn’t expressive enough for most problems. As such, humans had to go to great lengths to make the initial input data more amenable to processing by these methods: they had to manually engineer good representations for their data. This is called feature engineering. Deep learning, on the other hand, completely automates this step: with deep learning, you learn all features in one pass rather than having to engineer them yourself. This has greatly simplified machine learning workflows, …

## Next page (PDF 36) text, for continuity
then making fast progress on natural language processing tasks from 2017 to 2022, and culminating with a wave of transformative generative AI applications from 2022 to now.

Deep learning has enabled major breakthroughs, all in extremely challenging problems that had long eluded machines:

Fluent and highly versatile chatbots such as ChatGPT and Gemini

Programming assistants like GitHub Copilot

Photorealistic image generation

Human-level image classification

Human-level speech transcription

Human-level handwriting transcription and printed text transcription

Dramatically improved machine translation

Dramatically improved text-to-speech conversion

Human-level autonomous driving, already deployed to the public in Phoenix, San Francisco, Los Angeles, and Austin as of 2025

Improved recommender systems, as used by YouTube, Netflix, or Spotify

Superhuman Go, Chess, and Poker playing

We’re still exploring the full extent of what deep learning can do. We’ve started applying it with great success to a wide variety of problems that were thought to be impossible to solve just a few years ago—automatically transcribing the tens of thousands of ancient manuscripts held in the Vatican Secret Archive, detecting and classifying plant diseases in fields using a simple smartphone, assisting oncologists or radiologists with interpreting medical imaging data, predicting natural disasters such as floods, hurricanes, and even earthquakes. With every milestone, we’re getting closer to an age where deep learning assists us in every activity and every field of human endeavor—science, medicine, manufacturing, energy, transportation, software development, agriculture, and even artistic creation.

1.10 Beware of the short-term hype

This seemingly unstoppable string of successes has led …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.621 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.622 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.589 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.21 — about this book › How this book is organized: A road map
- PDF p.26 — 1 What is deep learning? › 1.2 Artificial intelligence
- PDF p.7 — brief contents › 
- PDF p.8 — contents › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.620 (printed 596) — 20 Conclusions › 20.1 Key concepts in review › 20.1.1 Various approaches to artificial intelligence
descent.

Generative AI is a specific subset of deep learning, where models are capable of generating text, images, videos, or sound. These models tend to be very large— billions of parameters. They’re trained in a self-supervised manner; that is, they’re trained to reconstruct artificially missing or corrupted parts of an input— for instance, denoising images, predicting the next word in a sentence, and so on. This learning process enables the models to learn sophisticated “maps” (embedding manifolds) of their input space, which can be used for sampling new inputs. These models have launched AI into its “consumer” era with the rise of products like ChatGPT or Midjourney.

Even though deep learning is just one among many approaches to machine learning, it isn’t on an equal footing with the others. Deep learning is a breakout success. Here’s why.

### PDF p.525 (printed 501) — 16 Text generation › 16.4 Going further with LLMs › Foundation models
As LLMs venture into different modalities, the “large language model” moniker can become a bit misleading. They do model language, but also images, audio, maybe even structured data. In the next chapter, we will see a distinct architecture, called diffusion models, that works quite differently in terms of underlying structure but has a similar feel—they too are trained on massive amounts of data at “internet scale” with a self-supervised loss.

An umbrella term for models like this is foundation models. More specifically, a foundation model is any model that is trained on broad data (generally using self-supervision at scale) that can be fine-tuned to a wide range of downstream tasks. In general, you can think of a foundation model as learning to reconstruct data pulled from large swaths of the internet, given a partial representation of it. While LLMs are the first and best-known of these models, there are many others. The hallmarks of a foundation model are the self-supervised learning objective (a reconstruction loss) and the fact that these models are not specialized to a single task and can be used for a number of downstream purposes. This is an important and striking shift th

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

### PDF p.8 (printed viii) — contents ›  › contents
preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

### PDF p.18 (printed xviii) — preface ›  › 
of Keras, and over the last decade, the Keras developer community has made remarkable progress in this area. We’ve put deep learning into the hands of millions of people, who, in turn, are using it to solve problems that were, until recently, thought to be unsolvable.

The book you’re holding is another step on the way toward broadening access to the field. We aim to make the concepts behind deep learning and their implementation as approachable as possible. Doing so doesn’t require watering anything down; we believe that there are no difficult ideas in deep learning. This book will start with the very basics of the field and, layer by layer, build up to the cutting-edge generative AI models being deployed today. This is the third edition of Deep Learning with Python. In an effort to make this content as broadly available as possible, we are making this edition available for free, online at https://deeplearningwithpython.io/. We hope you will consider purchasing a physical copy to support the project and read what we consider to be the best presentation of the content. This third edition amounts to a complete rewrite of the entire book, with a broader introduction to today’s popula

### PDF p.490 (printed 466) — 16 Text generation ›  › This chapter covers
A brief history of generative modeling

Training a miniature GPT model from scratch

Using a pretrained Transformer model to build

a chatbot

Building a multimodal model that can describe

images in natural language

When I first claimed that in a not-so-distant future, most of the cultural content we consume would be created with substantial help from AIs, I was met with utter disbelief, even from long-time machine learning practitioners. That was in 2014. Fast-forward a decade, and that disbelief had receded at an incredible speed. Generative AI tools are now common additions to word processors, image editors, and development environments. Prestigious awards are going out to literature and art created with generative models—to considerable controversy and debate.1 It no

1 In 2022, Jason Allen used the image generation software Midjourney to win an award for digital artists, and in 2024, Rie Kudan won one of Japan’s most prestigious literary awards for a novel written with substantial help from generative software.

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › The relevance of deep learning for small-data problems
What qualifies as “enough samples” to train a model is relative—relative to the size and depth of the model you’re trying to train, for starters. It isn’t possible to train a ConvNet to solve a complex problem with just a few tens of samples, but a few hundred can potentially suffice if the model is small and well regularized and the task is simple. Because ConvNets learn local, translation-invariant features, they’re highly data efficient on perceptual problems. Training a ConvNet from scratch on a very small image dataset will still yield reasonable results despite a relative lack of data, without the need for any custom feature engineering. You’ll see this in action in this section.

What’s more, deep learning models are by nature highly repurposable: you can take, say, an image-classification or speech-to-text model trained on a large-scale dataset and reuse it on a significantly different problem with only minor changes. Specifically, in the case of computer vision, many pretrained classification models are publicly available for download and can be used to bootstrap powerful vision models out of very little data. This is one of the greatest strengths of deep learning: feature

### PDF p.589 (printed 565) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.1 Deep learning models struggle to adapt to novelty
Deep learning models are big, parametric curves fitted to large datasets. That’s the source of their power—they’re easy to train, and they scale really well, both in terms of model size and dataset size. But that’s also a source of significant weaknesses. Curve fitting has inherent limitations.
