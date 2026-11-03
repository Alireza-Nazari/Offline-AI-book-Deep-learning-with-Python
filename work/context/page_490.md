# Generation bundle for page_490  (PDF page 490, printed page 466)

Page kind: chapter_opener
Chapter: 16 Text generation
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_490.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_490.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p490_e01] (chapter_number) 16
[p490_e02] (chapter_title) Text generation
[p490_e03] (chapter_covers) This chapter covers
[p490_e04] (bullet) A brief history of generative modeling
[p490_e05] (bullet) Training a miniature GPT model from scratch
[p490_e06] (bullet) Using a pretrained Transformer model to build
[p490_e07] (bullet) a chatbot
[p490_e08] (bullet) Building a multimodal model that can describe
[p490_e09] (bullet) images in natural language
[p490_e10] (body) When I first claimed that in a not-so-distant future, most of the cultural content we consume would be created with substantial help from AIs, I was met with utter disbelief, even from long-time machine learning practitioners. That was in 2014. Fast-forward a decade, and that disbelief had receded at an incredible speed. Generative AI tools are now common additions to word processors, image editors, and development environments. Prestigious awards are going out to literature and art created with generative models—to considerable controversy and debate.1 It no
[p490_e11] (footnote) 1 In 2022, Jason Allen used the image generation software Midjourney to win an award for digital artists, and in 2024, Rie Kudan won one of Japan’s most prestigious literary awards for a novel written with substantial help from generative software.

## Chapter outline
Chapter: 16 Text generation (PDF pages 490-531)
  - 16.1 A brief history of sequence generation (p.492)
  - 16.2 Training a mini-GPT (p.494)
      - 16.2.1 Building the model (p.497)
      - 16.2.2 Pretraining the model (p.500)
      - 16.2.3 Generative decoding (p.502)
      - 16.2.4 Sampling strategies (p.504)
  - 16.3 Using a pretrained LLM (p.508)
      - 16.3.1 Text generation with the Gemma model (p.509)
      - 16.3.2 Instruction fine-tuning (p.512)
      - 16.3.3 Low-Rank Adaptation (LoRA) (p.514)
  - 16.4 Going further with LLMs (p.519)
      - 16.4.1 Reinforcement Learning with Human Feedback (RLHF) (p.519)
      - 16.4.2 Multimodal LLMs (p.522)
      - 16.4.3 Retrieval Augmented Generation (RAG) (p.525)
      - 16.4.4 “Reasoning” models (p.526)
  - 16.5 Where are LLMs heading next? (p.528)

## Previous page (PDF 489) text, for continuity
The Transformer is a sequence modeling architecture that uses attention as the only mechanism to pass information across a sequence: – The Transformer works by stacking blocks of alternating attention and two-layer feedforward networks. – The Transformer can scale to many parameters and lots of training data while still improving accuracy in the language modeling problem. – Unlike RNNs, the Transformer involves no sequence-length loops at training time, making the model much easier to train in parallel across many machines. – A Transformer encoder uses bidirectional attention to build a rich representation of a sequence. – A Transformer decoder uses causal attention to predict the next word in a language model setup.

## Next page (PDF 491) text, for continuity
longer feels like science fiction to consider a world where AI and artistic endeavors are often intertwined.

In any practical sense, AI is nowhere close to rivaling human screenwriters, painters, or composers. But replacing humans need not, and should not, be the point. In many fields, but especially in creative ones, people will use AI to augment their capabilities— more augmented intelligence than artificial intelligence. Much of artistic creation consists of pattern recognition and technical skill. Our perceptual modalities, language, and artwork all have statistical structure, and deep learning models excel at learning this structure. Machine learning models can learn the statistical latent spaces of images, music, and stories, and they can then sample from these spaces, creating new artworks with characteristics similar to those the model has seen in its training data. Such sampling is hardly an act of artistic creation in itself—it’s a mere mathematical operation. Only our interpretation, as human spectators, gives meaning to what the model generates. But in the hands of a skilled artist, algorithmic generation can be steered to become meaningful—and beautiful. Latent space sampling can become a brush that empowers the artist, augments our creative affordances, and expands the space of what we can imagine. It can even make artistic creation more accessible by eliminating the need for technical skill and practice—setting up a new medium of pure expression, factoring art apart from craft.

Figure 16.1 An image generated with the generative image software Midjourney. The prompt was “A hand-drawn, sci-fi landscape of residents living in a building shaped like a red letter K.”

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.493 — 16 Text generation › 16.1 A brief history of sequence generation
- PDF p.35 — 1 What is deep learning? › 1.9 What deep learning has achieved so far
- PDF p.36 — 1 What is deep learning? › 1.10 Beware of the short-term hype
- PDF p.508 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.492 — 16 Text generation › 16.1 A brief history of sequence generation
- PDF p.534 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.502 — 16 Text generation › 16.2 Training a mini-GPT

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.35 (printed 11) — 1 What is deep learning? › 1.9 What deep learning has achieved so far › The age of generative AI
Perhaps the most well-known example of deep learning today is the recent wave of generative AI applications—chatbot assistants like ChatGPT, Gemini, and Claude, as well as image generation services like Midjourney. These applications have captured the public imagination with their ability to produce informative or even creative content in response to simple prompts, blurring the lines between human and machine creativity.

Generative AI is powered by very large “foundation models” that learn to reconstruct the text and image content fed into them—reconstruct a sharp image from a noisy version, predict the next word in a sentence, and so on. This means that the targets from figure 1.8 are taken from the input itself. This is referred to as self-supervised learning, and it enables those models to use vast amounts of unlabeled data. Doing away with the manual data annotations that bottlenecked previous brands of machine learning has unlocked a level of scale never seen before—some of these foundation models have hundreds of billions of parameters and are trained on over 1 petabyte of data, at the cost of tens of millions of dollars. These foundation models operate as a kind of fuzzy d

### PDF p.493 (printed 469) — 16 Text generation › 16.1 A brief history of sequence generation › 
The language modeling setup enabled each of these models to generate text, and the developers at OpenAI noticed that with each leap in scale, the quality of this generative output shot up substantially.

With GPT-1, the model’s generative capabilities were mostly a by-product of its pretraining and not the primary focus. They evaluated the model by fine-tuning it with an extra dense layer for classification, as we did with RoBERTa in the last chapter. With GPT-2, the authors noticed that you could prompt the model with a few examples of a task and generate quality output without any fine-tuning. For instance, you could prompt the model with the following to receive a French translation of the word cheese:

3 Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever, “Improving Language Understanding by Generative Pre-Training,” (2018), https://mng.bz/GweD.

### PDF p.532 (printed 508) — 17 Image generation › 17.1 Deep learning for image generation › This chapter covers
Variational autoencoders

Diffusion models

Using a pretrained text-to-image model

Exploring the latent image spaces learned by

text-to-image models

The most popular and successful application of creative AI today is image generation: learning latent visual spaces and sampling from them to create entirely new pictures, interpolated from real ones—pictures of imaginary people, imaginary places, imaginary cats and dogs, and so on.

### PDF p.493 (printed 469) — 16 Text generation › 16.1 A brief history of sequence generation › 
In 2018, a year after the “Attention Is All You Need” paper we discussed in the last chapter, a group of researchers at an organization called OpenAI put out a new paper “Improving Language Understanding by Generative Pre-Training.”3 They combined a few ingredients:

Unsupervised pretraining of a language model—essentially training a model to “guess the next token” in a sequence, as we did with our Shakespeare generator in chapter 15

The Transformer architecture

Textual data on various topics via thousands of self-published books

The authors showed that such a pretrained model could be fine-tuned to achieve state-of-the-art performance on a wide array of text classification tasks—from gauging the similarity of two sentences to answering a multiple-choice question. They called the pretrained model GPT, short for Generative Pretrained Transformer.

GPT didn’t come with any modeling or training advancements. What was interesting about the results was that such a general training setup could beat out more involved techniques across a number of tasks. There was no complex text normalization, no need to customize the model architecture or training data per benchmark, just a lot of pre

### PDF p.36 (printed 12) — 1 What is deep learning? › 1.10 Beware of the short-term hype › 
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

We’re still exploring the full extent of what deep learning can do. We’ve started applying it with great success to a wide variety of problems that were thought to be impossible to solve just a few years ago—automatically transcribing the tens of thousands of ancient manuscripts held in the Vatican

### PDF p.620 (printed 596) — 20 Conclusions › 20.1 Key concepts in review › 20.1.1 Various approaches to artificial intelligence
descent.

Generative AI is a specific subset of deep learning, where models are capable of generating text, images, videos, or sound. These models tend to be very large— billions of parameters. They’re trained in a self-supervised manner; that is, they’re trained to reconstruct artificially missing or corrupted parts of an input— for instance, denoising images, predicting the next word in a sentence, and so on. This learning process enables the models to learn sophisticated “maps” (embedding manifolds) of their input space, which can be used for sampling new inputs. These models have launched AI into its “consumer” era with the rise of products like ChatGPT or Midjourney.

Even though deep learning is just one among many approaches to machine learning, it isn’t on an equal footing with the others. Deep learning is a breakout success. Here’s why.

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

### PDF p.502 (printed 478) — 16 Text generation › 16.2 Training a mini-GPT › 16.2.3 Generative decoding
To sample some output from our model, we can follow the approach we used to generate Shakespeare or Spanish translations in chapter 15. We feed a prompt of fixed tokens into the model. For each position in the input sequence, the model outputs a probability distribution over the entire vocabulary for the next token. By selecting the most likely next token at the last location, adding it to our sequence, and then repeating this process, we are able to generate a new sequence, one token at a time.

Listing 16.9 A simple generation function for the mini-GPT model
