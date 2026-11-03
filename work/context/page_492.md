# Generation bundle for page_492  (PDF page 492, printed page 468)

Page kind: content
Chapter: 16 Text generation
Section: 16.1 A brief history of sequence generation
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_492.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_492.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p492_e03] (body) Iannis Xenakis, a visionary pioneer of electronic and algorithmic music, beautifully expressed this same idea in the 1960s, in the context of the application of automation technology to music composition:2
[p492_e04] (quote) Freed from tedious calculations, the composer is able to devote himself to the general problems that the new musical form poses and to explore the nooks and crannies of this form while modifying the values of the input data. For example, he may test all instrumental combinations from soloists to chamber orchestras, to large orchestras. With the aid of electronic computers the composer becomes a sort of pilot: he presses the buttons, introduces coordinates, and supervises the controls of a cosmic vessel sailing in the space of sound, across sonic constellations and galaxies that he could formerly glimpse only as a distant dream.
[p492_e05] (body) The potential for generative AI extends well beyond artistic endeavors. In many professions, people create content where pattern recognition is even more apparent: think of summarizing large documents, transcribing speech, editing for typos, or flagging common mistakes in code. These rote tasks play directly to the strengths of deep learning approaches. There is a lot to consider regarding how we choose to deploy AI in the workplace—with real societal implications.
[p492_e06] (body) In the following two chapters, we will explore the potential of deep learning to assist with creation. We will learn to curate latent spaces in text and image domains and pull new content from these spaces. We will start with text, scaling up the idea of a language model we first worked with in the last chapter. These large language models, or LLMs for short, are behind digital assistants like ChatGPT and a quickly growing list of real-world applications.
[p492_e07] (heading_section) 16.1 A brief history of sequence generation
[p492_e08] (body) Until quite recently, the idea of generating sequences from a model was a niche subtopic within machine learning—generative recurrent networks only began to hit the mainstream in 2016. However, these techniques have a fairly long history, starting with the development of the LSTM algorithm in 1997.
[p492_e09] (body) In 2002, Douglas Eck applied LSTM to music generation for the first time, with promising results. Eck became a researcher at Google Brain, and in 2016, he started a new research group called Magenta, which focused on applying modern deep learning techniques to produce engaging music. Sometimes, good ideas take 15 years to get started. In the late 2000s and early 2010s, Alex Graves pioneered the use of recurrent networks for new types of sequence data generation. In particular, some see his 2013 work on applying recurrent mixture density networks to generate human-like handwriting using timeseries of pen positions as a turning point. Graves left a commented-out remark hidden in a 2013 LaTeX file uploaded to the preprint server arXiv: “Generating sequential data is the closest computers get to dreaming.” This work and the notion of machines that dream were significant inspirations when I started developing Keras.
[p492_e10] (footnote) 2 Iannis Xenakis, “Musiques formelles: nouveaux principes formels de composition musicale,” special issue of La Revue musicale, nos. 253-254 (1963).

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

## Previous page (PDF 491) text, for continuity
longer feels like science fiction to consider a world where AI and artistic endeavors are often intertwined.

In any practical sense, AI is nowhere close to rivaling human screenwriters, painters, or composers. But replacing humans need not, and should not, be the point. In many fields, but especially in creative ones, people will use AI to augment their capabilities— more augmented intelligence than artificial intelligence. Much of artistic creation consists of pattern recognition and technical skill. Our perceptual modalities, language, and artwork all have statistical structure, and deep learning models excel at learning this structure. Machine learning models can learn the statistical latent spaces of images, music, and stories, and they can then sample from these spaces, creating new artworks with characteristics similar to those the model has seen in its training data. Such sampling is hardly an act of artistic creation in itself—it’s a mere mathematical operation. Only our interpretation, as human spectators, gives meaning to what the model generates. But in the hands of a skilled artist, algorithmic generation can be steered to become meaningful—and beautiful. Latent space sampling can become a brush that empowers the artist, augments our creative affordances, and expands the space of what we can imagine. It can even make artistic creation more accessible by eliminating the need for technical skill and practice—setting up a new medium of pure expression, factoring art apart from craft.

Figure 16.1 An image generated with the generative image software Midjourney. The prompt was “A hand-drawn, sci-fi landscape of residents living in a building shaped like a red letter K.”

## Next page (PDF 493) text, for continuity
In 2018, a year after the “Attention Is All You Need” paper we discussed in the last chapter, a group of researchers at an organization called OpenAI put out a new paper “Improving Language Understanding by Generative Pre-Training.”3 They combined a few ingredients:

Unsupervised pretraining of a language model—essentially training a model to “guess the next token” in a sequence, as we did with our Shakespeare generator in chapter 15

The Transformer architecture

Textual data on various topics via thousands of self-published books

The authors showed that such a pretrained model could be fine-tuned to achieve state-of-the-art performance on a wide array of text classification tasks—from gauging the similarity of two sentences to answering a multiple-choice question. They called the pretrained model GPT, short for Generative Pretrained Transformer.

GPT didn’t come with any modeling or training advancements. What was interesting about the results was that such a general training setup could beat out more involved techniques across a number of tasks. There was no complex text normalization, no need to customize the model architecture or training data per benchmark, just a lot of pretraining data and compute. In the following years, OpenAI set about scaling this idea with a single-minded focus. The model architecture changed only slightly. Over four years, OpenAI released three versions of GPT, scaling up as follows:

Released in 2018, GPT-1 had 117 million parameters and was trained on 1 billion tokens.

Released in 2019, GPT-2 had 1.5 billion parameters and was trained on more than 10 billion tokens.

Released in 2020, GPT-3 had 175 billion parameters and was trained on somewhere around half a trillion tokens.

The language modeling setup enabled each of these models to …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.35 — 1 What is deep learning? › 1.9 What deep learning has achieved so far
- PDF p.533 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.407 — 14 Text classification › 14.1 A brief history of natural language processing
- PDF p.490 — 16 Text generation › 
- PDF p.15 — contents › 
- PDF p.21 — about this book › How this book is organized: A road map
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.36 — 1 What is deep learning? › 1.10 Beware of the short-term hype

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.35 (printed 11) — 1 What is deep learning? › 1.9 What deep learning has achieved so far › The age of generative AI
Perhaps the most well-known example of deep learning today is the recent wave of generative AI applications—chatbot assistants like ChatGPT, Gemini, and Claude, as well as image generation services like Midjourney. These applications have captured the public imagination with their ability to produce informative or even creative content in response to simple prompts, blurring the lines between human and machine creativity.

Generative AI is powered by very large “foundation models” that learn to reconstruct the text and image content fed into them—reconstruct a sharp image from a noisy version, predict the next word in a sentence, and so on. This means that the targets from figure 1.8 are taken from the input itself. This is referred to as self-supervised learning, and it enables those models to use vast amounts of unlabeled data. Doing away with the manual data annotations that bottlenecked previous brands of machine learning has unlocked a level of scale never seen before—some of these foundation models have hundreds of billions of parameters and are trained on over 1 petabyte of data, at the cost of tens of millions of dollars. These foundation models operate as a kind of fuzzy d

### PDF p.14 (printed xiv) — contents ›  › 
14.5

Sequence models 405

Training a recurrent model 406 ■ Understanding word embeddings 409 ■ Using a word embedding 410 ■ Pretraining a word embedding 414 ■ Using the pretrained embedding for classification 418

15

Language models and the Transformer 421 15.1

The language model 421

Training a Shakespeare language model 422 ■ Generating Shakespeare 426

15.2

Sequence-to-sequence learning 428

English-to-Spanish translation 430 ■ Sequence-to-sequence learning with RNNs 432

15.3

The Transformer architecture 437

Dot-product attention 439 ■ Transformer encoder block 444 Transformer decoder block 446 ■ Sequence-to-sequence learning with a Transformer 448 ■ Embedding positional information 451

15.4

Classification with a pretrained Transformer 454

Pretraining a Transformer encoder 454 ■ Loading a pretrained Transformer 455 ■ Preprocessing IMDb movie reviews 458 Fine-tuning a pretrained Transformer 460

15.5

What makes the Transformer effective? 461 16 Text generation 466 16.1 A brief history of sequence generation 468 16.2 Training a mini-GPT 470

Building the model 473 ■ Pretraining the model 476 Generative decoding 478 ■ Sampling strategies 480

16.3

Using a pretrained LLM

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

### PDF p.490 (printed 466) — 16 Text generation ›  › This chapter covers
A brief history of generative modeling

Training a miniature GPT model from scratch

Using a pretrained Transformer model to build

a chatbot

Building a multimodal model that can describe

images in natural language

When I first claimed that in a not-so-distant future, most of the cultural content we consume would be created with substantial help from AIs, I was met with utter disbelief, even from long-time machine learning practitioners. That was in 2014. Fast-forward a decade, and that disbelief had receded at an incredible speed. Generative AI tools are now common additions to word processors, image editors, and development environments. Prestigious awards are going out to literature and art created with generative models—to considerable controversy and debate.1 It no

1 In 2022, Jason Allen used the image generation software Midjourney to win an award for digital artists, and in 2024, Rie Kudan won one of Japan’s most prestigious literary awards for a novel written with substantial help from generative software.

### PDF p.533 (printed 509) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.1 Sampling from latent spaces of images
The key idea of image generation is to develop a low-dimensional latent space of representations (which, like everything else in deep learning, is a vector space) where any point can be mapped to a “valid” image: an image that looks like the real thing. The module capable of realizing this mapping, taking as input a latent point and outputting an image (a grid of pixels), is usually called a generator, or sometimes a decoder. Once such a latent space has been learned, you can sample points from it, and, by mapping them back to image space, generate images that have never been seen before (see figure 17.1)—the in-betweens of the training images.

Figure 17.1 Using a latent vector space to sample new images

Further, text-conditioning makes it possible to map a space of prompts in natural language to the latent space (see figure 17.2), making it possible to do language-guided image generation—generating pictures that correspond to a text description. This category of models is called text-to-image models.

Interpolating between many training images in the latent space enables such models to generate infinite combinations of visual concepts, including many that no one had explicitly c

### PDF p.620 (printed 596) — 20 Conclusions › 20.1 Key concepts in review › 20.1.1 Various approaches to artificial intelligence
descent.

Generative AI is a specific subset of deep learning, where models are capable of generating text, images, videos, or sound. These models tend to be very large— billions of parameters. They’re trained in a self-supervised manner; that is, they’re trained to reconstruct artificially missing or corrupted parts of an input— for instance, denoising images, predicting the next word in a sentence, and so on. This learning process enables the models to learn sophisticated “maps” (embedding manifolds) of their input space, which can be used for sampling new inputs. These models have launched AI into its “consumer” era with the rise of products like ChatGPT or Midjourney.

Even though deep learning is just one among many approaches to machine learning, it isn’t on an equal footing with the others. Deep learning is a breakout success. Here’s why.

### PDF p.16 (printed xvi) — contents ›  › 
19.3

How to build intelligence 576

The kaleidoscope hypothesis 577 ■ The essence of intelligence: Abstraction acquisition and recombination 578 The importance of setting the right target 578 ■ A new target: On-the-fly adaptation 580 ■ ARC Prize 581 ■ The test-time adaptation era 582 ■ ARC-AGI 2 583

19.4

The missing ingredients: Search and symbols 584

The two poles of abstraction 585 ■ Cognition as a combination of both kinds of abstraction 587 ■ Why deep learning isn’t a complete answer to abstraction generation 588 ■ An alternative approach to AI: Program synthesis 589 ■ Blending deep learning and program synthesis 590 ■ Modular component recombination and lifelong learning 592 ■ The long-term vision 593

20

Conclusions 595 20.1

Key concepts in review 595

Various approaches to artificial intelligence 596 ■ What makes deep learning special within the field of machine learning 596 How to think about deep learning 597 ■ Key enabling technologies 598 ■ The universal machine learning workflow 599 ■ Key network architectures 600

20.2

Limitations of deep learning 605 20.3 What might lie ahead 606 20.4 Staying up to date in a fast-moving field 607

Practice on real-world problem

### PDF p.532 (printed 508) — 17 Image generation › 17.1 Deep learning for image generation › This chapter covers
Variational autoencoders

Diffusion models

Using a pretrained text-to-image model

Exploring the latent image spaces learned by

text-to-image models

The most popular and successful application of creative AI today is image generation: learning latent visual spaces and sampling from them to create entirely new pictures, interpolated from real ones—pictures of imaginary people, imaginary places, imaginary cats and dogs, and so on.
