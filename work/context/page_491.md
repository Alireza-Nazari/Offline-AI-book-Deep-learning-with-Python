# Generation bundle for page_491  (PDF page 491, printed page 467)

Page kind: content
Chapter: 16 Text generation
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_491.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_491.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p491_e04] (body) longer feels like science fiction to consider a world where AI and artistic endeavors are often intertwined.
[p491_e05] (body) In any practical sense, AI is nowhere close to rivaling human screenwriters, painters, or composers. But replacing humans need not, and should not, be the point. In many fields, but especially in creative ones, people will use AI to augment their capabilities— more augmented intelligence than artificial intelligence. Much of artistic creation consists of pattern recognition and technical skill. Our perceptual modalities, language, and artwork all have statistical structure, and deep learning models excel at learning this structure. Machine learning models can learn the statistical latent spaces of images, music, and stories, and they can then sample from these spaces, creating new artworks with characteristics similar to those the model has seen in its training data. Such sampling is hardly an act of artistic creation in itself—it’s a mere mathematical operation. Only our interpretation, as human spectators, gives meaning to what the model generates. But in the hands of a skilled artist, algorithmic generation can be steered to become meaningful—and beautiful. Latent space sampling can become a brush that empowers the artist, augments our creative affordances, and expands the space of what we can imagine. It can even make artistic creation more accessible by eliminating the need for technical skill and practice—setting up a new medium of pure expression, factoring art apart from craft.
[p491_e06] (figure_caption) Figure 16.1 An image generated with the generative image software Midjourney. The prompt was “A hand-drawn, sci-fi landscape of residents living in a building shaped like a red letter K.”

## Figures
- Figure 16.1: An image generated with the generative image software Midjourney. The prompt was “A hand-drawn, sci-fi landscape of residents living in a building shaped like a red letter K.”  | labels inside the figure: (none - image)

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

## Previous page (PDF 490) text, for continuity
Text generation

This chapter covers

A brief history of generative modeling

Training a miniature GPT model from scratch

Using a pretrained Transformer model to build

a chatbot

Building a multimodal model that can describe

images in natural language

When I first claimed that in a not-so-distant future, most of the cultural content we consume would be created with substantial help from AIs, I was met with utter disbelief, even from long-time machine learning practitioners. That was in 2014. Fast-forward a decade, and that disbelief had receded at an incredible speed. Generative AI tools are now common additions to word processors, image editors, and development environments. Prestigious awards are going out to literature and art created with generative models—to considerable controversy and debate.1 It no

1 In 2022, Jason Allen used the image generation software Midjourney to win an award for digital artists, and in 2024, Rie Kudan won one of Japan’s most prestigious literary awards for a novel written with substantial help from generative software.

## Next page (PDF 492) text, for continuity
Iannis Xenakis, a visionary pioneer of electronic and algorithmic music, beautifully expressed this same idea in the 1960s, in the context of the application of automation technology to music composition:2

Freed from tedious calculations, the composer is able to devote himself to the general problems that the new musical form poses and to explore the nooks and crannies of this form while modifying the values of the input data. For example, he may test all instrumental combinations from soloists to chamber orchestras, to large orchestras. With the aid of electronic computers the composer becomes a sort of pilot: he presses the buttons, introduces coordinates, and supervises the controls of a cosmic vessel sailing in the space of sound, across sonic constellations and galaxies that he could formerly glimpse only as a distant dream.

The potential for generative AI extends well beyond artistic endeavors. In many professions, people create content where pattern recognition is even more apparent: think of summarizing large documents, transcribing speech, editing for typos, or flagging common mistakes in code. These rote tasks play directly to the strengths of deep learning approaches. There is a lot to consider regarding how we choose to deploy AI in the workplace—with real societal implications.

In the following two chapters, we will explore the potential of deep learning to assist with creation. We will learn to curate latent spaces in text and image domains and pull new content from these spaces. We will start with text, scaling up the idea of a language model we first worked with in the last chapter. These large language models, or LLMs for short, are behind digital assistants like ChatGPT and a quickly growing list of real-world applications.

16.1 A brief history of  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.533 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.532 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.15 — contents › 
- PDF p.534 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.561 — 17 Image generation › 17.3 Text-to-image models
- PDF p.615 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.600 — 19 The future of AI › 19.3 How to build intelligence

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.532 (printed 508) — 17 Image generation › 17.1 Deep learning for image generation › This chapter covers
Variational autoencoders

Diffusion models

Using a pretrained text-to-image model

Exploring the latent image spaces learned by

text-to-image models

The most popular and successful application of creative AI today is image generation: learning latent visual spaces and sampling from them to create entirely new pictures, interpolated from real ones—pictures of imaginary people, imaginary places, imaginary cats and dogs, and so on.

### PDF p.534 (printed 510) — 17 Image generation › 17.1 Deep learning for image generation › 
Of course, there are still challenges to overcome. Like with all deep learning models, the latent space doesn’t encode a consistent model of the physical world, so you might occasionally see hands with extra fingers, incoherent lighting, or garbled objects. The coherence of generated images is still an area of active research. In the case of figure 17.2, despite having seen tens of thousands of images of people riding bikes, the model doesn’t understand in a human sense what it means to ride a bike—concepts like pedaling, steering, or maintaining upright balance. That’s why your bike-riding horse is unlikely to get depicted pedaling with its hind legs in a believable manner, the way a human artist would draw it.

There’s a range of different strategies for learning such latent spaces of image representations, each with its own characteristics. The most common types of image generation models are

Diffusion models

Variational autoencoders (VAEs)

Generative adversarial networks (GANs)

While previous editions of this book covered GANs, they have gradually fallen out of fashion in recent years and have been all but replaced by diffusion models. In this edition, we’ll cover both VAEs

### PDF p.533 (printed 509) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.1 Sampling from latent spaces of images
The key idea of image generation is to develop a low-dimensional latent space of representations (which, like everything else in deep learning, is a vector space) where any point can be mapped to a “valid” image: an image that looks like the real thing. The module capable of realizing this mapping, taking as input a latent point and outputting an image (a grid of pixels), is usually called a generator, or sometimes a decoder. Once such a latent space has been learned, you can sample points from it, and, by mapping them back to image space, generate images that have never been seen before (see figure 17.1)—the in-betweens of the training images.

Figure 17.1 Using a latent vector space to sample new images

Further, text-conditioning makes it possible to map a space of prompts in natural language to the latent space (see figure 17.2), making it possible to do language-guided image generation—generating pictures that correspond to a text description. This category of models is called text-to-image models.

Interpolating between many training images in the latent space enables such models to generate infinite combinations of visual concepts, including many that no one had explicitly c

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

### PDF p.15 (printed xv) — contents ›  › 
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

The limitations of deep l

### PDF p.615 (printed 591) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
broader than what can be represented with current deep learning models, and some of these programs could achieve superior generalization power. Importantly, such programs will not be differentiable end to end, although specific modules will remain differentiable, and thus will need to be generated via a combination of discrete program search and gradient descent.

We’ll move away from having, on one hand, hardcoded algorithmic intelligence (handcrafted software) and, on the other hand, learned geometric intelligence (deep learning). Instead, we’ll have a blend of formal algorithmic modules that provide reasoning and abstraction capabilities and geometric modules that provide informal intuition and pattern-recognition capabilities (figure 19.15). The entire system will be learned with little or no human involvement. This should dramatically expand the scope of problems that can be solved with machine learning—the space of programs that we can generate automatically, given appropriate training data. Systems like AlphaGo—or even RNNs—can be seen as a prehistoric ancestor of such hybrid algorithmic-geometric models.

Figure 19.15 A learned program relying on both geometric primitives (

### PDF p.35 (printed 11) — 1 What is deep learning? › 1.9 What deep learning has achieved so far › The age of generative AI
Perhaps the most well-known example of deep learning today is the recent wave of generative AI applications—chatbot assistants like ChatGPT, Gemini, and Claude, as well as image generation services like Midjourney. These applications have captured the public imagination with their ability to produce informative or even creative content in response to simple prompts, blurring the lines between human and machine creativity.

Generative AI is powered by very large “foundation models” that learn to reconstruct the text and image content fed into them—reconstruct a sharp image from a noisy version, predict the next word in a sentence, and so on. This means that the targets from figure 1.8 are taken from the input itself. This is referred to as self-supervised learning, and it enables those models to use vast amounts of unlabeled data. Doing away with the manual data annotations that bottlenecked previous brands of machine learning has unlocked a level of scale never seen before—some of these foundation models have hundreds of billions of parameters and are trained on over 1 petabyte of data, at the cost of tens of millions of dollars. These foundation models operate as a kind of fuzzy d

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 20.3 What might lie ahead
Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.

Future key developments may include

Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, co
