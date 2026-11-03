# Generation bundle for page_621  (PDF page 621, printed page 597)

Page kind: content
Chapter: 20 Conclusions
Section: 20.1 Key concepts in review
Subsection: 20.1.3 How to think about deep learning
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_621.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_621.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p621_e03] (body) to extract from perceptual data almost anything a human could. Hence, it’s sometimes said that deep learning has “solved perception”—although that’s true only for a fairly narrow definition of perception.
[p621_e04] (body) Due to its unprecedented technical successes, deep learning has singlehandedly brought about the third and by far the largest AI summer: a period of intense interest, investment, and hype in the field of AI. As this book is being written, we’re in the middle of it. Whether this period will end in the near future and what happens after it ends are topics of debate. One thing is certain: in stark contrast with previous AI summers, deep learning has provided enormous business value to both large and small technology companies and has become a huge consumer success, enabling human-level speech recognition, chatbot assistants, photorealistic image generation, human-level machine translation, and more. The hype may (and likely will) recede, but the sustained economic and technological impact of deep learning will remain. In that sense, deep learning could be analogous to the internet: it may be overly hyped up for a few years, but in the longer term, it will still be a major revolution that will transform our economy and our lives. One reason I’m particularly optimistic about deep learning is that even if we were to make no further technological progress in the next decade, deploying existing algorithms to every applicable problem would be a game changer for most industries. Deep learning is nothing short of a revolution, and progress is currently happening at an incredibly fast rate due to an exponential investment in resources and headcount. From where we stand, the future looks bright, although short-term expectations are somewhat overoptimistic; deploying deep learning to the full extent of its potential will likely take multiple decades.
[p621_e05] (heading_subsection) 20.1.3 How to think about deep learning
[p621_e06] (body) The most surprising thing about deep learning is how simple it is. Fifteen years ago, no one expected that we would achieve such amazing results on machine-perception and natural language processing problems by using simple parametric models trained with gradient descent. Now it turns out that all you need is sufficiently large parametric models trained with gradient descent on sufficiently many examples. As Feynman once said about the universe, “It’s not complicated, it’s just a lot of it.”1
[p621_e07] (body) In deep learning, everything is a vector; that is, everything is a point in a geometric space. Model inputs (text, images, and so on) and targets are first vectorized—turned into an initial input vector space and target vector space. Each layer in a deep learning model operates one simple geometric transformation on the data that goes through it. Together, the chain of layers in the model forms one complex geometric transformation, broken down into a series of simple ones. This complex transformation attempts to map the input space to the target space, one point at a time. This transformation is parameterized by the weights of the layers, which are iteratively updated based on how well the model is currently performing. A key characteristic of this geometric
[p621_e08] (footnote) 1 Richard Feynman, interview, The World from Another Point of View, Yorkshire Television, 1972.

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

## Previous page (PDF 620) text, for continuity
20.1.1 Various approaches to artificial intelligence

First, deep learning isn’t synonymous with artificial intelligence (AI), or even with machine learning:

Artificial intelligence (AI) is an ancient, broad field that can generally be understood as “all attempts to automate human cognitive processes.” This can range from the very basic, such as an Excel spreadsheet, to the very advanced, like a humanoid robot that can walk and talk.

Machine learning is a specific subfield of AI that aims at automatically developing programs (called models) purely from exposure to training data. This process of turning data into a program is called learning. Although machine learning has been around for a long time, it only started to take off in the 1990s, before becoming the dominant form of AI in the 2000s.

Deep learning is one of many branches of machine learning, where the models are long chains of geometric transformations, applied one after the other. These operations are structured into modules called layers: deep learning models are typically stacks of layers—or, more generally, graphs of layers. These layers are parameterized by weights, which are the parameters learned during training. The knowledge of a model is stored in its weights, and the process of learning consists of finding “good values” for these weights—values that minimize a loss function. Because the chain of geometric transformations considered is differentiable, updating the weights to minimize the loss function is done efficiently via gradient

descent.

Generative AI is a specific subset of deep learning, where models are capable of generating text, images, videos, or sound. These models tend to be very large— billions of parameters. They’re trained in a self-supervised manner; that is, they’re trained to  …

## Next page (PDF 622) text, for continuity
transformation is that it must be differentiable, which is required for us to be able to learn its parameters via gradient descent. Intuitively, this means the geometric morphing from inputs to outputs must be smooth and continuous—a significant constraint.

The entire process of applying this complex geometric transformation to the input data can be visualized in 3D by imagining a person trying to uncrumple a paper ball: the crumpled paper ball is the manifold of the input data that the model starts with. Each movement operated by the person on the paper ball is similar to a simple geometric transformation operated by one layer. The full uncrumpling gesture sequence is the complex transformation of the entire model. Deep learning models are mathematical machines for uncrumpling complicated manifolds of high-dimensional data. That’s the magic of deep learning—turning meaning into vectors, into geometric spaces, and then incrementally learning complex geometric transformations that map one space to another. All you need are spaces of sufficiently high dimensionality to capture the full scope of the relationships found in the original data. The whole thing hinges on two core ideas: that meaning is derived from the pairwise relationship between things (between words in a language, between pixels in an image, and so on) and that these relationships can be captured by a distance function. But note that whether the brain implements meaning via geometric spaces is an entirely separate question. Vector spaces are efficient to work with from a computational standpoint, but different data structures for intelligence can easily be envisioned—in particular, graphs. Neural networks initially emerged from the idea of using graphs as a way to encode meaning, which is why they’re named …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.35 — 1 What is deep learning? › 1.9 What deep learning has achieved so far
- PDF p.600 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.34 — 1 What is deep learning? › 1.7 What makes deep learning different
- PDF p.36 — 1 What is deep learning? › 1.10 Beware of the short-term hype
- PDF p.32 — 1 What is deep learning? › 1.6 Understanding how deep learning works, in three figures
- PDF p.63 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.629 — 20 Conclusions › 20.2 Limitations of deep learning
- PDF p.38 — 1 What is deep learning? › 1.12 The promise of AI

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.34 (printed 10) — 1 What is deep learning? › 1.7 What makes deep learning different › What makes deep learning different
Is there anything special about deep neural networks that makes them the “right” approach for companies to invest in and for researchers to flock to? Will we still be using deep neural networks in 20 years?

Deep learning has several properties that justify its status as an AI revolution, and it’s here to stay. We may not be using neural networks many decades from now, but whatever we use will directly inherit from modern deep learning and its core concepts. These important properties can be broadly sorted into three categories:

Simplicity—Deep learning makes problem solving much easier, because it automates what used to be the most crucial step in a machine learning workflow: feature engineering. Previous machine learning techniques—shallow learning—only involved transforming the input data into one or two successive representation spaces, which wasn’t expressive enough for most problems. As such, humans had to go to great lengths to make the initial input data more amenable to processing by these methods: they had to manually engineer good representations for their data. This is called feature engineering. Deep learning, on the other hand, completely automates this step: with de

### PDF p.25 (printed 1) — 1 What is deep learning? ›  › This chapter covers
High-level definitions of fundamental concepts

A soft introduction to the principles behind

machine learning

Deep learning’s rising popularity and future

potential

Over the past decade, artificial intelligence (AI) has been a subject of intense media hype. Machine learning, deep learning, and AI come up in countless articles, often outside of technology-minded publications. We’re promised a future of intelligent chatbots, self-driving cars, and virtual assistants—a future sometimes painted in a grim light and other times as utopian, where human jobs will be scarce and most economic activity will be handled by robots or AI agents. For a practitioner of machine learning, it’s important to be able to recognize the signal amid the noise, so that you can tell world-changing developments from overhyped press releases. Our future is at stake, and it’s one in which you have an active role to play: after reading this book, you’ll be one of those who can develop these AI systems. So let’s

### PDF p.8 (printed viii) — contents ›  › contents
preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

### PDF p.35 (printed 11) — 1 What is deep learning? › 1.9 What deep learning has achieved so far › What deep learning has achieved so far
Over the past decade, deep learning has achieved nothing short of a technological revolution, starting with remarkable results on perceptual tasks from 2013 to 2017,

### PDF p.600 (printed 576) — 19 The future of AI › 19.3 How to build intelligence › 
However, the deep learning paradigm has remained limited to cognitive automation: The “intelligence” label in “artificial intelligence” has been a category error. It would be more accurate to call our field “artificial cognition,” with “cognitive automation” and “artificial intelligence” being two nearly independent subfields within it. In this subdivision, AI would be a greenfield where almost everything remains to be discovered. Now, I don’t mean to diminish the achievements of deep learning. Cognitive automation is incredibly useful, and the way deep learning models are capable of automating tasks from exposure to data alone represents an especially powerful form of cognitive automation, far more practical and versatile than explicit programming. Doing this well is a game changer for essentially every industry. But it’s still a long way from human (or animal) intelligence. Our models, so far, can only perform local generalization: they map space X to space Y via a smooth geometric transform learned from a dense sampling of X-to-Y data points, and any disruption within spaces X or Y invalidates this mapping. They can only generalize to new situations that stay similar to past dat

### PDF p.17 (printed xvii) — preface ›  › preface
If you’ve picked up this book, you’re probably aware of the extraordinary progress that deep learning has brought to the field of artificial intelligence. In just a handful of years, we went from near-unusable computer vision and natural language processing to highly performant systems deployed at scale in products you use every day. The consequences of this sudden progress extend to almost every industry. Deep learning is applied to a diverse range of important problems across domains as different as medical imaging, agriculture, autonomous driving, education, disaster prevention, and manufacturing. Digital assistants are becoming pervasive on nearly every consumer computing device.

Yet, we believe deep learning is still in its early days. AI is going to represent a much greater wave of disruption than anything that came before it. The technology will continue to make its way to every problem where it can provide some utility—a transformation that will take decades to play out. This is a transformation with profound societal implications, on a global scale, spanning industries and cultures. We strongly believe the only way to ensure such a transformation is beneficial for the peo

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

### PDF p.38 (printed 14) — 1 What is deep learning? › 1.12 The promise of AI › 1.12 The promise of AI
Although we may have unrealistic short-term expectations for AI, the long-term picture is looking bright. We’re only getting started in applying deep learning to many important problems for which it could prove transformative, from medical diagnoses to digital assistants.
