# Generation bundle for page_616  (PDF page 616, printed page 592)

Page kind: content
Chapter: 19 The future of AI
Section: 19.4 The missing ingredients: Search and symbols
Subsection: 19.4.6 Modular component recombination and lifelong learning
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_616.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_616.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p616_e03] (body) Deep learning can help program synthesis do the same: although each specific program we’d like to generate might be a fundamentally discrete object that performs non-interpolative data manipulation, evidence so far indicates that the space of all useful programs may look a lot like a continuous manifold. That means that a deep learning model that has been trained on millions of successful program-generation episodes might start to develop solid intuition about the path through program space that the search process should take to go from a specification to the corresponding program—just like a software engineer might have immediate intuition about the overall architecture of the script they’re about to write and about the intermediate functions and classes they should use as stepping stones on the way to the goal. Remember that human reasoning is heavily guided by value-centric abstraction— that is, by pattern recognition and intuition. The same should be true of program synthesis. I expect the general approach of guiding program search via learned heuristics to see increasing research interest over the next 10 to 20 years.
[p616_e04] (heading_subsection) 19.4.6 Modular component recombination and lifelong learning
[p616_e05] (body) If models become more complex and are built on top of richer algorithmic primitives, then this increased complexity will require higher reuse between tasks, rather than training a new model from scratch every time we have a new task or a new dataset. Many datasets don’t contain enough information for us to develop a new, complex model from scratch, and it will be necessary to use information from previously encountered datasets (much as you don’t learn English from scratch every time you open a new book—that would be impossible). Training models from scratch on every new task is also inefficient due to the large overlap between the current tasks and previously encountered tasks.
[p616_e06] (body) With modern foundation models, we’re starting to move closer to a world where AI systems possess enormous amounts of acquired knowledge and skills and can bring them to bear on whatever comes their way. But LLMs are missing a key ingredient: recombination. LLMs are very good at fetching and reapplying memorized functions, but they’re not yet able to recombine those functions on the fly into brand-new programs adapted to the situation at hand. They are, in fact, entirely incapable of performing function composition, as investigated in a recent paper by Dziri et al.5 What’s more, the kind of functions they learn aren’t sufficiently abstract or modular, making them a poor fit for recombination in the first place. Remember how we pointed out that LLMs have low accuracy when adding large integers? You probably wouldn’t want to build your next codebase on top of such brittle functions. To solve compositional generalization, we’re going to need to reuse robust program components like the functions and classes found in human programming languages. These components will be evolved specifically for modular reuse in a new context—unlike the patterns that LLMs memorize. And our AIs will recombine them on the fly to synthesize
[p616_e07] (footnote) 5 Nouha Dziri et. al., “Faith and Fate: Limits of Transformers on Compositionality,” Proceedings of the 37th International Conference on Neural Information Processing Systems (2023), https://arxiv.org/abs/2305.18654.

## Chapter outline
Chapter: 19 The future of AI (PDF pages 588-618)
  - 19.1 The limitations of deep learning (p.588)
      - 19.1.1 Deep learning models struggle to adapt to novelty (p.589)
      - 19.1.2 Deep learning models are highly sensitive to phrasing and other distractors (p.591)
      - 19.1.3 Deep learning models struggle to learn generalizable programs (p.593)
      - 19.1.4 The risk of anthropomorphizing machine-learning models (p.593)
  - 19.2 Scale isn’t all you need (p.594)
      - 19.2.1 Automatons vs. intelligent agents (p.595)
      - 19.2.2 Local generalization vs. extreme generalization (p.597)
      - 19.2.3 The purpose of intelligence (p.599)
      - 19.2.4 Climbing the spectrum of generalization (p.599)
  - 19.3 How to build intelligence (p.600)
      - 19.3.1 The kaleidoscope hypothesis (p.601)
      - 19.3.2 The essence of intelligence: Abstraction acquisition and recombination (p.602)
      - 19.3.3 The importance of setting the right target (p.602)
      - 19.3.4 A new target: On-the-fly adaptation (p.604)
      - 19.3.5 ARC Prize (p.605)
      - 19.3.6 The test-time adaptation era (p.606)
      - 19.3.7 ARC-AGI 2 (p.607)
  - 19.4 The missing ingredients: Search and symbols (p.608)
      - 19.4.1 The two poles of abstraction (p.609)
      - 19.4.2 Cognition as a combination of both kinds of abstraction (p.611)
      - 19.4.3 Why deep learning isn’t a complete answer to abstraction generation (p.612)
      - 19.4.4 An alternative approach to AI: Program synthesis (p.613)
      - 19.4.5 Blending deep learning and program synthesis (p.614)
      - 19.4.6 Modular component recombination and lifelong learning (p.616)
      - 19.4.7 The long-term vision (p.618)

## Previous page (PDF 615) text, for continuity
broader than what can be represented with current deep learning models, and some of these programs could achieve superior generalization power. Importantly, such programs will not be differentiable end to end, although specific modules will remain differentiable, and thus will need to be generated via a combination of discrete program search and gradient descent.

We’ll move away from having, on one hand, hardcoded algorithmic intelligence (handcrafted software) and, on the other hand, learned geometric intelligence (deep learning). Instead, we’ll have a blend of formal algorithmic modules that provide reasoning and abstraction capabilities and geometric modules that provide informal intuition and pattern-recognition capabilities (figure 19.15). The entire system will be learned with little or no human involvement. This should dramatically expand the scope of problems that can be solved with machine learning—the space of programs that we can generate automatically, given appropriate training data. Systems like AlphaGo—or even RNNs—can be seen as a prehistoric ancestor of such hybrid algorithmic-geometric models.

Figure 19.15 A learned program relying on both geometric primitives (pattern recognition, intuition) and algorithmic primitives (reasoning, search, memory)

Using deep learning to guide program search

Today, program synthesis faces a major obstacle: it’s tremendously inefficient. To caricature, typical program synthesis techniques work by trying every possible program in a search space until it finds one that matches the specification provided. As the complexity of a program specification increases, or as the vocabulary of primitives used to write programs expands, the program search process runs into what’s known as combinatorial explosion: the set of possibl …

## Next page (PDF 617) text, for continuity
new programs adapted to the current task. Crucially, libraries of such reusable components will be built through the cumulative experience of all instances of our AIs and will then be accessible by all in perpetuity. Any single problem encountered by our AIs would only need to be solved once—making them constantly self-improving.

Think of the process of software development today: once an engineer solves a specific problem (HTTP queries in Python, for instance), they package it as an abstract, reusable library, accessible by anyone on the planet. Engineers who face a similar problem in the future will be able to search for existing libraries, download one, and use it in their own project. In a similar way, in the future, meta-learning systems will be able to assemble new programs by sifting through a global library of high-level reusable blocks. When the system finds itself developing similar program subroutines for several different tasks, it can come up with an abstract, reusable version of the subroutine and store it in the global library (see figure 19.16). These subroutines can be either geometric (deep learning modules with pretrained representations) or algorithmic (closer to the libraries that contemporary software engineers manipulate).

Figure 19.16 A meta-learner capable of quickly developing task-specific models using reusable primitives (both algorithmic and geometric), thus achieving extreme generalization

19.4.7 The long-term vision

In short, here’s my long-term vision for AI:

Models will be more like programs and will have capabilities that go far beyond the continuous geometric transformations of the input data we currently work with. These programs will arguably be much closer to the abstract mental models

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.630 — 20 Conclusions › 20.3 What might lie ahead
- PDF p.593 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.613 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.614 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.618 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.612 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.589 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.602 — 19 The future of AI › 19.3 How to build intelligence

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 20.3 What might lie ahead
Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.

Future key developments may include

Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, co

### PDF p.613 (printed 589) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.4 An alternative approach to AI: Program synthesis
Figure 19.14 A schematic view of program synthesis: given a program specification and a set of building blocks, a search process assembles the building blocks into candidate programs, which are then tested against the specification. The search continues until a valid program is found.

Table 19.2 Machine learning vs. program synthesis

Machine learning Program synthesis

Model: differentiable parametric function Model: graph of operators from a programming language Engine: gradient descent Engine: discrete search (such as genetic search) Requires a lot of data to produce reliable results Data efficient: can work with a couple of training examples

Program synthesis is how we’re going to add program-centric abstraction capabilities to our AI systems. It’s the missing piece of the puzzle.

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

### PDF p.613 (printed 589) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.4 An alternative approach to AI: Program synthesis
Until 2024, AI systems capable of genuine discrete reasoning were all hardcoded by human programmers—for instance, software that relies on search algorithms, graph manipulation, and formal logic. In the test-time adaptation (TTA) era, this is finally starting to change. A branch of TTA that is especially promising is program synthesis—a field that is still very niche today, but that I expect to take off in a big way over the next few decades.

Program synthesis consists of automatically generating simple programs by using a search algorithm (possibly genetic search, as in genetic programming) to explore a large space of possible programs (see figure 19.14). The search stops when a program is found that matches the required specifications, often provided as a set of input-output pairs. This is highly reminiscent of machine learning: given training data provided as input-output pairs, we find a program that matches inputs to outputs and can generalize to new inputs. The difference is that instead of learning parameter values in a hardcoded program (a neural network), we generate source code via a discrete search process (see table 19.2).

### PDF p.618 (printed 594) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
that humans maintain about their surroundings and themselves, and they will be capable of stronger generalization due to their rich algorithmic nature.

In particular, models will blend algorithmic modules providing formal reasoning, search, and abstraction capabilities with geometric modules providing informal intuition and pattern-recognition capabilities. This will achieve a blend of value-centric and program-centric abstraction. AlphaGo or self-driving cars (systems that required a lot of manual software engineering and human-made design decisions) provide an early example of what such a blend of symbolic and geometric AI could look like.

Such models will be grown automatically rather than hardcoded by human engineers, using modular parts stored in a global library of reusable subroutines—a library evolved by learning high-performing models on thousands of previous tasks and datasets. As frequent problem-solving patterns are identified by the meta-learning system, they will be turned into reusable subroutines—much like functions and classes in software engineering—and added to the global library.

The process that searches over possible combinations of subroutines to grow new 

### PDF p.614 (printed 590) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.5 Blending deep learning and program synthesis
Of course, deep learning isn’t going anywhere. Program synthesis isn’t its replacement; it is its complement. It’s the hemisphere that has been so far missing from our artificial brains. We’re going to be using both, in combination. There are two major ways this will take place:

Developing systems that integrate both deep learning modules and discrete algorithmic modules

Using deep learning to make the program search process itself more efficient

Let’s review each of these possible avenues.

### PDF p.612 (printed 588) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.3 Why deep learning isn’t a complete answer to abstraction generation
Now, here’s a caveat. So far, I’ve presented each type of abstraction as entirely separate from the other—opposite, even. In practice, however, they’re more of a spectrum: to an extent, you could do reasoning by embedding discrete programs in continuous manifolds—just like you may fit a polynomial function through any set of discrete points, as long as you have enough coefficients. And inversely, you could use discrete programs to emulate continuous distance functions—after all, when you’re doing linear algebra on a computer, you’re working with continuous spaces, entirely via discrete programs that operate on 1s and 0s. However, there are clearly types of problems that are better suited to one or the other. Try to train a deep learning model to sort a list of five numbers, for instance. With the right architecture, it’s not impossible, but it’s an exercise in frustration. You’ll need a massive amount of training data to make it happen—and even then, the model will still make occasional mistakes when presented with new numbers. And if you want to start sorting lists of 10 numbers instead, you’ll need to completely retrain the model— on even more data. Meanwhile, writing a sorting a

### PDF p.612 (printed 588) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.3 Why deep learning isn’t a complete answer to abstraction generation
Meanwhile, writing a sorting algorithm in Python takes just a few lines, and the resulting program, once validated on a couple more examples, will work every time on lists of any size. That’s pretty strong generalization: going from a couple of demonstration examples and test examples to a program that can successfully process literally any list of numbers. In reverse, perception problems are a terrible fit for discrete reasoning processes. Try to write a pure-Python program to classify MNIST digits, without using any machine learning technique: you’re in for a ride. You’ll find yourself painstakingly coding functions that can detect the number of closed loops in a digit, the coordinates of the center of mass of a digit, and so on. After thousands of lines of code, you might achieve 90% test accuracy. In this case, fitting a parametric model is much simpler; it can better utilize the large amount of data that’s available, and it achieves much more robust results. If you have lots of data and you’re faced with a problem where the manifold hypothesis applies, go with deep learning. For this reason, it’s unlikely that we’ll see the rise of an approach that would reduce reasoning probl
