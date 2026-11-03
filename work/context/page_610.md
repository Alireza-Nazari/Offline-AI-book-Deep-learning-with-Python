# Generation bundle for page_610  (PDF page 610, printed page 586)

Page kind: content
Chapter: 19 The future of AI
Section: 19.4 The missing ingredients: Search and symbols
Subsection: 19.4.1 The two poles of abstraction
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_610.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_610.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p610_e03] (figure_caption) Figure 19.12 Value-centric analogy relates instances via a continuous notion of similarity to obtain abstract prototypes.
[p610_e04] (body) recognition, perception, and intuition. If you can do a task without thinking about it, you’re heavily relying on value-centric analogies. If you’re watching a movie and you start subconsciously categorizing the different characters into “types,” that’s value-centric abstraction.
[p610_e05] (heading_subsubsection) Program-centric analogy
[p610_e06] (body) Crucially, there’s more to cognition than the kind of immediate, approximate, intuitive categorization that value-centric analogy enables. There’s another type of abstraction-generation mechanism, slower, exact, deliberate: program-centric (or structure-centric) analogy.
[p610_e07] (body) In software engineering, you often write different functions or classes that seem to have a lot in common. When you notice these redundancies, you start asking, Could there be a more abstract function that performs the same job that could be reused twice? Could there be an abstract base class that both of your classes could inherit from? The definition of abstraction you’re using here corresponds to program-centric analogy. You’re not trying to compare your classes and functions by how similar they look, the way you’d compare two human faces, via an implicit distance function. Rather, you’re interested in whether there are parts of them that have exactly the same structure. You’re looking for what is called a subgraph isomorphism (see figure 19.13): programs can be represented as graphs of operators, and you’re trying to find subgraphs (program subsets) that are exactly shared across your different programs. This kind of analogy-making via exact structural match within different discrete structures isn’t at all exclusive to specialized fields like computer science, or mathematics— you’re constantly using it without noticing. It underlies reasoning, planning, and the general concept of rigor (as opposed to intuition). Any time you’re thinking about objects connected to each other by a discrete network of relationships (rather than a continuous similarity function), you’re using program-centric analogies.

## Figures
- Figure 19.12: Value-centric analogy relates instances via a continuous notion of similarity to obtain abstract prototypes.  | labels inside the figure: Instances in the wild, Similarity clustering, Abstract prototypes

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

## Previous page (PDF 609) text, for continuity
I’ve said that intelligence consists of two components: abstraction acquisition and abstraction recombination. They are tightly coupled—what kind of abstractions you manipulate determines how, and how well, you can recombine them. Deep learning models only manipulate abstractions stored via parametric curves, fitted via gradient descent. Could there be a better way?

19.4.1 The two poles of abstraction

Abstraction acquisition starts with comparing things to one another. Crucially, there are two distinct ways to compare things, from which arise two different kinds of abstraction and two modes of thinking, each better suited to a different kind of problem. Together, these two poles of abstraction form the basis for all of our thoughts.

The first way to relate things to each other is similarity comparison, which gives rise to value-centric analogies. The second way is exact structural match, which gives rise to program-centric analogies (or structure-centric analogies). In both cases, you start from instances of a thing, and you merge together related instances to produce an abstraction that captures the common elements of the underlying instances. What varies is how you tell that two instances are related and how you merge instances into abstractions. Let’s take a close look at each type.

Value-centric analogy

Let’s say you come across a number of different beetles in your backyard, belonging to multiple species. You’ll notice similarities between them. Some will be more similar to one another, and some will be less similar: the notion of similarity is implicitly a smooth, continuous distance function that defines a latent manifold where your instances live. Once you’ve seen enough beetles, you can start clustering more similar instances together and merging them into …

## Next page (PDF 611) text, for continuity
Figure 19.13 Program-centric analogy identifies and isolates isomorphic substructures across different instances.

19.4.2 Cognition as a combination of both kinds of abstraction

Table 19.1 compares these two poles of abstraction side by side.

Table 19.1 The two poles of abstraction

Value-centric abstraction Program-centric abstraction

Relates things by distance Relates things by exact structural match Continuous, grounded in geometry Discrete, grounded in topology Produces abstractions by “averaging” instances Produces abstractions by isolating isomorphic into “prototypes” substructures across instances Underlies perception and intuition Underlies reasoning and planning Immediate, fuzzy, approximative Slow, exact, rigorous Requires a lot of experience to produce reliable Experience efficient: can operate on as few as two results instances

Everything we do, everything we think, is a combination of these two types of abstraction. You’d be hard pressed to find tasks that only involve one of the two. Even a seemingly “pure perception” task, like recognizing objects in a scene, involves a fair amount of implicit reasoning about the relationships between the objects you’re looking at. And even a seemingly “pure reasoning” task, like finding the proof of a mathematical theorem, involves a good amount of intuition. When a mathematician puts their pen to the paper, they’ve already got a fuzzy vision of the direction in which they’re going. The discrete reasoning steps they take to get to the destination are guided by high-level intuition.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.612 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.618 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.630 — 20 Conclusions › 20.3 What might lie ahead
- PDF p.615 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.593 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.613 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.617 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.616 (printed 592) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
Deep learning can help program synthesis do the same: although each specific program we’d like to generate might be a fundamentally discrete object that performs non-interpolative data manipulation, evidence so far indicates that the space of all useful programs may look a lot like a continuous manifold. That means that a deep learning model that has been trained on millions of successful program-generation episodes might start to develop solid intuition about the path through program space that the search process should take to go from a specification to the corresponding program—just like a software engineer might have immediate intuition about the overall architecture of the script they’re about to write and about the intermediate functions and classes they should use as stepping stones on the way to the goal. Remember that human reasoning is heavily guided by value-centric abstraction— that is, by pattern recognition and intuition. The same should be true of program synthesis. I expect the general approach of guiding program search via learned heuristics to see increasing research interest over the next 10 to 20 years.

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 20.3 What might lie ahead
Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.

Future key developments may include

Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, co

### PDF p.612 (printed 588) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.3 Why deep learning isn’t a complete answer to abstraction generation
Deep learning is very good at encoding value-centric abstraction, but it has basically no ability to generate program-centric abstraction. Human-like intelligence is a tight interleaving of both types, so we’re literally missing half of what we need—arguably the most important half.

### PDF p.618 (printed 594) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
that humans maintain about their surroundings and themselves, and they will be capable of stronger generalization due to their rich algorithmic nature.

In particular, models will blend algorithmic modules providing formal reasoning, search, and abstraction capabilities with geometric modules providing informal intuition and pattern-recognition capabilities. This will achieve a blend of value-centric and program-centric abstraction. AlphaGo or self-driving cars (systems that required a lot of manual software engineering and human-made design decisions) provide an early example of what such a blend of symbolic and geometric AI could look like.

Such models will be grown automatically rather than hardcoded by human engineers, using modular parts stored in a global library of reusable subroutines—a library evolved by learning high-performing models on thousands of previous tasks and datasets. As frequent problem-solving patterns are identified by the meta-learning system, they will be turned into reusable subroutines—much like functions and classes in software engineering—and added to the global library.

The process that searches over possible combinations of subroutines to grow new 

### PDF p.612 (printed 588) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.3 Why deep learning isn’t a complete answer to abstraction generation
Now, here’s a caveat. So far, I’ve presented each type of abstraction as entirely separate from the other—opposite, even. In practice, however, they’re more of a spectrum: to an extent, you could do reasoning by embedding discrete programs in continuous manifolds—just like you may fit a polynomial function through any set of discrete points, as long as you have enough coefficients. And inversely, you could use discrete programs to emulate continuous distance functions—after all, when you’re doing linear algebra on a computer, you’re working with continuous spaces, entirely via discrete programs that operate on 1s and 0s. However, there are clearly types of problems that are better suited to one or the other. Try to train a deep learning model to sort a list of five numbers, for instance. With the right architecture, it’s not impossible, but it’s an exercise in frustration. You’ll need a massive amount of training data to make it happen—and even then, the model will still make occasional mistakes when presented with new numbers. And if you want to start sorting lists of 10 numbers instead, you’ll need to completely retrain the model— on even more data. Meanwhile, writing a sorting a

### PDF p.613 (printed 589) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.4 An alternative approach to AI: Program synthesis
Figure 19.14 A schematic view of program synthesis: given a program specification and a set of building blocks, a search process assembles the building blocks into candidate programs, which are then tested against the specification. The search continues until a valid program is found.

Table 19.2 Machine learning vs. program synthesis

Machine learning Program synthesis

Model: differentiable parametric function Model: graph of operators from a programming language Engine: gradient descent Engine: discrete search (such as genetic search) Requires a lot of data to produce reliable results Data efficient: can work with a couple of training examples

Program synthesis is how we’re going to add program-centric abstraction capabilities to our AI systems. It’s the missing piece of the puzzle.

### PDF p.71 (printed 47) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › Automatic differentiation with computation graphs
A useful way to think about backpropagation is in terms of computation graphs. A computation graph is the data structure at the heart of the deep learning revolution. It’s a directed acyclic graph of operations—in our case, tensor operations. For instance, figure 2.21 is the graph representation of our first model.

Computation graphs have been an extremely successful abstraction in computer science because they enable us to treat computation as data: a computable expression is encoded as a machine-readable data structure that can be used as the input or output of another program. For instance, you could imagine a program that receives a computation graph and returns a new computation graph that implements a large-scale distributed version of the same computation—this would mean that you could distribute any computation without having to write the distribution logic yourself. Or imagine … a program that receives a computation graph and can automatically generate the derivative of the expression it represents. It’s much easier to do these things if your computation is expressed as an explicit graph data structure rather than, say, lines of ASCII characters in a `.py` file. To explai

### PDF p.293 (printed 269) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › Modularity, hierarchy, and reuse
If you want to make a complex system simpler, there’s a universal recipe you can apply: just structure your amorphous soup of complexity into modules, organize the modules into a hierarchy, and start reusing the same modules in multiple places as appropriate (“reuse” is another word for abstraction). That’s the modularity-hierarchy-reuse (MHR) formula (see figure 9.1), and it underlies system architecture across pretty much every domain where the term architecture is used. It’s at the heart of the organization of any system of meaningful complexity, whether it’s a cathedral, your own body, the US Navy, or the Keras codebase.

If you’re a software engineer, you’re already keenly familiar with these principles: an effective codebase is one that is modular, hierarchical, and where you don’t reimplement the same thing twice but instead rely on reusable classes and functions. If you factor your code by following these principles, you could say you’re doing “software architecture.” Deep learning itself is simply the application of this recipe to continuous optimization via gradient descent: you take a classic optimization technique (gradient descent
