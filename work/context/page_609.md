# Generation bundle for page_609  (PDF page 609, printed page 585)

Page kind: content
Chapter: 19 The future of AI
Section: 19.4 The missing ingredients: Search and symbols
Subsection: 19.4.1 The two poles of abstraction
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_609.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_609.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p609_e03] (body) I’ve said that intelligence consists of two components: abstraction acquisition and abstraction recombination. They are tightly coupled—what kind of abstractions you manipulate determines how, and how well, you can recombine them. Deep learning models only manipulate abstractions stored via parametric curves, fitted via gradient descent. Could there be a better way?
[p609_e04] (heading_subsection) 19.4.1 The two poles of abstraction
[p609_e05] (body) Abstraction acquisition starts with comparing things to one another. Crucially, there are two distinct ways to compare things, from which arise two different kinds of abstraction and two modes of thinking, each better suited to a different kind of problem. Together, these two poles of abstraction form the basis for all of our thoughts.
[p609_e06] (body) The first way to relate things to each other is similarity comparison, which gives rise to value-centric analogies. The second way is exact structural match, which gives rise to program-centric analogies (or structure-centric analogies). In both cases, you start from instances of a thing, and you merge together related instances to produce an abstraction that captures the common elements of the underlying instances. What varies is how you tell that two instances are related and how you merge instances into abstractions. Let’s take a close look at each type.
[p609_e07] (heading_subsubsection) Value-centric analogy
[p609_e08] (body) Let’s say you come across a number of different beetles in your backyard, belonging to multiple species. You’ll notice similarities between them. Some will be more similar to one another, and some will be less similar: the notion of similarity is implicitly a smooth, continuous distance function that defines a latent manifold where your instances live. Once you’ve seen enough beetles, you can start clustering more similar instances together and merging them into a set of prototypes that captures the shared visual features of each cluster (figure 19.12). These prototypes are abstract: they don’t look like any specific instance you’ve seen, although they encode properties that are common across all of them. When you encounter a new beetle, you won’t need to compare it to every single beetle you’ve seen before to know what to do with it. You can simply compare it to your handful of prototypes to find the closest prototype—the beetle’s category—and use it to make useful predictions: Is the beetle likely to bite you? Will it eat your apples?
[p609_e09] (body) Does this sound familiar? It’s pretty much a description of what unsupervised machine learning (such as the K-means clustering algorithm) does. In general, all of modern machine learning, unsupervised or not, works by learning latent manifolds that describe a space of instances, encoded via prototypes. (Remember the ConvNet features you visualized in chapter 10? They were visual prototypes.) Value-centric analogy is the kind of analogy-making that enables deep learning models to perform local generalization. It’s also what many of your own cognitive abilities run on. As a human, you perform value-centric analogies all the time. It’s the type of abstraction that underlies pattern

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

## Previous page (PDF 608) text, for continuity
And so, in March 2025, the ARC Prize Foundation introduced ARC-AGI-2. It kept the exact same format as the first version but significantly improved the task content. The new iteration was designed to raise the bar, incorporating tasks that demand more complex reasoning chains and are inherently more resistant to exhaustive search methods. The goal was to create a benchmark where computational efficiency becomes a more critical factor for success, pushing systems toward more genuinely intelligent, efficient strategies rather than simply exploring billions of possibilities. While most ARC-AGI-1 tasks could be solved almost instantaneously by a human without requiring much cognitive effort, all tasks in ARC-AGI-2 require some amount of deliberate thinking (see figure 19.11)—for instance, the average time for task completion among human test takers in our experiments was 5 minutes.

Figure 19.11 Typical ARC-AGI-1 task (left) vs. typical ARC-AGI-2 task (right)

The initial AI testing results on ARC-AGI 2 were sobering: even o3 struggled significantly with this new set of challenges, its scores plummeting back into the low double digits when constrained to reasonable computational budgets. As for base LLMs? Their performance on ARC-AGI-2 was effectively back at 0%—fitting, as base LLMs don’t possess fluid intelligence. The challenge of building AI with truly efficient, human-like fluid intelligence is still far from solved. We’re going to need something beyond current TTA techniques.

19.4 The missing ingredients: Search and symbols

What would it take to fully solve ARC-AGI, in particular version 2? Hopefully, this challenge will get you thinking. That’s the entire point of ARC-AGI: to give you a goal of a different kind, that will nudge you in a new direction—hopefully, a p …

## Next page (PDF 610) text, for continuity
Figure 19.12 Value-centric analogy relates instances via a continuous notion of similarity to obtain abstract prototypes.

recognition, perception, and intuition. If you can do a task without thinking about it, you’re heavily relying on value-centric analogies. If you’re watching a movie and you start subconsciously categorizing the different characters into “types,” that’s value-centric abstraction.

Program-centric analogy

Crucially, there’s more to cognition than the kind of immediate, approximate, intuitive categorization that value-centric analogy enables. There’s another type of abstraction-generation mechanism, slower, exact, deliberate: program-centric (or structure-centric) analogy.

In software engineering, you often write different functions or classes that seem to have a lot in common. When you notice these redundancies, you start asking, Could there be a more abstract function that performs the same job that could be reused twice? Could there be an abstract base class that both of your classes could inherit from? The definition of abstraction you’re using here corresponds to program-centric analogy. You’re not trying to compare your classes and functions by how similar they look, the way you’d compare two human faces, via an implicit distance function. Rather, you’re interested in whether there are parts of them that have exactly the same structure. You’re looking for what is called a subgraph isomorphism (see figure 19.13): programs can be represented as graphs of operators, and you’re trying to find subgraphs (program subsets) that are exactly shared across your different programs. This kind of analogy-making via exact structural match within different discrete structures isn’t at all exclusive to specialized fields like computer science, or mathematics— …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.611 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.612 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.602 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.630 — 20 Conclusions › 20.3 What might lie ahead
- PDF p.16 — contents › 
- PDF p.617 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.618 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.611 (printed 587) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.2 Cognition as a combination of both kinds of abstraction
Table 19.1 compares these two poles of abstraction side by side.

Table 19.1 The two poles of abstraction

Value-centric abstraction Program-centric abstraction

Relates things by distance Relates things by exact structural match Continuous, grounded in geometry Discrete, grounded in topology Produces abstractions by “averaging” instances Produces abstractions by isolating isomorphic into “prototypes” substructures across instances Underlies perception and intuition Underlies reasoning and planning Immediate, fuzzy, approximative Slow, exact, rigorous Requires a lot of experience to produce reliable Experience efficient: can operate on as few as two results instances

Everything we do, everything we think, is a combination of these two types of abstraction. You’d be hard pressed to find tasks that only involve one of the two. Even a seemingly “pure perception” task, like recognizing objects in a scene, involves a fair amount of implicit reasoning about the relationships between the objects you’re looking at. And even a seemingly “pure reasoning” task, like finding the proof of a mathematical theorem, involves a good amount of intuition. When a mathematician puts their pen to the pa

### PDF p.602 (printed 578) — 19 The future of AI › 19.3 How to build intelligence › 19.3.2 The essence of intelligence: Abstraction acquisition and recombination
Intelligence is the ability to mine your experience to identify these atoms of meaning that can seemingly be reused across many different situations—the core beads of the kaleidoscope. Once extracted, they’re called abstractions. Whenever you encounter a new situation, you make sense of it by recombining on the fly abstractions from your collections, to weave a brand new “model” adapted to the situation.

This process consists of two key parts:

Abstraction acquisition—Efficiently extracting compact, reusable abstractions from a stream of experience or data. This involves identifying underlying structures, principles, or invariants.

On-the-fly recombination—Efficiently selecting and recombining these abstractions in novel ways to model new problems and situations, even ones far removed from past experience.

The emphasis on efficiency is crucial. How intelligent you are is determined by how efficiently you can acquire good abstractions from limited experience and how efficiently you can recombine them to navigate uncertainty and novelty. If you need hundreds of thousands of hours of practice to acquire a skill, you are not very intelligent. If you need to enumerate every possible 

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

### PDF p.611 (printed 587) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
Figure 19.13 Program-centric analogy identifies and isolates isomorphic substructures across different instances.

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 20.3 What might lie ahead
Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.

Future key developments may include

Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, co

### PDF p.612 (printed 588) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.3 Why deep learning isn’t a complete answer to abstraction generation
Deep learning is very good at encoding value-centric abstraction, but it has basically no ability to generate program-centric abstraction. Human-like intelligence is a tight interleaving of both types, so we’re literally missing half of what we need—arguably the most important half.

### PDF p.616 (printed 592) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
Deep learning can help program synthesis do the same: although each specific program we’d like to generate might be a fundamentally discrete object that performs non-interpolative data manipulation, evidence so far indicates that the space of all useful programs may look a lot like a continuous manifold. That means that a deep learning model that has been trained on millions of successful program-generation episodes might start to develop solid intuition about the path through program space that the search process should take to go from a specification to the corresponding program—just like a software engineer might have immediate intuition about the overall architecture of the script they’re about to write and about the intermediate functions and classes they should use as stepping stones on the way to the goal. Remember that human reasoning is heavily guided by value-centric abstraction— that is, by pattern recognition and intuition. The same should be true of program synthesis. I expect the general approach of guiding program search via learned heuristics to see increasing research interest over the next 10 to 20 years.

### PDF p.602 (printed 578) — 19 The future of AI › 19.3 How to build intelligence › 19.3.2 The essence of intelligence: Abstraction acquisition and recombination
So how can we move beyond these limitations?
