# Generation bundle for page_611  (PDF page 611, printed page 587)

Page kind: content
Chapter: 19 The future of AI
Section: 19.4 The missing ingredients: Search and symbols
Subsection: 19.4.2 Cognition as a combination of both kinds of abstraction
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_611.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_611.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p611_e03] (code)
```
ls = obj.as_list()
my_list = get_data()
ls_sum = 0
total = 0
ls_entries = 0
num_elems = 0
for n in ls:
for e in my_list:
if n is not None:
if e is not None:
ls_sum += n
total += e
ls_entries += 1
num_elems += 1
avg = ls sum / ls entries
mean = total / num_elems
print('avg:', avg)
update_mean(mean)
```
[p611_e04] (code)
```
def compute_mean(ls):
total = 0
num_elems = 0
for e in ls:
if e is not None:
total += e
num_elems += 1
return total / num_elems
```
[p611_e05] (figure_caption) Figure 19.13 Program-centric analogy identifies and isolates isomorphic substructures across different instances.
[p611_e06] (heading_subsection) 19.4.2 Cognition as a combination of both kinds of abstraction
[p611_e07] (body) Table 19.1 compares these two poles of abstraction side by side.
[p611_e08] (table_caption) Table 19.1 The two poles of abstraction
[p611_e09] (figure_caption) Value-centric abstraction Program-centric abstraction
[p611_e10] (table_text) Relates things by distance Relates things by exact structural match Continuous, grounded in geometry Discrete, grounded in topology Produces abstractions by “averaging” instances Produces abstractions by isolating isomorphic into “prototypes” substructures across instances Underlies perception and intuition Underlies reasoning and planning Immediate, fuzzy, approximative Slow, exact, rigorous Requires a lot of experience to produce reliable Experience efficient: can operate on as few as two results instances
[p611_e11] (body) Everything we do, everything we think, is a combination of these two types of abstraction. You’d be hard pressed to find tasks that only involve one of the two. Even a seemingly “pure perception” task, like recognizing objects in a scene, involves a fair amount of implicit reasoning about the relationships between the objects you’re looking at. And even a seemingly “pure reasoning” task, like finding the proof of a mathematical theorem, involves a good amount of intuition. When a mathematician puts their pen to the paper, they’ve already got a fuzzy vision of the direction in which they’re going. The discrete reasoning steps they take to get to the destination are guided by high-level intuition.

## Figures
- Figure 19.13: Program-centric analogy identifies and isolates isomorphic substructures across different instances.  | labels inside the figure: Instance, Instance, Shared abstraction, Shared abstraction
- Figure None: Value-centric abstraction Program-centric abstraction  | labels inside the figure: (none - image)

## Table extraction (raw, may be noisy - verify against the page image)
[{"bbox": [102.5, 361.4, 473.8, 477.9], "rows": [["Value-centric abstraction", "Program-centric abstraction"], ["Relatesthingsbydistance\nContinuous groundedingeometry\n,\nProducesabstractionsby“averaging”instances\ninto“prototypes”\nUnderliesperceptionandintuition\nImmediate fuzzy approximative\n, ,\nRequiresalotofexperiencetoproducereliable\nresults", "Relatesthingsbyexactstructuralmatch\nDiscrete groundedintopology\n,\nProducesabstractionsbyisolatingisomorphic\nsubstructuresacrossinstances\nUnderliesreasoningandplanning\nSlow exact rigorous\n, ,\nExperienceeffi cient:canoperateonasfewastwo\ninstances"]]}]

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

## Previous page (PDF 610) text, for continuity
Figure 19.12 Value-centric analogy relates instances via a continuous notion of similarity to obtain abstract prototypes.

recognition, perception, and intuition. If you can do a task without thinking about it, you’re heavily relying on value-centric analogies. If you’re watching a movie and you start subconsciously categorizing the different characters into “types,” that’s value-centric abstraction.

Program-centric analogy

Crucially, there’s more to cognition than the kind of immediate, approximate, intuitive categorization that value-centric analogy enables. There’s another type of abstraction-generation mechanism, slower, exact, deliberate: program-centric (or structure-centric) analogy.

In software engineering, you often write different functions or classes that seem to have a lot in common. When you notice these redundancies, you start asking, Could there be a more abstract function that performs the same job that could be reused twice? Could there be an abstract base class that both of your classes could inherit from? The definition of abstraction you’re using here corresponds to program-centric analogy. You’re not trying to compare your classes and functions by how similar they look, the way you’d compare two human faces, via an implicit distance function. Rather, you’re interested in whether there are parts of them that have exactly the same structure. You’re looking for what is called a subgraph isomorphism (see figure 19.13): programs can be represented as graphs of operators, and you’re trying to find subgraphs (program subsets) that are exactly shared across your different programs. This kind of analogy-making via exact structural match within different discrete structures isn’t at all exclusive to specialized fields like computer science, or mathematics— …

## Next page (PDF 612) text, for continuity
These two poles are complementary, and it’s their interleaving that enables extreme generalization. No mind could be complete without both of them.

19.4.3 Why deep learning isn’t a complete answer to abstraction generation

Deep learning is very good at encoding value-centric abstraction, but it has basically no ability to generate program-centric abstraction. Human-like intelligence is a tight interleaving of both types, so we’re literally missing half of what we need—arguably the most important half.

Now, here’s a caveat. So far, I’ve presented each type of abstraction as entirely separate from the other—opposite, even. In practice, however, they’re more of a spectrum: to an extent, you could do reasoning by embedding discrete programs in continuous manifolds—just like you may fit a polynomial function through any set of discrete points, as long as you have enough coefficients. And inversely, you could use discrete programs to emulate continuous distance functions—after all, when you’re doing linear algebra on a computer, you’re working with continuous spaces, entirely via discrete programs that operate on 1s and 0s. However, there are clearly types of problems that are better suited to one or the other. Try to train a deep learning model to sort a list of five numbers, for instance. With the right architecture, it’s not impossible, but it’s an exercise in frustration. You’ll need a massive amount of training data to make it happen—and even then, the model will still make occasional mistakes when presented with new numbers. And if you want to start sorting lists of 10 numbers instead, you’ll need to completely retrain the model— on even more data. Meanwhile, writing a sorting algorithm in Python takes just a few lines, and the resulting program, once validated on a  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.609 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.630 — 20 Conclusions › 20.3 What might lie ahead
- PDF p.602 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.613 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.16 — contents › 
- PDF p.604 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.592 — 19 The future of AI › 19.1 The limitations of deep learning

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.609 (printed 585) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.1 The two poles of abstraction
Abstraction acquisition starts with comparing things to one another. Crucially, there are two distinct ways to compare things, from which arise two different kinds of abstraction and two modes of thinking, each better suited to a different kind of problem. Together, these two poles of abstraction form the basis for all of our thoughts.

The first way to relate things to each other is similarity comparison, which gives rise to value-centric analogies. The second way is exact structural match, which gives rise to program-centric analogies (or structure-centric analogies). In both cases, you start from instances of a thing, and you merge together related instances to produce an abstraction that captures the common elements of the underlying instances. What varies is how you tell that two instances are related and how you merge instances into abstractions. Let’s take a close look at each type.

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 20.3 What might lie ahead
Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.

Future key developments may include

Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, co

### PDF p.609 (printed 585) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › Value-centric analogy
Let’s say you come across a number of different beetles in your backyard, belonging to multiple species. You’ll notice similarities between them. Some will be more similar to one another, and some will be less similar: the notion of similarity is implicitly a smooth, continuous distance function that defines a latent manifold where your instances live. Once you’ve seen enough beetles, you can start clustering more similar instances together and merging them into a set of prototypes that captures the shared visual features of each cluster (figure 19.12). These prototypes are abstract: they don’t look like any specific instance you’ve seen, although they encode properties that are common across all of them. When you encounter a new beetle, you won’t need to compare it to every single beetle you’ve seen before to know what to do with it. You can simply compare it to your handful of prototypes to find the closest prototype—the beetle’s category—and use it to make useful predictions: Is the beetle likely to bite you? Will it eat your apples?

Does this sound familiar? It’s pretty much a description of what unsupervised machine learning (such as the K-means clustering algorithm) does. In

### PDF p.616 (printed 592) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
Deep learning can help program synthesis do the same: although each specific program we’d like to generate might be a fundamentally discrete object that performs non-interpolative data manipulation, evidence so far indicates that the space of all useful programs may look a lot like a continuous manifold. That means that a deep learning model that has been trained on millions of successful program-generation episodes might start to develop solid intuition about the path through program space that the search process should take to go from a specification to the corresponding program—just like a software engineer might have immediate intuition about the overall architecture of the script they’re about to write and about the intermediate functions and classes they should use as stepping stones on the way to the goal. Remember that human reasoning is heavily guided by value-centric abstraction— that is, by pattern recognition and intuition. The same should be true of program synthesis. I expect the general approach of guiding program search via learned heuristics to see increasing research interest over the next 10 to 20 years.

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

### PDF p.613 (printed 589) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.4 An alternative approach to AI: Program synthesis
Figure 19.14 A schematic view of program synthesis: given a program specification and a set of building blocks, a search process assembles the building blocks into candidate programs, which are then tested against the specification. The search continues until a valid program is found.

Table 19.2 Machine learning vs. program synthesis

Machine learning Program synthesis

Model: differentiable parametric function Model: graph of operators from a programming language Engine: gradient descent Engine: discrete search (such as genetic search) Requires a lot of data to produce reliable results Data efficient: can work with a couple of training examples

Program synthesis is how we’re going to add program-centric abstraction capabilities to our AI systems. It’s the missing piece of the puzzle.

### PDF p.618 (printed 594) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
that humans maintain about their surroundings and themselves, and they will be capable of stronger generalization due to their rich algorithmic nature.

In particular, models will blend algorithmic modules providing formal reasoning, search, and abstraction capabilities with geometric modules providing informal intuition and pattern-recognition capabilities. This will achieve a blend of value-centric and program-centric abstraction. AlphaGo or self-driving cars (systems that required a lot of manual software engineering and human-made design decisions) provide an early example of what such a blend of symbolic and geometric AI could look like.

Such models will be grown automatically rather than hardcoded by human engineers, using modular parts stored in a global library of reusable subroutines—a library evolved by learning high-performing models on thousands of previous tasks and datasets. As frequent problem-solving patterns are identified by the meta-learning system, they will be turned into reusable subroutines—much like functions and classes in software engineering—and added to the global library.

The process that searches over possible combinations of subroutines to grow new 
