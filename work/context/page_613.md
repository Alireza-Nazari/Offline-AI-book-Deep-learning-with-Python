# Generation bundle for page_613  (PDF page 613, printed page 589)

Page kind: content
Chapter: 19 The future of AI
Section: 19.4 The missing ingredients: Search and symbols
Subsection: 19.4.4 An alternative approach to AI: Program synthesis
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_613.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_613.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p613_e03] (heading_subsection) 19.4.4 An alternative approach to AI: Program synthesis
[p613_e04] (body) Until 2024, AI systems capable of genuine discrete reasoning were all hardcoded by human programmers—for instance, software that relies on search algorithms, graph manipulation, and formal logic. In the test-time adaptation (TTA) era, this is finally starting to change. A branch of TTA that is especially promising is program synthesis—a field that is still very niche today, but that I expect to take off in a big way over the next few decades.
[p613_e05] (body) Program synthesis consists of automatically generating simple programs by using a search algorithm (possibly genetic search, as in genetic programming) to explore a large space of possible programs (see figure 19.14). The search stops when a program is found that matches the required specifications, often provided as a set of input-output pairs. This is highly reminiscent of machine learning: given training data provided as input-output pairs, we find a program that matches inputs to outputs and can generalize to new inputs. The difference is that instead of learning parameter values in a hardcoded program (a neural network), we generate source code via a discrete search process (see table 19.2).
[p613_e06] (code)
```
Input:
[3, 5, 1, 2, 7]
Output: [1, 2, 3, 5, 7]
Input:
[8, 5, 2, 9, 1, 13]
Output: [1, 2, 5, 8, 9, 13]
```
[p613_e07] (code)
```
+
=
if
==
fo
r
-
else
*
+=
-
=
*=
```
[p613_e08] (figure_caption) Figure 19.14 A schematic view of program synthesis: given a program specification and a set of building blocks, a search process assembles the building blocks into candidate programs, which are then tested against the specification. The search continues until a valid program is found.
[p613_e09] (table_caption) Table 19.2 Machine learning vs. program synthesis
[p613_e10] (figure_caption) Machine learning Program synthesis
[p613_e11] (table_text) Model: differentiable parametric function Model: graph of operators from a programming language Engine: gradient descent Engine: discrete search (such as genetic search) Requires a lot of data to produce reliable results Data efficient: can work with a couple of training examples
[p613_e12] (body) Program synthesis is how we’re going to add program-centric abstraction capabilities to our AI systems. It’s the missing piece of the puzzle.

## Figures
- Figure 19.14: A schematic view of program synthesis: given a program specification and a set of building blocks, a search process assembles the building blocks into candidate programs, which are then tested against the specification. The search continues until a valid program is found.  | labels inside the figure: Search process, Vocabulary of, building blocks, Candidate
- Figure None: Machine learning Program synthesis  | labels inside the figure: (none - image)

## Table extraction (raw, may be noisy - verify against the page image)
[{"bbox": [66.0, 510.6, 474.0, 566.5], "rows": [["Machinelearning", "Programsynthesis"], ["Model:differentiableparametricfunction\nEngine:gradientdescent\nRequiresalotofdatatoproducereliableresults", "Model:graphofoperatorsfromaprogramminglanguage\nEngine:discretesearch(suchasgeneticsearch)\nDataeffi cient:canworkwithacoupleoftrainingexamples"]]}]

## Equation candidates (formulas written inline in the text)
- [p613_e07] +
=
if
==
fo
r
-
else
*
+=
-
=
*=

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

## Previous page (PDF 612) text, for continuity
These two poles are complementary, and it’s their interleaving that enables extreme generalization. No mind could be complete without both of them.

19.4.3 Why deep learning isn’t a complete answer to abstraction generation

Deep learning is very good at encoding value-centric abstraction, but it has basically no ability to generate program-centric abstraction. Human-like intelligence is a tight interleaving of both types, so we’re literally missing half of what we need—arguably the most important half.

Now, here’s a caveat. So far, I’ve presented each type of abstraction as entirely separate from the other—opposite, even. In practice, however, they’re more of a spectrum: to an extent, you could do reasoning by embedding discrete programs in continuous manifolds—just like you may fit a polynomial function through any set of discrete points, as long as you have enough coefficients. And inversely, you could use discrete programs to emulate continuous distance functions—after all, when you’re doing linear algebra on a computer, you’re working with continuous spaces, entirely via discrete programs that operate on 1s and 0s. However, there are clearly types of problems that are better suited to one or the other. Try to train a deep learning model to sort a list of five numbers, for instance. With the right architecture, it’s not impossible, but it’s an exercise in frustration. You’ll need a massive amount of training data to make it happen—and even then, the model will still make occasional mistakes when presented with new numbers. And if you want to start sorting lists of 10 numbers instead, you’ll need to completely retrain the model— on even more data. Meanwhile, writing a sorting algorithm in Python takes just a few lines, and the resulting program, once validated on a  …

## Next page (PDF 614) text, for continuity
19.4.5 Blending deep learning and program synthesis

Of course, deep learning isn’t going anywhere. Program synthesis isn’t its replacement; it is its complement. It’s the hemisphere that has been so far missing from our artificial brains. We’re going to be using both, in combination. There are two major ways this will take place:

Developing systems that integrate both deep learning modules and discrete algorithmic modules

Using deep learning to make the program search process itself more efficient

Let’s review each of these possible avenues.

Integrating deep learning modules and algorithmic modules into hybrid systems

Today, many of the most powerful AI systems are hybrid: they use both deep learning models and handcrafted symbol-manipulation programs. In DeepMind’s AlphaGo, for example, most of the intelligence on display is designed and hardcoded by human programmers (such as Monte Carlo Tree Search). Learning from data happens only in specialized submodules (value networks and policy networks). Or consider the Waymo self-driving car: it’s able to handle a large variety of situations because it maintains a model of the world around it—a literal 3D model—full of assumptions hardcoded by human engineers. This model is constantly updated via deep learning perception modules (powered by Keras) that interface it with the surroundings of the car.

For both of these systems—AlphaGo and self-driving vehicles—the combination of human-created discrete programs and learned continuous models is what unlocks a level of performance that would be impossible with either approach in isolation, such as an end-to-end deep net or a piece of software without machine learning elements. So far, the discrete algorithmic elements of such hybrid systems are painstakingly hardcoded by hum …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.615 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.630 — 20 Conclusions › 20.3 What might lie ahead
- PDF p.593 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.16 — contents › 
- PDF p.617 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.618 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.611 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.616 (printed 592) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
Deep learning can help program synthesis do the same: although each specific program we’d like to generate might be a fundamentally discrete object that performs non-interpolative data manipulation, evidence so far indicates that the space of all useful programs may look a lot like a continuous manifold. That means that a deep learning model that has been trained on millions of successful program-generation episodes might start to develop solid intuition about the path through program space that the search process should take to go from a specification to the corresponding program—just like a software engineer might have immediate intuition about the overall architecture of the script they’re about to write and about the intermediate functions and classes they should use as stepping stones on the way to the goal. Remember that human reasoning is heavily guided by value-centric abstraction— that is, by pattern recognition and intuition. The same should be true of program synthesis. I expect the general approach of guiding program search via learned heuristics to see increasing research interest over the next 10 to 20 years.

### PDF p.615 (printed 591) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › Using deep learning to guide program search
Today, program synthesis faces a major obstacle: it’s tremendously inefficient. To caricature, typical program synthesis techniques work by trying every possible program in a search space until it finds one that matches the specification provided. As the complexity of a program specification increases, or as the vocabulary of primitives used to write programs expands, the program search process runs into what’s known as combinatorial explosion: the set of possible programs to consider grows very fast, in fact, much faster than merely exponentially fast. As a result, today, program synthesis can only be used to generate very short programs. You’re not going to be generating a new OS for your computer anytime soon.

To move forward, we’re going to need to make program synthesis efficient by bringing it closer to the way humans write software. When you open your editor to code up a script, you’re not thinking about every possible program you could potentially write. You only have in mind a handful of possible approaches: you can use your understanding of the problem and your past experience to drastically cut through the space of possible options to consider.

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 20.3 What might lie ahead
Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.

Future key developments may include

Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, co

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

### PDF p.615 (printed 591) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
broader than what can be represented with current deep learning models, and some of these programs could achieve superior generalization power. Importantly, such programs will not be differentiable end to end, although specific modules will remain differentiable, and thus will need to be generated via a combination of discrete program search and gradient descent.

We’ll move away from having, on one hand, hardcoded algorithmic intelligence (handcrafted software) and, on the other hand, learned geometric intelligence (deep learning). Instead, we’ll have a blend of formal algorithmic modules that provide reasoning and abstraction capabilities and geometric modules that provide informal intuition and pattern-recognition capabilities (figure 19.15). The entire system will be learned with little or no human involvement. This should dramatically expand the scope of problems that can be solved with machine learning—the space of programs that we can generate automatically, given appropriate training data. Systems like AlphaGo—or even RNNs—can be seen as a prehistoric ancestor of such hybrid algorithmic-geometric models.

Figure 19.15 A learned program relying on both geometric primitives (

### PDF p.618 (printed 594) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
that humans maintain about their surroundings and themselves, and they will be capable of stronger generalization due to their rich algorithmic nature.

In particular, models will blend algorithmic modules providing formal reasoning, search, and abstraction capabilities with geometric modules providing informal intuition and pattern-recognition capabilities. This will achieve a blend of value-centric and program-centric abstraction. AlphaGo or self-driving cars (systems that required a lot of manual software engineering and human-made design decisions) provide an early example of what such a blend of symbolic and geometric AI could look like.

Such models will be grown automatically rather than hardcoded by human engineers, using modular parts stored in a global library of reusable subroutines—a library evolved by learning high-performing models on thousands of previous tasks and datasets. As frequent problem-solving patterns are identified by the meta-learning system, they will be turned into reusable subroutines—much like functions and classes in software engineering—and added to the global library.

The process that searches over possible combinations of subroutines to grow new 

### PDF p.606 (printed 582) — 19 The future of AI › 19.3 How to build intelligence › 19.3.6 The test-time adaptation era
In 2024, everything changed. That year saw a major narrative shift—one that was partly catalyzed by ARC Prize. The prevailing “Scale is all you need” story that was a bedrock dogma of 2023 started giving way to “Actually, we need on-the-fly recombination.” The results of the competition, announced in December 2024, were illuminating: the leading solutions did not emerge from simply scaling existing deep learning architectures. They all used some form of test-time adaptation (TTA)—either test-time search or test-time training.

TTA refers to methods where the AI system performs active reasoning or learning during the test itself, using the specific problem information provided—the key component that was missing from the classic deep learning paradigm. There are several ways to implement test-time adaptation:

Test-time training—The model adjusts some of its parameters based on the examples given in the test task, using gradient descent.

Search methods—The system searches through many possible reasoning steps or potential solutions at test time to find the best one. This could be done in natural language (chain-of-thought synthesis) or in a space of symbolic, verifiable programs (pr

### PDF p.610 (printed 586) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › Program-centric analogy
Crucially, there’s more to cognition than the kind of immediate, approximate, intuitive categorization that value-centric analogy enables. There’s another type of abstraction-generation mechanism, slower, exact, deliberate: program-centric (or structure-centric) analogy.

In software engineering, you often write different functions or classes that seem to have a lot in common. When you notice these redundancies, you start asking, Could there be a more abstract function that performs the same job that could be reused twice? Could there be an abstract base class that both of your classes could inherit from? The definition of abstraction you’re using here corresponds to program-centric analogy. You’re not trying to compare your classes and functions by how similar they look, the way you’d compare two human faces, via an implicit distance function. Rather, you’re interested in whether there are parts of them that have exactly the same structure. You’re looking for what is called a subgraph isomorphism (see figure 19.13): programs can be represented as graphs of operators, and you’re trying to find subgraphs (program subsets) that are exactly shared across your different programs. This 
