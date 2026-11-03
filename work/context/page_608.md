# Generation bundle for page_608  (PDF page 608, printed page 584)

Page kind: content
Chapter: 19 The future of AI
Section: 19.4 The missing ingredients: Search and symbols
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_608.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_608.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p608_e03] (body) And so, in March 2025, the ARC Prize Foundation introduced ARC-AGI-2. It kept the exact same format as the first version but significantly improved the task content. The new iteration was designed to raise the bar, incorporating tasks that demand more complex reasoning chains and are inherently more resistant to exhaustive search methods. The goal was to create a benchmark where computational efficiency becomes a more critical factor for success, pushing systems toward more genuinely intelligent, efficient strategies rather than simply exploring billions of possibilities. While most ARC-AGI-1 tasks could be solved almost instantaneously by a human without requiring much cognitive effort, all tasks in ARC-AGI-2 require some amount of deliberate thinking (see figure 19.11)—for instance, the average time for task completion among human test takers in our experiments was 5 minutes.
[p608_e04] (figure_caption) Figure 19.11 Typical ARC-AGI-1 task (left) vs. typical ARC-AGI-2 task (right)
[p608_e05] (body) The initial AI testing results on ARC-AGI 2 were sobering: even o3 struggled significantly with this new set of challenges, its scores plummeting back into the low double digits when constrained to reasonable computational budgets. As for base LLMs? Their performance on ARC-AGI-2 was effectively back at 0%—fitting, as base LLMs don’t possess fluid intelligence. The challenge of building AI with truly efficient, human-like fluid intelligence is still far from solved. We’re going to need something beyond current TTA techniques.
[p608_e06] (heading_section) 19.4 The missing ingredients: Search and symbols
[p608_e07] (body) What would it take to fully solve ARC-AGI, in particular version 2? Hopefully, this challenge will get you thinking. That’s the entire point of ARC-AGI: to give you a goal of a different kind, that will nudge you in a new direction—hopefully, a productive direction. Now, let’s take a quick look at the key ingredients you’re going to need if you want to answer the call.

## Figures
- Figure 19.11: Typical ARC-AGI-1 task (left) vs. typical ARC-AGI-2 task (right)  | labels inside the figure: (none - image)

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

## Previous page (PDF 607) text, for continuity
19.3.7 ARC-AGI 2

Does that mean AGI is solved? Was o3 as intelligent as a human?

Not quite. First, while o3’s performance was a landmark achievement, it came at a tremendous cost—tens of thousands of dollars of compute per ARC-AGI puzzle. Intelligence isn’t just about capability; it’s fundamentally about efficiency. Brute-forcing the solution space given enormous compute is a shortcut that makes all kinds of tasks possible without requiring intelligence. In principle, you could even solve ARC-AGI by simply walking down the tree of every possible solution program and testing each one until you find one that works on the demonstration pairs. The o3 results, impressive as they were, felt more like cracking a code with a supercomputer than a display of nimble, human-like fluid reasoning. The entire point of intelligence is to achieve results with the least amount of resources possible. Second, we found that o3 was still stumped by many tasks that humans found very easy (like the one in figure 19.10). This strongly suggests that o3 wasn’t quite human-level yet. Here’s the thing—the 2019 version of ARC-AGI was intended to be easy. It was essentially a binary test of fluid intelligence: either you have no fluid intelligence, like all base LLMs, in which case you score near zero, or you do display some genuine fluid intelligence, in which case you immediately score extremely high, like any human—or o3. There wasn’t much room in between. It was clear that the benchmark needed to evolve alongside the AI capabilities it was designed to measure. There was a need for a new ARC-AGI version that was less brute-forcible and that could better differentiate between systems possessing varying levels of fluid reasoning ability, up to human-level fluid intelligence. Good news: we had been …

## Next page (PDF 609) text, for continuity
I’ve said that intelligence consists of two components: abstraction acquisition and abstraction recombination. They are tightly coupled—what kind of abstractions you manipulate determines how, and how well, you can recombine them. Deep learning models only manipulate abstractions stored via parametric curves, fitted via gradient descent. Could there be a better way?

19.4.1 The two poles of abstraction

Abstraction acquisition starts with comparing things to one another. Crucially, there are two distinct ways to compare things, from which arise two different kinds of abstraction and two modes of thinking, each better suited to a different kind of problem. Together, these two poles of abstraction form the basis for all of our thoughts.

The first way to relate things to each other is similarity comparison, which gives rise to value-centric analogies. The second way is exact structural match, which gives rise to program-centric analogies (or structure-centric analogies). In both cases, you start from instances of a thing, and you merge together related instances to produce an abstraction that captures the common elements of the underlying instances. What varies is how you tell that two instances are related and how you merge instances into abstractions. Let’s take a close look at each type.

Value-centric analogy

Let’s say you come across a number of different beetles in your backyard, belonging to multiple species. You’ll notice similarities between them. Some will be more similar to one another, and some will be less similar: the notion of similarity is implicitly a smooth, continuous distance function that defines a latent manifold where your instances live. Once you’ve seen enough beetles, you can start clustering more similar instances together and merging them into …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.605 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.606 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.630 — 20 Conclusions › 20.3 What might lie ahead
- PDF p.604 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.602 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.37 — 1 What is deep learning? › 1.10 Beware of the short-term hype
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.603 — 19 The future of AI › 19.3 How to build intelligence

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.606 (printed 582) — 19 The future of AI › 19.3 How to build intelligence › 
The ARC-AGI benchmark has proven remarkably resistant to the prevailing deep learning scaling paradigm. Most other benchmarks have saturated quickly in the age of LLMs. That’s because they can be hacked via memorization, whereas ARC-AGI is designed to be resistant to it. From 2019, when ARC-AGI was first released, to 2025, base LLMs underwent a roughly 50,000× scale-up—from, say, GPT-2 (2019) to GPT-4.5 (2025), but their performance on the 2019 version of ARC-AGI only went from 0% to around 10%. Given that you, reader, would easily score above 95%, that isn’t very good. If you scale up your system by 50,000× and you’re still not making meaningful progress, that’s like a big warning sign telling you that you need to try new ideas. Simply making models bigger or training them on more data has not unlocked the kind of fluid intelligence that ARC-AGI requires. ARC-AGI was clearly showing that on-the-fly recombination capabilities are necessary to tackle reasoning.

### PDF p.605 (printed 581) — 19 The future of AI › 19.3 How to build intelligence › 
of intelligence. ARC-AGI is meant to be approachable by both machines and humans, and it looks very similar to human IQ tests, such as Raven’s progressive matrices. At test time, you’ll see a series of “tasks.” Each task is explained via three or four “examples” that take the form of an input grid and a corresponding output grid (see figure 19.9). You’ll then be given a brand-new input grid, and you’ll have three tries to produce the correct output grid, before moving on to the next task.

Figure 19.9 An ARC-AGI task: the nature of the task is demonstrated by a couple of input-output pair examples. Provided with a new input, you must construct the corresponding output.

Compared to IQ tests, two things are unique about ARC-AGI. First, ARC seeks to measure generalization power by only testing you on tasks you’ve never seen before. That means that ARC-AGI is a game you can’t practice for, at least in theory: the tasks you will get tested on will have their own unique logic that you will have to understand on the fly. You can’t just memorize specific strategies from past tasks.

In addition, ARC-AGI tries to control for the prior knowledge that you bring to the test. You never approac

### PDF p.605 (printed 581) — 19 The future of AI › 19.3 How to build intelligence › 19.3.5 ARC Prize
In 2024, to accelerate progress toward AI systems capable of the kind of fluid abstraction and reasoning measured by ARC-AGI, I partnered with Mike Knoop to establish the nonprofit ARC Prize Foundation. The foundation runs a yearly competition, with a substantial prize pool (over $1 million in its 2024 iteration) to incentivize researchers to develop AI that can solve ARC-AGI and thus display genuine fluid intelligence.

### PDF p.606 (printed 582) — 19 The future of AI › 19.3 How to build intelligence › 19.3.6 The test-time adaptation era
In 2024, everything changed. That year saw a major narrative shift—one that was partly catalyzed by ARC Prize. The prevailing “Scale is all you need” story that was a bedrock dogma of 2023 started giving way to “Actually, we need on-the-fly recombination.” The results of the competition, announced in December 2024, were illuminating: the leading solutions did not emerge from simply scaling existing deep learning architectures. They all used some form of test-time adaptation (TTA)—either test-time search or test-time training.

TTA refers to methods where the AI system performs active reasoning or learning during the test itself, using the specific problem information provided—the key component that was missing from the classic deep learning paradigm. There are several ways to implement test-time adaptation:

Test-time training—The model adjusts some of its parameters based on the examples given in the test task, using gradient descent.

Search methods—The system searches through many possible reasoning steps or potential solutions at test time to find the best one. This could be done in natural language (chain-of-thought synthesis) or in a space of symbolic, verifiable programs (pr

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 
benchmarks that amount to memorization tests, it fails to address the fundamental limitations of deep learning, which stem from the core paradigm of fitting static, interpolative curves to data. Five years of exponential scaling of base LLMs haven’t overcome these constraints because the underlying approach remains unchanged.

By 2024, this realization spurred a transition toward test-time adaptation (TTA), where models perform search or fine-tuning during the inference phase to adapt to novel problems. While TTA methods have yielded major breakthroughs, such as OpenAI’s o3 surpassing human baseline on ARC-AGI-1 in late 2024, this performance has come at an extreme computational cost. Efficient, human-like adaptation is still a completely open problem, and the slightly harder ARC-AGI-2 benchmark remains completely unsolved as of today. We still need further conceptual advances beyond mere scaling or brute-force search.

### PDF p.633 (printed 609) — index ›  › A
annotations 214 annotations/trimaps/ folder 311 abstraction acquisition 578 anomaly detection 352 activation append() method 192 channel 302 ARC-AGI (Abstraction & Reasoning Corpus for functions 109

Artificial General Intelligence) 581, 583 adam optimizer 247, 379, 490 architecture patterns, batch normalization AdamW optimizer 529 276–278 adapt() method 399 ARC Prize 581 add() method 192 arXiv 608 adversarial examples 567 ASICs (application-specific integrated circuits) 555 AGI (artificial general intelligence) 13, 571, 594 attention 464 AI (artificial intelligence) 2, 564, 596 dot-product attention 439 AI winters 14 properties of 462 building intelligence 576–584 Transformer architecture 437 future developments 606 attention_mask 446, 447 intelligent agents vs. automatons 571 attention mechanisms 462 local generalization vs. extreme dot-product attention 439 generalization 573 augmented_train_dataset 263 promise of 14 augmenting text data 413 purpose of intelligence 575 autoencoders 510 scaling models 571 automated hyperparameter tuning software 183 search and symbols 584–594 spectrum of generalization 576 automatons 571 AI summer 597 autoregressive model 426 algorithmic modules 

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 20.3 What might lie ahead
Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.

Future key developments may include

Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, co

### PDF p.603 (printed 579) — 19 The future of AI › 19.3 How to build intelligence › 
So far, the driving success metric of the field of AI has been to solve specific tasks, from chess

to Go, from MNIST classification to ImageNet, from high school math tests to the bar exam. Consequently, the history of the field has been defined by a series of “successes” where we figured out how to solve these tasks without featuring any intelligence.

If that sounds like a surprising statement, keep in mind that human-like intelligence isn’t characterized by skill at any particular task—rather, it is the ability to adapt to novelty to efficiently acquire new skills and master never-before-seen tasks. By fixing the task, you make it possible to provide an arbitrarily precise description of what needs to be done—either via hardcoding human-provided knowledge or by supplying humongous amounts of data. You make it possible for engineers to “buy” more skill for their AI by just adding data or adding hardcoded knowledge, without increasing the generalization power of the AI (see figure 19.8). If you have near-infinite training data, even a very crude algorithm like nearest-neighbor search can play video games with superhuman skill. Likewise, if you have a near-infinite amount of human
