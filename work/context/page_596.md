# Generation bundle for page_596  (PDF page 596, printed page 572)

Page kind: content
Chapter: 19 The future of AI
Section: 19.2 Scale isn’t all you need
Subsection: 19.2.1 Automatons vs. intelligent agents
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_596.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_596.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p596_e03] (body) Let’s zoom out a little bit and ask, What’s the purpose of intelligence? Why did it arise in the first place? We can only speculate, but we can make fairly informed speculations. We can start by looking at brains—the organ that produces intelligence. Brains are an evolutionary adaptation—a mechanism developed incrementally over hundreds of millions of years, via random trial and error guided by natural selection, that dramatically expanded the ability of organisms to adapt to their environment. Brains originally appeared more than half a billion years ago as a way to store and execute behavioral programs. Behavioral programs are just sets of instructions that make an organism reactive to its environment: “If this happens, then do that.” They link the organism’s sensory inputs to its motor controls. In the beginning, brains would have served to hardcode behavioral programs (as neural connectivity patterns), which would allow an organism to react appropriately to its sensory input. This is the way insect brains still work—flies, ants, C. elegans (see figure 19.5), etc. Because the original “source code” of these programs was DNA, which would get decoded as neural connectivity patterns, evolution was suddenly able to search over behavior space in a largely unbounded way—a major evolutionary shift.
[p596_e04] (figure_caption) Figure 19.5 The brain network of the C. elegans worm: a behavioral automaton “programmed” by natural evolution. Figure created by Emma Towlson (from “Network control principles predict neuron function in the Caenorhabditis elegans connectome,” Yan et al., Nature, Oct. 2017).

## Figures
- Figure 19.5: The brain network of the C. elegans worm: a behavioral automaton “programmed” by natural evolution. Figure created by Emma Towlson (from “Network control principles predict neuron function in the Caenorhabditis elegans connectome,” Yan et al., Nature, Oct. 2017).  | labels inside the figure: (none - image)

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

## Previous page (PDF 595) text, for continuity
been released, and it was essentially a scaled-up version of GPT-3: more parameters, more training data. Its significantly improved performance seemed to suggest that you could just keep going—that there could be a GPT-5 that would simply be more of the same and from which artificial general intelligence (AGI) would spontaneously emerge.

Proponents of this view would point to “scaling laws” as evidence. Scaling laws are an empirical relationship observed between the size of a deep learning model (as well as the size of its training dataset) and its performance on specific tasks. They suggest that increasing the size of a model reliably leads to better performance in a predictable manner. But the key thing that scaling law enthusiasts are missing is that the benchmarks they’re using to measure “performance” are effectively memorization tests, the kind we like to give university students. LLMs perform well on these tests by memorizing the answers, and naturally, cramming more questions and more answers into the models improves their performance accordingly. The reality is that scaling up our models hasn’t led to any progress on the issues I’ve listed so far in these pages—inability to adapt to novelty, oversensitivity to phrasing, and the inability to infer generalizable programs for reasoning problems—because these issues are inherent to curve fitting, the paradigm of deep learning. I started pointing out these problems in 2017, and we’re still struggling with them today—with models that are now four or five orders of magnitude larger and more knowledgeable. We have not made any progress on these problems because the models we’re using are still the same. They’ve been the same for over seven years—they’re still parametric curves fitted to a dataset via gradient descent, …

## Next page (PDF 597) text, for continuity
Evolution was the programmer, and brains were computers carefully executing the code evolution gave them. Because neural connectivity is a very general computing substrate, the sensorimotor space of all brain-enabled species could suddenly start undergoing a dramatic expansion. Eyes, ears, mandibles, 4 legs, 24 legs—as long as you have a brain, evolution will kindly figure out for you behavioral programs that make good use of these. Brains can handle any modality, or combination of modalities, you throw at them.

Now, mind you, these early brains weren’t exactly intelligent per se. They were very much automatons: they would merely execute behavioral programs hardcoded in the organism’s DNA. They could only be described as intelligent in the same sense that a thermostat is “intelligent.” Or a list-sorting program. Or a trained deep neural network (of the artificial kind). This is an important distinction, so let’s look at it carefully: What’s the difference between automatons and actual intelligent agents?

19.2.2 Local generalization vs. extreme generalization

The field of AI has long suffered from conflating the notions of intelligence and automation. An automation system (or automaton) is static, crafted to accomplish specific things in a specific context—”If this, then that”—while an intelligent agent can adapt on the fly to novel, unexpected situations. When an automaton is exposed to something that doesn’t match what it was “programmed” to do (whether we’re talking about human-written programs, evolution-generated programs, or the implicit programming process of fitting a model on a training dataset), it will fail.

Meanwhile, intelligent agents, like us humans, will use their fluid intelligence to find a way forward. How do you tell the difference between a stude …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.599 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.600 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.614 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.593 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.617 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.615 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.613 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.599 (printed 575) — 19 The future of AI › 19.2 Scale isn’t all you need › 19.2.3 The purpose of intelligence
This distinction between highly adaptable intelligent agents and rigid automatons leads us back to brain evolution. Why did brains—originally a mere medium for natural evolution to develop behavioral automatons—eventually turn intelligent? Like every significant evolutionary milestone, it happened because natural selection constraints encouraged it to happen.

Brains are responsible for behavior generation. If the set of situations an organism had to face was mostly static and known in advance, behavior generation would be an easy problem: evolution would just figure out the correct behaviors via random trial and error and hardcode them into the organism’s DNA. This first stage of brain evolution— brains as automatons—would already be optimal. However, crucially, as organism complexity—and alongside it, environmental complexity—kept increasing, the situations animals had to deal with became much more dynamic and more unpredictable. A day in your life, if you look closely, is unlike any day you’ve ever experienced and unlike any day ever experienced by any of your evolutionary ancestors. You need to be able to face unknown and surprising situations constantly. There is no way for ev

### PDF p.599 (printed 575) — 19 The future of AI › 19.2 Scale isn’t all you need › 19.2.4 Climbing the spectrum of generalization
As a crude caricature, you could summarize the evolutionary history of biological intelligence as a slow climb up the spectrum of generalization. It started with automaton-like brains that could only perform local generalization. Over time, evolution started producing organisms capable of increasingly broader generalization that could thrive in ever-more complex and variable environments. Eventually, in the past few million years—an instant in evolutionary terms—certain hominin species started trending toward an implementation of biological intelligence capable of extreme

### PDF p.15 (printed xv) — contents ›  › 
19.2

Scale isn’t all you need 570

Automatons vs. intelligent agents 571 ■ Local generalization vs. extreme generalization 573 ■ The purpose of intelligence 575 Climbing the spectrum of generalization 575

### PDF p.604 (printed 580) — 19 The future of AI › 19.3 How to build intelligence › 19.3.4 A new target: On-the-fly adaptation
To make AI actually intelligent and give it the ability to deal with the incredible variability and ever-changing nature of the real world, first, we need to move away from seeking to achieve task-specific skill and, instead, start targeting generalization power itself. We need new metrics of progress that will help us develop increasingly intelligent systems: metrics that will point in the right direction and that will give us an actionable feedback signal. As long as we set our goal to be “create a model that solves task X,” the shortcut rule will apply, and we’ll end up with a model that does X, period.

In my view, intelligence can be precisely quantified as an efficiency ratio: the conversion ratio between the amount of relevant information you have available about the world (which could be either past experience or innate prior knowledge) and your future operating area, the set of novel situations where you will be able to produce appropriate behavior (you can view this as your skill set). A more intelligent agent will be able to handle a broader set of future tasks and situations using a smaller amount of past experience. To measure such a ratio, you just need to fix the inf

### PDF p.613 (printed 589) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.4 An alternative approach to AI: Program synthesis
Until 2024, AI systems capable of genuine discrete reasoning were all hardcoded by human programmers—for instance, software that relies on search algorithms, graph manipulation, and formal logic. In the test-time adaptation (TTA) era, this is finally starting to change. A branch of TTA that is especially promising is program synthesis—a field that is still very niche today, but that I expect to take off in a big way over the next few decades.

Program synthesis consists of automatically generating simple programs by using a search algorithm (possibly genetic search, as in genetic programming) to explore a large space of possible programs (see figure 19.14). The search stops when a program is found that matches the required specifications, often provided as a set of input-output pairs. This is highly reminiscent of machine learning: given training data provided as input-output pairs, we find a program that matches inputs to outputs and can generalize to new inputs. The difference is that instead of learning parameter values in a hardcoded program (a neural network), we generate source code via a discrete search process (see table 19.2).

### PDF p.600 (printed 576) — 19 The future of AI › 19.3 How to build intelligence › 
The progress of AI over the past 70 years bears striking similarities to this evolution. Early AI systems were pure automatons, like the ELIZA chat program from the 1960s, or SHRDLU,2 a 1970 AI capable of manipulating simple objects from natural language commands. In the 1990s and 2000s, we saw the rise of machine learning systems capable of local generalization that could deal with some level of uncertainty and novelty. In the 2010s, deep learning further expanded the local generalization power of these systems by enabling engineers to use much larger datasets and much more expressive models. Today, we may be on the cusp of the next evolutionary step. We are moving toward systems that achieve broad generalization, which I define as the ability to deal with unknown unknowns within a single broad domain of tasks (including situations the system was not trained to handle and that its creators could not have anticipated). Examples are a self-driving car capable of safely dealing with any situation you throw at it or a domestic robot that could pass the “Woz test of intelligence”—entering a random kitchen and making a cup of coffee.3 By combining deep learning and painstakingly handcra

### PDF p.615 (printed 591) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
broader than what can be represented with current deep learning models, and some of these programs could achieve superior generalization power. Importantly, such programs will not be differentiable end to end, although specific modules will remain differentiable, and thus will need to be generated via a combination of discrete program search and gradient descent.

We’ll move away from having, on one hand, hardcoded algorithmic intelligence (handcrafted software) and, on the other hand, learned geometric intelligence (deep learning). Instead, we’ll have a blend of formal algorithmic modules that provide reasoning and abstraction capabilities and geometric modules that provide informal intuition and pattern-recognition capabilities (figure 19.15). The entire system will be learned with little or no human involvement. This should dramatically expand the scope of problems that can be solved with machine learning—the space of programs that we can generate automatically, given appropriate training data. Systems like AlphaGo—or even RNNs—can be seen as a prehistoric ancestor of such hybrid algorithmic-geometric models.

Figure 19.15 A learned program relying on both geometric primitives (

### PDF p.603 (printed 579) — 19 The future of AI › 19.3 How to build intelligence › 
So far, the driving success metric of the field of AI has been to solve specific tasks, from chess

to Go, from MNIST classification to ImageNet, from high school math tests to the bar exam. Consequently, the history of the field has been defined by a series of “successes” where we figured out how to solve these tasks without featuring any intelligence.

If that sounds like a surprising statement, keep in mind that human-like intelligence isn’t characterized by skill at any particular task—rather, it is the ability to adapt to novelty to efficiently acquire new skills and master never-before-seen tasks. By fixing the task, you make it possible to provide an arbitrarily precise description of what needs to be done—either via hardcoding human-provided knowledge or by supplying humongous amounts of data. You make it possible for engineers to “buy” more skill for their AI by just adding data or adding hardcoded knowledge, without increasing the generalization power of the AI (see figure 19.8). If you have near-infinite training data, even a very crude algorithm like nearest-neighbor search can play video games with superhuman skill. Likewise, if you have a near-infinite amount of human
