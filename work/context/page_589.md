# Generation bundle for page_589  (PDF page 589, printed page 565)

Page kind: content
Chapter: 19 The future of AI
Section: 19.1 The limitations of deep learning
Subsection: 19.1.1 Deep learning models struggle to adapt to novelty
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_589.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_589.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p589_e03] (heading_subsection) 19.1.1 Deep learning models struggle to adapt to novelty
[p589_e04] (body) Deep learning models are big, parametric curves fitted to large datasets. That’s the source of their power—they’re easy to train, and they scale really well, both in terms of model size and dataset size. But that’s also a source of significant weaknesses. Curve fitting has inherent limitations.
[p589_e05] (body) First and foremost, a parametric curve is only capable of information storage—it’s a kind of database. Recall our discussion of Transformers as an “interpolative database” from chapter 15? Second, crucially, this database is static. The model’s parameters are determined during a distinct “training time” phase. Afterward, these parameters are frozen, and this fixed version is used during “inference time” for making predictions on new data. The only thing you can do with a static database is information retrieval. And that’s exactly what deep learning models excel at: recognizing or generating patterns highly similar to those encountered during training. The flip side is that they are inherently poor at adaptation. The database is backward-looking—it fits past data but can’t handle a changing future. At inference time, you’d better hope that the situations the model faces are part of the training data distribution, because otherwise, the model will break down. A model trained on ImageNet will classify a leopard-print sofa as an actual leopard, for instance—sofas were not part of its training data. This also applies to the largest of generative models. In recent years, the rise of large language models (LLMs) and their application to programming assistance and reasoning-like problems has provided extensive empirical proof of this. Despite frequent claims that LLMs can perform in-context learning to pick up new skills from just a few examples, there is overwhelming evidence that what they’re actually doing is fetching vector functions they’ve memorized during training and reapplying them to the task at hand. By learning to do next-token prediction across a web-sized text dataset, an LLM has collected millions of potentially useful mini text-processing programs, and it can easily be prompted into reusing them on a new problem. But show it something that has no direct equivalent in its training data, and it’s helpless. Take a look at the puzzle in figure 19.1. Did you figure out the solution? Good. It’s not very hard, is it? But today, no state-of-the-art LLM or vision-language model can do this because this particular problem doesn’t directly map to anything they’ve seen at training time—even after having been trained on the entire internet and then some. An LLM’s ability to solve a given problem has nothing to do with problem complexity, and everything to do with familiarity—they will break their teeth on any sufficiently novel problem, no matter how simple. This failure mode even applies to tiny variations of a pattern that an LLM encountered many times in its training data. For instance, for a few months after the release of ChatGPT, if you asked it, “What’s heavier, 10 kilos of steel or one kilo of feathers?,” it would answer that they weigh the same. That’s because the question “What’s heavier, one kilo of steel or one kilo of feathers?” is found many times on the internet—as a trick question. The right answer, of course, is that they both weigh the same, so the GPT

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

## Previous page (PDF 588) text, for continuity
The future of AI

This chapter covers

The limitations of deep learning

The nature of intelligence

What’s missing from current approaches

What the future might look like

To use a tool appropriately, you should not only understand what it can do but also be aware of what it can’t do. I’m going to present an overview of some key limitations of deep learning. Then, I’ll offer some speculative thoughts about the future evolution of AI and what it would take to get to human-level general intelligence. This should be especially interesting to you if you’d like to get into fundamental research.

19.1 The limitations of deep learning

There are infinitely many things you can do with deep learning. But deep learning can’t do everything. To use a tool well, you should be aware of its limitations, not just its strengths. So where does deep learning fall short?

## Next page (PDF 590) text, for continuity
Figure 19.1 An easy yet novel puzzle

model would just repeat the answer it had memorized without paying any attention to the actual numbers in the query, or what the query really meant. Similarly, LLMs struggle to adapt to variations of the Monty Hall problem (see figure 19.2) and will tend to always output the canonical answer to the puzzle, which they’ve seen many times during training, regardless of whether it makes sense in context.

Figure 19.2 A variation of the Monty Hall problem

To note, these specific prompts were patched later on by special-casing them. Today, there are over 25,000 people who are employed full time to provide training data for LLMs by reviewing failure cases and suggesting better answers. LLM maintenance is a constant game of whack-a-mole where failing prompts are patched one at a time,

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.595 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.593 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.629 — 20 Conclusions › 20.2 Limitations of deep learning
- PDF p.592 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.621 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.35 — 1 What is deep learning? › 1.9 What deep learning has achieved so far
- PDF p.525 — 16 Text generation › 16.4 Going further with LLMs

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.629 (printed 605) — 20 Conclusions › 20.2 Limitations of deep learning › 20.2 Limitations of deep learning
Building deep learning models is like playing with LEGO bricks: layers can be plugged together to map essentially anything to anything, given that you have appropriate training data available and that the mapping is achievable via a continuous geometric transformation of reasonable complexity.

Here’s the catch, though—this mapping is often not learnable in a way that will generalize. Deep learning models operate like vast, interpolative databases of patterns. Their pattern-matching strength is also their core weakness:

They fundamentally struggle to adapt to novelty. Because their parameters are fixed after training, they can only retrieve or replicate patterns similar to their training data. Faced with inputs significantly outside this familiar distribution—no matter how simple the underlying task—their performance degrades drastically, as they lack mechanisms for fluid generalization beyond their memorized experience. This explains why even large models fail on novel tasks or simple variations of familiar problems—like ARC-AGI tasks.

They’re very sensitive to phrasing and other distractors. Deep learning models exhibit high sensitivity to superficial variations in input presen

### PDF p.593 (printed 569) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.3 Deep learning models struggle to learn generalizable programs
The problem with deep learning models isn’t just that they’re limited to blindly reapplying patterns they’ve memorized at training time or that they’re highly sensitive to how their input is presented. Even if you just need to query and apply a well-known program, and you know exactly how to address this program in latent space, you still face a major issue: the programs memorized by deep learning models often don’t generalize well. They will work for some input values and fail for some other input values. This is especially true for programs that encode any kind of discrete logic.

### PDF p.170 (printed 146) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Why deep learning works
Remember the crumpled paper ball metaphor from chapter 2? A sheet of paper represents a 2D manifold within 3D space (figure 5.9). A deep learning model is a tool for uncrumpling paper balls—that is, for disentangling latent manifolds.

Figure 5.9 Uncrumpling a complicated manifold of data

A deep learning model is basically a very high-dimensional curve. The curve is smooth and continuous (with additional constraints on its structure, originating from model architecture priors) because it needs to be differentiable. And that curve is fitted to data points via gradient descent—smoothly and incrementally. By construction, deep learning is about taking a big, complex curve—a manifold—and incrementally adjusting its parameters until it fits some training data points.

The curve involves enough parameters that it could fit anything. Indeed, if you let your model train for long enough, it will effectively end up purely memorizing its training data and won’t generalize at all. However, the data you’re fitting to isn’t made of isolated points sparsely distributed across the underlying space. Your data forms a highly structured, low-dimensional manifold within the input space—that’s the man

### PDF p.595 (printed 571) — 19 The future of AI › 19.2 Scale isn’t all you need › 
been released, and it was essentially a scaled-up version of GPT-3: more parameters, more training data. Its significantly improved performance seemed to suggest that you could just keep going—that there could be a GPT-5 that would simply be more of the same and from which artificial general intelligence (AGI) would spontaneously emerge.

Proponents of this view would point to “scaling laws” as evidence. Scaling laws are an empirical relationship observed between the size of a deep learning model (as well as the size of its training dataset) and its performance on specific tasks. They suggest that increasing the size of a model reliably leads to better performance in a predictable manner. But the key thing that scaling law enthusiasts are missing is that the benchmarks they’re using to measure “performance” are effectively memorization tests, the kind we like to give university students. LLMs perform well on these tests by memorizing the answers, and naturally, cramming more questions and more answers into the models improves their performance accordingly. The reality is that scaling up our models hasn’t led to any progress on the issues I’ve listed so far in these pages—inability 

### PDF p.593 (printed 569) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.3 Deep learning models struggle to learn generalizable programs
Consider the problem of adding two numbers, represented as character sequences— like “4 3 5 7 + 8 9 3 6.” Try training a Transformer on hundreds of thousands of such digit pairs: you will reach a very high accuracy. Very high, but not 100%—you will keep regularly seeing incorrect answers, because the Transformer doesn’t manage to encode the exact addition algorithm (you know, the one you learned in primary school). It is instead guessing the output by interpolating between the data points it has seen at training time. This applies to state-of-the-art LLMs, too—at least those that weren’t explicitly hardcoded to execute snippets like “4357 + 8936” in Python to provide the right answer. They’ve seen enough examples of digit addition that they can add numbers, but they only have about 70% accuracy—quite underwhelming. Further, their accuracy is strongly dependent on which digits are being added, with more common digits leading to higher accuracy. The reason why a deep learning model does not end up learning an exact addition algorithm even after seeing millions of examples is that it is just a static chain of simple, continuous geometric transformations mapping one vector space into a

### PDF p.35 (printed 11) — 1 What is deep learning? › 1.9 What deep learning has achieved so far › 
law. In addition, deep learning models are trained by iterating over small batches of data, allowing them to be trained on datasets of arbitrary size. (The only bottleneck is the amount of parallel computational power available, which, thanks to Moore’s law, is a fast-moving barrier.)

Versatility and reusability—Unlike many prior machine learning approaches, deep learning models can be trained on additional data without restarting from scratch, making them viable for continuous online learning—an important property for very large production models. Furthermore, trained deep learning models are repurposable and thus reusable: this is the big idea behind “foundation models”—large models trained on humongous amounts of data, which can be used across many new tasks with little retraining, or even none at all.

### PDF p.629 (printed 605) — 20 Conclusions › 20.2 Limitations of deep learning › 20.2 Limitations of deep learning
You should always resist the temptation to anthropomorphize deep learning models. Their performance is built on pointwise statistical patterns rather than human-like experiential grounding, making it brittle when encountering deviations from training data.

The narrative that simply scaling up model size and training data would lead to general intelligence has proven insufficient. While scaling enhances performance on

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 
benchmarks that amount to memorization tests, it fails to address the fundamental limitations of deep learning, which stem from the core paradigm of fitting static, interpolative curves to data. Five years of exponential scaling of base LLMs haven’t overcome these constraints because the underlying approach remains unchanged.

By 2024, this realization spurred a transition toward test-time adaptation (TTA), where models perform search or fine-tuning during the inference phase to adapt to novel problems. While TTA methods have yielded major breakthroughs, such as OpenAI’s o3 surpassing human baseline on ARC-AGI-1 in late 2024, this performance has come at an extreme computational cost. Efficient, human-like adaptation is still a completely open problem, and the slightly harder ARC-AGI-2 benchmark remains completely unsolved as of today. We still need further conceptual advances beyond mere scaling or brute-force search.
