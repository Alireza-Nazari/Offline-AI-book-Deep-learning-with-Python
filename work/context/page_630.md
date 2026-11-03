# Generation bundle for page_630  (PDF page 630, printed page 606)

Page kind: content
Chapter: 20 Conclusions
Section: 20.3 What might lie ahead
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_630.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_630.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p630_e03] (body) benchmarks that amount to memorization tests, it fails to address the fundamental limitations of deep learning, which stem from the core paradigm of fitting static, interpolative curves to data. Five years of exponential scaling of base LLMs haven’t overcome these constraints because the underlying approach remains unchanged.
[p630_e04] (body) By 2024, this realization spurred a transition toward test-time adaptation (TTA), where models perform search or fine-tuning during the inference phase to adapt to novel problems. While TTA methods have yielded major breakthroughs, such as OpenAI’s o3 surpassing human baseline on ARC-AGI-1 in late 2024, this performance has come at an extreme computational cost. Efficient, human-like adaptation is still a completely open problem, and the slightly harder ARC-AGI-2 benchmark remains completely unsolved as of today. We still need further conceptual advances beyond mere scaling or brute-force search.
[p630_e05] (heading_section) 20.3 What might lie ahead
[p630_e06] (body) Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.
[p630_e07] (body) Future key developments may include
[p630_e08] (bullet) Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.
[p630_e09] (bullet) Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, combinatorial spaces of programs efficiently, much like human developers use experience and intuition to narrow down their choices.
[p630_e10] (bullet) Modular recombination and lifelong learning—We’ll move away from monolithic, end-to-end models trained from scratch. Instead, future AI systems will use massive libraries of reusable, modular components that can be repurposed across many problems, acquired from experience. These libraries will feature both “geometric” (deep learning based) and “algorithmic” modules. When faced with a new problem, such AI systems will fetch relevant modules and dynamically recombine them into a new model adapted to the situation at hand. Whenever the system ends up developing a reusable component as a by-product of

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

## Previous page (PDF 629) text, for continuity
And this is a stacked RNN layer for binary classification of vector sequences:

20.2 Limitations of deep learning

Building deep learning models is like playing with LEGO bricks: layers can be plugged together to map essentially anything to anything, given that you have appropriate training data available and that the mapping is achievable via a continuous geometric transformation of reasonable complexity.

Here’s the catch, though—this mapping is often not learnable in a way that will generalize. Deep learning models operate like vast, interpolative databases of patterns. Their pattern-matching strength is also their core weakness:

They fundamentally struggle to adapt to novelty. Because their parameters are fixed after training, they can only retrieve or replicate patterns similar to their training data. Faced with inputs significantly outside this familiar distribution—no matter how simple the underlying task—their performance degrades drastically, as they lack mechanisms for fluid generalization beyond their memorized experience. This explains why even large models fail on novel tasks or simple variations of familiar problems—like ARC-AGI tasks.

They’re very sensitive to phrasing and other distractors. Deep learning models exhibit high sensitivity to superficial variations in input presentation, such as minor phrasing changes (consider prompt sensitivity in LLMs) or imperceptible perturbations (consider adversarial examples in vision), indicating a lack of robust, human-like understanding.

They often can’t learn generalizable algorithms. The continuous, geometric nature of deep learning models makes them fundamentally ill-suited for learning exact, discrete, step-by-step algorithms, such as those that are the bread and butter of classical computer science. Models …

## Next page (PDF 631) text, for continuity
this problem-solving loop, the new component would get added to the library, becoming available for every future task the system might encounter. Ultimately, developing AI that mirrors human-like fluid intelligence will require blending continuous pattern recognition together with discrete, symbolic programs and fully embracing the paradigm of on-the-fly adaptation.

20.4 Staying up to date in a fast-moving field

As final parting words, I want to give you some pointers about how to keep learning and updating your knowledge and skills after you’ve turned the last page of this book. The field of modern deep learning, as we know it today, is only a few years old, despite a long, slow prehistory stretching back decades. With an exponential increase in financial resources and research headcount since 2013, the field as a whole is now moving at a frenetic pace. What you’ve learned in this book won’t stay relevant forever, and it isn’t all you’ll need for the rest of your career.

Fortunately, there are plenty of free online resources that you can use to stay up to date and expand your horizons. Here are a few.

20.4.1 Practice on real-world problems using Kaggle

An effective way to acquire real-world experience is to try your hand at machine learning competitions on Kaggle (https://kaggle.com). The only real way to learn is through practice and actual coding—that’s the philosophy of this book, and Kaggle competitions are the natural continuation of this. On Kaggle, you’ll find an array of constantly renewed data science competitions, many of which involve deep learning, prepared by companies interested in obtaining novel solutions to some of their most challenging machine learning problems. Fairly large monetary prizes are offered to top entrants.

By participating in a few …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.615 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.606 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.16 — contents › 
- PDF p.614 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.613 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.618 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.595 — 19 The future of AI › 19.2 Scale isn’t all you need

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.606 (printed 582) — 19 The future of AI › 19.3 How to build intelligence › 19.3.6 The test-time adaptation era
In 2024, everything changed. That year saw a major narrative shift—one that was partly catalyzed by ARC Prize. The prevailing “Scale is all you need” story that was a bedrock dogma of 2023 started giving way to “Actually, we need on-the-fly recombination.” The results of the competition, announced in December 2024, were illuminating: the leading solutions did not emerge from simply scaling existing deep learning architectures. They all used some form of test-time adaptation (TTA)—either test-time search or test-time training.

TTA refers to methods where the AI system performs active reasoning or learning during the test itself, using the specific problem information provided—the key component that was missing from the classic deep learning paradigm. There are several ways to implement test-time adaptation:

Test-time training—The model adjusts some of its parameters based on the examples given in the test task, using gradient descent.

Search methods—The system searches through many possible reasoning steps or potential solutions at test time to find the best one. This could be done in natural language (chain-of-thought synthesis) or in a space of symbolic, verifiable programs (pr

### PDF p.616 (printed 592) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
Deep learning can help program synthesis do the same: although each specific program we’d like to generate might be a fundamentally discrete object that performs non-interpolative data manipulation, evidence so far indicates that the space of all useful programs may look a lot like a continuous manifold. That means that a deep learning model that has been trained on millions of successful program-generation episodes might start to develop solid intuition about the path through program space that the search process should take to go from a specification to the corresponding program—just like a software engineer might have immediate intuition about the overall architecture of the script they’re about to write and about the intermediate functions and classes they should use as stepping stones on the way to the goal. Remember that human reasoning is heavily guided by value-centric abstraction— that is, by pattern recognition and intuition. The same should be true of program synthesis. I expect the general approach of guiding program search via learned heuristics to see increasing research interest over the next 10 to 20 years.

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

### PDF p.618 (printed 594) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
that humans maintain about their surroundings and themselves, and they will be capable of stronger generalization due to their rich algorithmic nature.

In particular, models will blend algorithmic modules providing formal reasoning, search, and abstraction capabilities with geometric modules providing informal intuition and pattern-recognition capabilities. This will achieve a blend of value-centric and program-centric abstraction. AlphaGo or self-driving cars (systems that required a lot of manual software engineering and human-made design decisions) provide an early example of what such a blend of symbolic and geometric AI could look like.

Such models will be grown automatically rather than hardcoded by human engineers, using modular parts stored in a global library of reusable subroutines—a library evolved by learning high-performing models on thousands of previous tasks and datasets. As frequent problem-solving patterns are identified by the meta-learning system, they will be turned into reusable subroutines—much like functions and classes in software engineering—and added to the global library.

The process that searches over possible combinations of subroutines to grow new 

### PDF p.608 (printed 584) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
And so, in March 2025, the ARC Prize Foundation introduced ARC-AGI-2. It kept the exact same format as the first version but significantly improved the task content. The new iteration was designed to raise the bar, incorporating tasks that demand more complex reasoning chains and are inherently more resistant to exhaustive search methods. The goal was to create a benchmark where computational efficiency becomes a more critical factor for success, pushing systems toward more genuinely intelligent, efficient strategies rather than simply exploring billions of possibilities. While most ARC-AGI-1 tasks could be solved almost instantaneously by a human without requiring much cognitive effort, all tasks in ARC-AGI-2 require some amount of deliberate thinking (see figure 19.11)—for instance, the average time for task completion among human test takers in our experiments was 5 minutes.

Figure 19.11 Typical ARC-AGI-1 task (left) vs. typical ARC-AGI-2 task (right)

The initial AI testing results on ARC-AGI 2 were sobering: even o3 struggled significantly with this new set of challenges, its scores plummeting back into the low double digits when constrained to reasonable computational budget

### PDF p.613 (printed 589) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 19.4.4 An alternative approach to AI: Program synthesis
Until 2024, AI systems capable of genuine discrete reasoning were all hardcoded by human programmers—for instance, software that relies on search algorithms, graph manipulation, and formal logic. In the test-time adaptation (TTA) era, this is finally starting to change. A branch of TTA that is especially promising is program synthesis—a field that is still very niche today, but that I expect to take off in a big way over the next few decades.

Program synthesis consists of automatically generating simple programs by using a search algorithm (possibly genetic search, as in genetic programming) to explore a large space of possible programs (see figure 19.14). The search stops when a program is found that matches the required specifications, often provided as a set of input-output pairs. This is highly reminiscent of machine learning: given training data provided as input-output pairs, we find a program that matches inputs to outputs and can generalize to new inputs. The difference is that instead of learning parameter values in a hardcoded program (a neural network), we generate source code via a discrete search process (see table 19.2).

### PDF p.615 (printed 591) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
broader than what can be represented with current deep learning models, and some of these programs could achieve superior generalization power. Importantly, such programs will not be differentiable end to end, although specific modules will remain differentiable, and thus will need to be generated via a combination of discrete program search and gradient descent.

We’ll move away from having, on one hand, hardcoded algorithmic intelligence (handcrafted software) and, on the other hand, learned geometric intelligence (deep learning). Instead, we’ll have a blend of formal algorithmic modules that provide reasoning and abstraction capabilities and geometric modules that provide informal intuition and pattern-recognition capabilities (figure 19.15). The entire system will be learned with little or no human involvement. This should dramatically expand the scope of problems that can be solved with machine learning—the space of programs that we can generate automatically, given appropriate training data. Systems like AlphaGo—or even RNNs—can be seen as a prehistoric ancestor of such hybrid algorithmic-geometric models.

Figure 19.15 A learned program relying on both geometric primitives (

### PDF p.606 (printed 582) — 19 The future of AI › 19.3 How to build intelligence › 
The ARC-AGI benchmark has proven remarkably resistant to the prevailing deep learning scaling paradigm. Most other benchmarks have saturated quickly in the age of LLMs. That’s because they can be hacked via memorization, whereas ARC-AGI is designed to be resistant to it. From 2019, when ARC-AGI was first released, to 2025, base LLMs underwent a roughly 50,000× scale-up—from, say, GPT-2 (2019) to GPT-4.5 (2025), but their performance on the 2019 version of ARC-AGI only went from 0% to around 10%. Given that you, reader, would easily score above 95%, that isn’t very good. If you scale up your system by 50,000× and you’re still not making meaningful progress, that’s like a big warning sign telling you that you need to try new ideas. Simply making models bigger or training them on more data has not unlocked the kind of fluid intelligence that ARC-AGI requires. ARC-AGI was clearly showing that on-the-fly recombination capabilities are necessary to tackle reasoning.
