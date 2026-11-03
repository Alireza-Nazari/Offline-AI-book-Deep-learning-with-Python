# Generation bundle for page_604  (PDF page 604, printed page 580)

Page kind: content
Chapter: 19 The future of AI
Section: 19.3 How to build intelligence
Subsection: 19.3.4 A new target: On-the-fly adaptation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_604.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_604.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p604_e03] (figure_caption) Figure 19.8 A low-generalization system can achieve arbitrary skill at a fixed task given unlimited task-specific information.
[p604_e04] (heading_subsection) 19.3.4 A new target: On-the-fly adaptation
[p604_e05] (body) To make AI actually intelligent and give it the ability to deal with the incredible variability and ever-changing nature of the real world, first, we need to move away from seeking to achieve task-specific skill and, instead, start targeting generalization power itself. We need new metrics of progress that will help us develop increasingly intelligent systems: metrics that will point in the right direction and that will give us an actionable feedback signal. As long as we set our goal to be “create a model that solves task X,” the shortcut rule will apply, and we’ll end up with a model that does X, period.
[p604_e06] (body) In my view, intelligence can be precisely quantified as an efficiency ratio: the conversion ratio between the amount of relevant information you have available about the world (which could be either past experience or innate prior knowledge) and your future operating area, the set of novel situations where you will be able to produce appropriate behavior (you can view this as your skill set). A more intelligent agent will be able to handle a broader set of future tasks and situations using a smaller amount of past experience. To measure such a ratio, you just need to fix the information available to your system—its experience and its prior knowledge—and measure its performance on a set of reference situations or tasks that are known to be sufficiently different from what the system has had access to. Trying to maximize this ratio should lead you toward intelligence. Crucially, to avoid cheating, you’re going to need to make sure to test the system only on tasks it wasn’t programmed or trained to handle—in fact, you need tasks that the creators of the system could not have anticipated. In 2018 and 2019, I developed a benchmark dataset called the Abstraction & Reasoning Corpus for Artificial General Intelligence (ARC-AGI) 4 that seeks to capture this definition
[p604_e07] (footnote) 4 François Chollet, “On the Measure of Intelligence,” 2019.

## Figures
- Figure 19.8: A low-generalization system can achieve arbitrary skill at a fixed task given unlimited task-specific information.  | labels inside the figure: (none - image)

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

## Previous page (PDF 603) text, for continuity
optimized for prediction accuracy alone—what they were incentivized to achieve—at the expense of every other desirable characteristic of the system: inference cost, maintainability, explainability. The shortcut rule holds true in most Kaggle competitions as well—the models produced by Kaggle winners can rarely, if ever, be used in production.

The shortcut rule has been everywhere in AI over the past few decades. In the 1970s, psychologist and computer science pioneer Allen Newell, concerned that his field wasn’t making any meaningful progress toward a proper theory of cognition, proposed a new grand goal for AI: chess playing. The rationale was that playing chess, in humans, seemed to involve—perhaps even require—capabilities such as perception, reasoning and analysis, memory and study from books, and so on. Surely, if we could build a chess-playing machine, it would have to feature these attributes as well. Right? Over two decades later, the dream came true: in 1997, IBM’s Deep Blue beat Gary Kasparov, the best chess player in the world. Researchers had then to contend with the fact that creating a chess-champion AI had taught them little about human intelligence. The A-star algorithm at the heart of Deep Blue wasn’t a model of the human brain and couldn’t generalize to tasks other than similar board games. It turned out it was easier to build an AI that could only play chess than to build an artificial mind—so that’s the shortcut researchers took.

So far, the driving success metric of the field of AI has been to solve specific tasks, from chess

to Go, from MNIST classification to ImageNet, from high school math tests to the bar exam. Consequently, the history of the field has been defined by a series of “successes” where we figured out how to solve these tasks with …

## Next page (PDF 605) text, for continuity
of intelligence. ARC-AGI is meant to be approachable by both machines and humans, and it looks very similar to human IQ tests, such as Raven’s progressive matrices. At test time, you’ll see a series of “tasks.” Each task is explained via three or four “examples” that take the form of an input grid and a corresponding output grid (see figure 19.9). You’ll then be given a brand-new input grid, and you’ll have three tries to produce the correct output grid, before moving on to the next task.

Figure 19.9 An ARC-AGI task: the nature of the task is demonstrated by a couple of input-output pair examples. Provided with a new input, you must construct the corresponding output.

Compared to IQ tests, two things are unique about ARC-AGI. First, ARC seeks to measure generalization power by only testing you on tasks you’ve never seen before. That means that ARC-AGI is a game you can’t practice for, at least in theory: the tasks you will get tested on will have their own unique logic that you will have to understand on the fly. You can’t just memorize specific strategies from past tasks.

In addition, ARC-AGI tries to control for the prior knowledge that you bring to the test. You never approach a new problem entirely from scratch—you bring to it preexisting skills and information. ARC-AGI makes the assumption that all test takers should start from the set of knowledge priors, called Core Knowledge priors, which represent the knowledge systems humans are born with. Unlike an IQ test, ARC-AGI tasks will never involve acquired knowledge, like English sentences, for instance.

19.3.5 ARC Prize

In 2024, to accelerate progress toward AI systems capable of the kind of fluid abstraction and reasoning measured by ARC-AGI, I partnered with Mike Knoop to establish the nonprofit ARC Prize Fou …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.600 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.602 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.599 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.618 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.617 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.597 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.595 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.608 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.597 (printed 573) — 19 The future of AI › 19.2 Scale isn’t all you need › 19.2.2 Local generalization vs. extreme generalization
The field of AI has long suffered from conflating the notions of intelligence and automation. An automation system (or automaton) is static, crafted to accomplish specific things in a specific context—”If this, then that”—while an intelligent agent can adapt on the fly to novel, unexpected situations. When an automaton is exposed to something that doesn’t match what it was “programmed” to do (whether we’re talking about human-written programs, evolution-generated programs, or the implicit programming process of fitting a model on a training dataset), it will fail.

Meanwhile, intelligent agents, like us humans, will use their fluid intelligence to find a way forward. How do you tell the difference between a student who has memorized the past three years of exam questions but has no understanding of the subject and one who actually understands the material? You give them a brand-new problem. Humans are capable of far more than mapping immediate stimuli to immediate responses as a deep network or an insect would. We can assemble on-the-fly complex, abstract models of our current situation, of ourselves, and of other people, and can use these models to anticipate different possible fu

### PDF p.602 (printed 578) — 19 The future of AI › 19.3 How to build intelligence › 19.3.3 The importance of setting the right target
Biological intelligence was the answer to a question asked by nature. Likewise, if we want to develop true AI, first, we need to be asking the right questions. Ultimately, the capabilities of AI systems reflect the objectives they were designed and optimized for.

An effect you see constantly in systems design is the shortcut rule: if you focus on optimizing one success metric, you will achieve your goal, but at the expense of everything in the system that wasn’t covered by your success metric. You end up taking every available shortcut toward the goal. Your creations are shaped by the incentives you give yourself. You see this often in machine learning competitions. In 2009, Netflix ran a challenge that promised a $1 million prize to the team that would achieve the highest score on a movie recommendation task. It ended up never using the system created by the winning team because it was way too complex and compute intensive. The winners had

### PDF p.600 (printed 576) — 19 The future of AI › 19.3 How to build intelligence › 
The progress of AI over the past 70 years bears striking similarities to this evolution. Early AI systems were pure automatons, like the ELIZA chat program from the 1960s, or SHRDLU,2 a 1970 AI capable of manipulating simple objects from natural language commands. In the 1990s and 2000s, we saw the rise of machine learning systems capable of local generalization that could deal with some level of uncertainty and novelty. In the 2010s, deep learning further expanded the local generalization power of these systems by enabling engineers to use much larger datasets and much more expressive models. Today, we may be on the cusp of the next evolutionary step. We are moving toward systems that achieve broad generalization, which I define as the ability to deal with unknown unknowns within a single broad domain of tasks (including situations the system was not trained to handle and that its creators could not have anticipated). Examples are a self-driving car capable of safely dealing with any situation you throw at it or a domestic robot that could pass the “Woz test of intelligence”—entering a random kitchen and making a cup of coffee.3 By combining deep learning and painstakingly handcra

### PDF p.608 (printed 584) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
And so, in March 2025, the ARC Prize Foundation introduced ARC-AGI-2. It kept the exact same format as the first version but significantly improved the task content. The new iteration was designed to raise the bar, incorporating tasks that demand more complex reasoning chains and are inherently more resistant to exhaustive search methods. The goal was to create a benchmark where computational efficiency becomes a more critical factor for success, pushing systems toward more genuinely intelligent, efficient strategies rather than simply exploring billions of possibilities. While most ARC-AGI-1 tasks could be solved almost instantaneously by a human without requiring much cognitive effort, all tasks in ARC-AGI-2 require some amount of deliberate thinking (see figure 19.11)—for instance, the average time for task completion among human test takers in our experiments was 5 minutes.

Figure 19.11 Typical ARC-AGI-1 task (left) vs. typical ARC-AGI-2 task (right)

The initial AI testing results on ARC-AGI 2 were sobering: even o3 struggled significantly with this new set of challenges, its scores plummeting back into the low double digits when constrained to reasonable computational budget

### PDF p.599 (printed 575) — 19 The future of AI › 19.2 Scale isn’t all you need › 19.2.3 The purpose of intelligence
This distinction between highly adaptable intelligent agents and rigid automatons leads us back to brain evolution. Why did brains—originally a mere medium for natural evolution to develop behavioral automatons—eventually turn intelligent? Like every significant evolutionary milestone, it happened because natural selection constraints encouraged it to happen.

Brains are responsible for behavior generation. If the set of situations an organism had to face was mostly static and known in advance, behavior generation would be an easy problem: evolution would just figure out the correct behaviors via random trial and error and hardcode them into the organism’s DNA. This first stage of brain evolution— brains as automatons—would already be optimal. However, crucially, as organism complexity—and alongside it, environmental complexity—kept increasing, the situations animals had to deal with became much more dynamic and more unpredictable. A day in your life, if you look closely, is unlike any day you’ve ever experienced and unlike any day ever experienced by any of your evolutionary ancestors. You need to be able to face unknown and surprising situations constantly. There is no way for ev

### PDF p.607 (printed 583) — 19 The future of AI › 19.3 How to build intelligence › 19.3.7 ARC-AGI 2
Does that mean AGI is solved? Was o3 as intelligent as a human?

Not quite. First, while o3’s performance was a landmark achievement, it came at a tremendous cost—tens of thousands of dollars of compute per ARC-AGI puzzle. Intelligence isn’t just about capability; it’s fundamentally about efficiency. Brute-forcing the solution space given enormous compute is a shortcut that makes all kinds of tasks possible without requiring intelligence. In principle, you could even solve ARC-AGI by simply walking down the tree of every possible solution program and testing each one until you find one that works on the demonstration pairs. The o3 results, impressive as they were, felt more like cracking a code with a supercomputer than a display of nimble, human-like fluid reasoning. The entire point of intelligence is to achieve results with the least amount of resources possible. Second, we found that o3 was still stumped by many tasks that humans found very easy (like the one in figure 19.10). This strongly suggests that o3 wasn’t quite human-level yet. Here’s the thing—the 2019 version of ARC-AGI was intended to be easy. It was essentially a binary test of fluid intelligence: either you have n

### PDF p.602 (printed 578) — 19 The future of AI › 19.3 How to build intelligence › 19.3.2 The essence of intelligence: Abstraction acquisition and recombination
Intelligence is the ability to mine your experience to identify these atoms of meaning that can seemingly be reused across many different situations—the core beads of the kaleidoscope. Once extracted, they’re called abstractions. Whenever you encounter a new situation, you make sense of it by recombining on the fly abstractions from your collections, to weave a brand new “model” adapted to the situation.

This process consists of two key parts:

Abstraction acquisition—Efficiently extracting compact, reusable abstractions from a stream of experience or data. This involves identifying underlying structures, principles, or invariants.

On-the-fly recombination—Efficiently selecting and recombining these abstractions in novel ways to model new problems and situations, even ones far removed from past experience.

The emphasis on efficiency is crucial. How intelligent you are is determined by how efficiently you can acquire good abstractions from limited experience and how efficiently you can recombine them to navigate uncertainty and novelty. If you need hundreds of thousands of hours of practice to acquire a skill, you are not very intelligent. If you need to enumerate every possible 

### PDF p.15 (printed xv) — contents ›  › 
19.2

Scale isn’t all you need 570

Automatons vs. intelligent agents 571 ■ Local generalization vs. extreme generalization 573 ■ The purpose of intelligence 575 Climbing the spectrum of generalization 575
