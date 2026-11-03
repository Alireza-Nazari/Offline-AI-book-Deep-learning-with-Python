# Generation bundle for page_027  (PDF page 27, printed page 3)

Page kind: content
Chapter: 1 What is deep learning?
Section: 1.3 Machine learning
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_027.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_027.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p027_e03] (body) knowledge stored in explicit databases. This approach is known as symbolic AI. It was the dominant paradigm in AI from the 1950s to the late 1980s, and it reached its peak popularity during the expert systems boom of the 1980s.
[p027_e04] (body) Although symbolic AI proved suitable to solve well-defined, logical problems, such as playing chess, it turned out to be intractable to figure out explicit rules for solving more complex, fuzzy problems, such as image classification, speech recognition, or natural language translation. A new approach arose to take symbolic AI’s place: machine learning.
[p027_e05] (heading_section) 1.3
[p027_e06] (heading_section) Machine learning
[p027_e07] (body) In Victorian England, Lady Ada Lovelace was a friend and collaborator of Charles Babbage, the inventor of the Analytical Engine: the first-known general-purpose mechanical computer. Although visionary and far ahead of its time, the Analytical Engine wasn’t meant as a general-purpose computer when it was designed in the 1830s and 1840s, because the concept of general-purpose computation was yet to be invented. It was merely meant as a way to use mechanical operations to automate certain computations from the field of mathematical analysis—hence the name Analytical Engine. As such, it was the intellectual descendant of earlier attempts at encoding mathematical operations in gear form, such as the Pascaline, or Leibniz’s step reckoner, a refined version of the Pascaline. Designed by Blaise Pascal in 1642 (at age 19!), the Pascaline was the world’s first mechanical calculator—it could add, subtract, multiply, or even divide digits.
[p027_e08] (body) In 1843, Ada Lovelace remarked on the invention of the Analytical Engine:
[p027_e09] (quote) The Analytical Engine has no pretensions whatever to originate anything. It can do whatever we know how to order it to perform… Its province is to assist us in making available what we’re already acquainted with.
[p027_e10] (body) Even with 182 years of historical perspective, Lady Lovelace’s observation remains arresting. Could a general-purpose computer “originate” anything, or would it always be bound to dully execute processes we humans fully understand? Could it ever be capable of any original thought? Could it learn from experience? Could it show creativity?
[p027_e11] (body) Her remark was later quoted by AI pioneer Alan Turing as “Lady Lovelace’s objection” in his landmark 1950 paper “Computing Machinery and Intelligence,”1 which introduced the Turing test 2 as well as key concepts that would come to shape AI. Turing was of the opinion—highly provocative at the time—that computers could, in principle, be made to emulate all aspects of human intelligence. The usual way to make a computer do useful work is to have a human programmer write down rules—a computer program—to be followed to turn input data into appropriate answers, just like Lady Lovelace writing down step-by-step instructions for the Analytical
[p027_e12] (footnote) 1 A. M. Turing, “Computing Machinery and Intelligence,” Mind 59, no. 236 (1950): 433-460. 2 Although the Turing test has sometimes been interpreted as a literal test—a goal the field of AI should set out to reach—Turing merely meant it as a conceptual device in a philosophical discussion about the nature of cognition.

## Chapter outline
Chapter: 1 What is deep learning? (PDF pages 25-39)
  - 1.1 Artificial intelligence, machine learning, and deep learning (p.26)
  - 1.2 Artificial intelligence (p.26)
  - 1.3 Machine learning (p.27)
  - 1.4 Learning rules and representations from data (p.28)
  - 1.5 The “deep” in “deep learning” (p.31)
  - 1.6 Understanding how deep learning works, in three figures (p.32)
  - 1.7 What makes deep learning different (p.34)
  - 1.8 The age of generative AI (p.35)
  - 1.9 What deep learning has achieved so far (p.35)
  - 1.10 Beware of the short-term hype (p.36)
  - 1.11 Summer can turn to winter (p.38)
  - 1.12 The promise of AI (p.38)

## Previous page (PDF 26) text, for continuity
tackle these questions: What has deep learning achieved so far? How significant is it? Where are we headed next? Should you believe the hype?

1.1

Artificial intelligence, machine learning, and deep learning

First, we need to define clearly what we’re talking about when we mention AI. What are artificial intelligence, machine learning, and deep learning (figure 1.1)? How do they relate to each other?

1.2

Artificial intelligence

Artificial intelligence was born in the 1950s, when a handful of pioneers from the nascent field of computer sci

Figure 1.1 Artificial intelligence, machine learning,

ence started asking whether computers

and deep learning

could be made to “think”—a question whose ramifications we’re still exploring today.

While many of the underlying ideas had been brewing in the years and even decades prior, “artificial intelligence” finally crystallized as a field of research in 1956, when John McCarthy, then a young Assistant Professor of Mathematics at Dartmouth College, organized a summer workshop under the following proposal:

The study is to proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it. An attempt will be made to find how to make machines use language, form abstractions and concepts, solve kinds of problems now reserved for humans, and improve themselves. We think that a significant advance can be made in one or more of these problems if a carefully selected group of scientists work on it together for a summer.

At the end of the summer, the workshop concluded without having fully solved the riddle it set out to investigate. Nevertheless, it was attended by many people who would move on to become pione …

## Next page (PDF 28) text, for continuity
Engine to perform. Machine learning turns this around: the machine looks at the input data and the corresponding answers, and figures out what the rules should be (figure 1.2).

A machine learning system is trained rather than explicitly programmed. It’s presented with many examples relevant to a task, and

Figure 1.2 Machine learning: a new

it finds statistical structure in these examples

programming paradigm

that eventually allows the system to come up with rules for automating the task. For instance, if you wished to automate the task of tagging your vacation pictures, you could present a machine learning system with many examples of pictures already tagged by humans, and the system would learn statistical rules for associating specific pictures to specific tags like “landscape” or “food.”

Although machine learning only started to flourish in the 1990s, it has quickly become the most popular and most successful subfield of AI, a trend driven by the availability of faster hardware and larger datasets. Machine learning is related to mathematical statistics, but it differs from statistics in several important ways—in the same sense that medicine is related to chemistry but cannot be reduced to chemistry, as medicine deals with its own distinct systems with their own distinct properties. Unlike statistics, machine learning tends to deal with large, complex datasets (such as a dataset of millions of images, each consisting of tens of thousands of pixels) for which classical statistical analysis such as Bayesian analysis would be impractical. As a result, machine learning, and especially deep learning, exhibits comparatively little mathematical theory—maybe too little—and is fundamentally an engineering discipline. Unlike theoretical physics or mathematics, machine lea …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.38 — 1 What is deep learning? › 1.12 The promise of AI
- PDF p.603 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.37 — 1 What is deep learning? › 1.10 Beware of the short-term hype
- PDF p.600 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.631 — 20 Conclusions › 20.4 Staying up to date in a fast-moving field
- PDF p.35 — 1 What is deep learning? › 1.9 What deep learning has achieved so far
- PDF p.30 — 1 What is deep learning? › 1.4 Learning rules and representations from data

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.620 (printed 596) — 20 Conclusions › 20.1 Key concepts in review › 20.1.1 Various approaches to artificial intelligence
First, deep learning isn’t synonymous with artificial intelligence (AI), or even with machine learning:

Artificial intelligence (AI) is an ancient, broad field that can generally be understood as “all attempts to automate human cognitive processes.” This can range from the very basic, such as an Excel spreadsheet, to the very advanced, like a humanoid robot that can walk and talk.

Machine learning is a specific subfield of AI that aims at automatically developing programs (called models) purely from exposure to training data. This process of turning data into a program is called learning. Although machine learning has been around for a long time, it only started to take off in the 1990s, before becoming the dominant form of AI in the 2000s.

Deep learning is one of many branches of machine learning, where the models are long chains of geometric transformations, applied one after the other. These operations are structured into modules called layers: deep learning models are typically stacks of layers—or, more generally, graphs of layers. These layers are parameterized by weights, which are the parameters learned during training. The knowledge of a model is stored in its weights, a

### PDF p.38 (printed 14) — 1 What is deep learning? › 1.12 The promise of AI › 1.11 Summer can turn to winter
The danger of inflated short-term expectations is that when technology inevitably falls short, research investment could dry up, slowing progress for a long time. This has happened before. Twice in the past, AI went through a cycle of intense optimism followed by disappointment and skepticism, with a dearth of funding as a result. It started with symbolic AI in the 1960s. In those early days, projections about AI were flying high. One of the best-known pioneers and proponents of the symbolic AI approach was Marvin Minsky, who claimed in 1967, “Within a generation . . . the problem of creating ‘artificial intelligence’ will substantially be solved.” Three years later, in 1970, he made a more precisely quantified prediction: “In from three to eight years we will have a machine with the general intelligence of an average human being.” In 2025, such an achievement still appears to be far in the future—in that we have no way to predict how long it will take—but in the 1960s and early 1970s, several experts believed it to be right around the corner (as do many people today). A few years later, as these high expectations failed to materialize, researchers and government funds turned away 

### PDF p.600 (printed 576) — 19 The future of AI › 19.3 How to build intelligence › 
The progress of AI over the past 70 years bears striking similarities to this evolution. Early AI systems were pure automatons, like the ELIZA chat program from the 1960s, or SHRDLU,2 a 1970 AI capable of manipulating simple objects from natural language commands. In the 1990s and 2000s, we saw the rise of machine learning systems capable of local generalization that could deal with some level of uncertainty and novelty. In the 2010s, deep learning further expanded the local generalization power of these systems by enabling engineers to use much larger datasets and much more expressive models. Today, we may be on the cusp of the next evolutionary step. We are moving toward systems that achieve broad generalization, which I define as the ability to deal with unknown unknowns within a single broad domain of tasks (including situations the system was not trained to handle and that its creators could not have anticipated). Examples are a self-driving car capable of safely dealing with any situation you throw at it or a domestic robot that could pass the “Woz test of intelligence”—entering a random kitchen and making a cup of coffee.3 By combining deep learning and painstakingly handcra

### PDF p.603 (printed 579) — 19 The future of AI › 19.3 How to build intelligence › 
optimized for prediction accuracy alone—what they were incentivized to achieve—at the expense of every other desirable characteristic of the system: inference cost, maintainability, explainability. The shortcut rule holds true in most Kaggle competitions as well—the models produced by Kaggle winners can rarely, if ever, be used in production.

The shortcut rule has been everywhere in AI over the past few decades. In the 1970s, psychologist and computer science pioneer Allen Newell, concerned that his field wasn’t making any meaningful progress toward a proper theory of cognition, proposed a new grand goal for AI: chess playing. The rationale was that playing chess, in humans, seemed to involve—perhaps even require—capabilities such as perception, reasoning and analysis, memory and study from books, and so on. Surely, if we could build a chess-playing machine, it would have to feature these attributes as well. Right? Over two decades later, the dream came true: in 1997, IBM’s Deep Blue beat Gary Kasparov, the best chess player in the world. Researchers had then to contend with the fact that creating a chess-champion AI had taught them little about human intelligence. The A-star algo

### PDF p.603 (printed 579) — 19 The future of AI › 19.3 How to build intelligence › 
So far, the driving success metric of the field of AI has been to solve specific tasks, from chess

to Go, from MNIST classification to ImageNet, from high school math tests to the bar exam. Consequently, the history of the field has been defined by a series of “successes” where we figured out how to solve these tasks without featuring any intelligence.

If that sounds like a surprising statement, keep in mind that human-like intelligence isn’t characterized by skill at any particular task—rather, it is the ability to adapt to novelty to efficiently acquire new skills and master never-before-seen tasks. By fixing the task, you make it possible to provide an arbitrarily precise description of what needs to be done—either via hardcoding human-provided knowledge or by supplying humongous amounts of data. You make it possible for engineers to “buy” more skill for their AI by just adding data or adding hardcoded knowledge, without increasing the generalization power of the AI (see figure 19.8). If you have near-infinite training data, even a very crude algorithm like nearest-neighbor search can play video games with superhuman skill. Likewise, if you have a near-infinite amount of human

### PDF p.631 (printed 607) — 20 Conclusions › 20.4 Staying up to date in a fast-moving field › 
this problem-solving loop, the new component would get added to the library, becoming available for every future task the system might encounter. Ultimately, developing AI that mirrors human-like fluid intelligence will require blending continuous pattern recognition together with discrete, symbolic programs and fully embracing the paradigm of on-the-fly adaptation.

### PDF p.8 (printed viii) — contents ›  › contents
preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

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
