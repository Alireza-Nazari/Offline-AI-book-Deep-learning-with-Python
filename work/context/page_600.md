# Generation bundle for page_600  (PDF page 600, printed page 576)

Page kind: content
Chapter: 19 The future of AI
Section: 19.3 How to build intelligence
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_600.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_600.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p600_e03] (body) generalization, precipitating the start of the Anthropocene and forever changing the history of life on Earth.
[p600_e04] (body) The progress of AI over the past 70 years bears striking similarities to this evolution. Early AI systems were pure automatons, like the ELIZA chat program from the 1960s, or SHRDLU,2 a 1970 AI capable of manipulating simple objects from natural language commands. In the 1990s and 2000s, we saw the rise of machine learning systems capable of local generalization that could deal with some level of uncertainty and novelty. In the 2010s, deep learning further expanded the local generalization power of these systems by enabling engineers to use much larger datasets and much more expressive models. Today, we may be on the cusp of the next evolutionary step. We are moving toward systems that achieve broad generalization, which I define as the ability to deal with unknown unknowns within a single broad domain of tasks (including situations the system was not trained to handle and that its creators could not have anticipated). Examples are a self-driving car capable of safely dealing with any situation you throw at it or a domestic robot that could pass the “Woz test of intelligence”—entering a random kitchen and making a cup of coffee.3 By combining deep learning and painstakingly handcrafted abstract models of the world, we’re already making visible progress toward these goals. However, the deep learning paradigm has remained limited to cognitive automation: The “intelligence” label in “artificial intelligence” has been a category error. It would be more accurate to call our field “artificial cognition,” with “cognitive automation” and “artificial intelligence” being two nearly independent subfields within it. In this subdivision, AI would be a greenfield where almost everything remains to be discovered. Now, I don’t mean to diminish the achievements of deep learning. Cognitive automation is incredibly useful, and the way deep learning models are capable of automating tasks from exposure to data alone represents an especially powerful form of cognitive automation, far more practical and versatile than explicit programming. Doing this well is a game changer for essentially every industry. But it’s still a long way from human (or animal) intelligence. Our models, so far, can only perform local generalization: they map space X to space Y via a smooth geometric transform learned from a dense sampling of X-to-Y data points, and any disruption within spaces X or Y invalidates this mapping. They can only generalize to new situations that stay similar to past data, whereas human cognition is capable of extreme generalization, quickly adapting to radically novel situations and planning for long-term future situations.
[p600_e05] (heading_section) 19.3 How to build intelligence
[p600_e06] (body) So far, you’ve learned that there’s a lot more to intelligence than the sort of latent manifold interpolation that deep learning does. But what, then, do we need to start building real intelligence? What are the core pieces that are currently eluding us?
[p600_e07] (footnote) 2 Terry Winograd, “Procedures as a Representation for Data in a Computer Program for Understanding Natural Language,” 1971. 3 Michael Shick, “Wozniak: Could a Computer Make a Cup of Coffee?” Fast Company, March 2010.

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

## Previous page (PDF 599) text, for continuity
hand, humans are able to learn safe behaviors without having to die even once—again, thanks to our power of abstract modeling of novel situations.

19.2.3 The purpose of intelligence

This distinction between highly adaptable intelligent agents and rigid automatons leads us back to brain evolution. Why did brains—originally a mere medium for natural evolution to develop behavioral automatons—eventually turn intelligent? Like every significant evolutionary milestone, it happened because natural selection constraints encouraged it to happen.

Brains are responsible for behavior generation. If the set of situations an organism had to face was mostly static and known in advance, behavior generation would be an easy problem: evolution would just figure out the correct behaviors via random trial and error and hardcode them into the organism’s DNA. This first stage of brain evolution— brains as automatons—would already be optimal. However, crucially, as organism complexity—and alongside it, environmental complexity—kept increasing, the situations animals had to deal with became much more dynamic and more unpredictable. A day in your life, if you look closely, is unlike any day you’ve ever experienced and unlike any day ever experienced by any of your evolutionary ancestors. You need to be able to face unknown and surprising situations constantly. There is no way for evolution to find and hardcode as DNA the sequence of behaviors you’ve been executing to successfully navigate your day since you woke up a few hours ago. It has to be generated on the fly every day. The brain, as a good behavior-generation engine, simply adapted to fit this need. It optimized for adaptability and generality themselves, rather than merely optimizing for fitness to a fixed set of situations. This sh …

## Next page (PDF 601) text, for continuity
19.3.1 The kaleidoscope hypothesis

Intelligence is the ability to use your past experience (and innate prior knowledge) to face novel, unexpected future situations. Now, if the future you had to face was truly novel—sharing no common ground with anything you’ve seen before—you’d be unable to react to it, no matter how intelligent you are.

Intelligence works because nothing is ever truly without precedent. When we encounter something new, we’re able to make sense of it by drawing analogies to our past experience and articulating it in terms of the abstract concepts we’ve collected over time. A person from the 17th century seeing a jet plane for the first time might describe it as a large, loud metal bird that doesn’t flap its wings. A car? That’s a horseless carriage. If you’re trying to teach physics to a grade schooler, you can explain how electricity is like water in a pipe or how spacetime is like a rubber sheet getting distorted by heavy objects. Besides such clear-cut, explicit analogies, we’re constantly making smaller, implicit analogies—every second, with every thought. Analogies are how we navigate life. Shopping at a new supermarket? You’ll find your way by relating it to similar stores you’ve been to. Talking to someone new? They’ll remind you of a few people you’ve met before. Even seemingly random patterns, like the shape of clouds, instantly evoke in us vivid images—an elephant, a ship, a fish. These analogies aren’t just in our minds, either: physical reality itself is full of isomorphisms. Electromagnetism is analogous to gravity. Animals are all structurally similar to each other, due to shared origins. Silica crystals are similar to ice crystals. And so on. I call this the kaleidoscope hypothesis: our experience of the world seems to feature incredib …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.597 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.37 — 1 What is deep learning? › 1.10 Beware of the short-term hype
- PDF p.604 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.595 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.26 — 1 What is deep learning? › 1.2 Artificial intelligence
- PDF p.618 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.614 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.597 (printed 573) — 19 The future of AI › 19.2 Scale isn’t all you need › 19.2.2 Local generalization vs. extreme generalization
The field of AI has long suffered from conflating the notions of intelligence and automation. An automation system (or automaton) is static, crafted to accomplish specific things in a specific context—”If this, then that”—while an intelligent agent can adapt on the fly to novel, unexpected situations. When an automaton is exposed to something that doesn’t match what it was “programmed” to do (whether we’re talking about human-written programs, evolution-generated programs, or the implicit programming process of fitting a model on a training dataset), it will fail.

Meanwhile, intelligent agents, like us humans, will use their fluid intelligence to find a way forward. How do you tell the difference between a student who has memorized the past three years of exam questions but has no understanding of the subject and one who actually understands the material? You give them a brand-new problem. Humans are capable of far more than mapping immediate stimuli to immediate responses as a deep network or an insect would. We can assemble on-the-fly complex, abstract models of our current situation, of ourselves, and of other people, and can use these models to anticipate different possible fu

### PDF p.37 (printed 13) — 1 What is deep learning? › 1.10 Beware of the short-term hype › 
In 2016, OpenAI’s recruiting pitch was that it would achieve AGI by 2020! To be fair, though, only a minority of people in the tech industry believed in such an optimistic timeline back then. By early 2023, however, a significant fraction of engineers in the San Francisco Bay Area seemed convinced that AGI would be coming within the following couple of years. It’s crucial to approach such claims with a healthy dose of skepticism. Despite its name, today’s “artificial intelligence” is more accurately described as “cognitive automation”---the encoding and operationalization of human skills and knowledge. AI excels at solving problems with narrowly defined requirements or those where ample precise examples are available. It’s about enhancing the capabilities of computers, not about replicating human minds. To be clear, cognitive automation is incredibly useful. But intelligence—cognitive autonomy—is a different creature altogether. Think of it this way: AI is like a cartoon character, while intelligence is like a living being. A cartoon, no matter how realistic, can only act out the scenes it was drawn for. A living being, on the other hand, can adapt to the unexpected. “If the cartoo

### PDF p.620 (printed 596) — 20 Conclusions › 20.1 Key concepts in review › 20.1.1 Various approaches to artificial intelligence
First, deep learning isn’t synonymous with artificial intelligence (AI), or even with machine learning:

Artificial intelligence (AI) is an ancient, broad field that can generally be understood as “all attempts to automate human cognitive processes.” This can range from the very basic, such as an Excel spreadsheet, to the very advanced, like a humanoid robot that can walk and talk.

Machine learning is a specific subfield of AI that aims at automatically developing programs (called models) purely from exposure to training data. This process of turning data into a program is called learning. Although machine learning has been around for a long time, it only started to take off in the 1990s, before becoming the dominant form of AI in the 2000s.

Deep learning is one of many branches of machine learning, where the models are long chains of geometric transformations, applied one after the other. These operations are structured into modules called layers: deep learning models are typically stacks of layers—or, more generally, graphs of layers. These layers are parameterized by weights, which are the parameters learned during training. The knowledge of a model is stored in its weights, a

### PDF p.604 (printed 580) — 19 The future of AI › 19.3 How to build intelligence › 19.3.4 A new target: On-the-fly adaptation
To make AI actually intelligent and give it the ability to deal with the incredible variability and ever-changing nature of the real world, first, we need to move away from seeking to achieve task-specific skill and, instead, start targeting generalization power itself. We need new metrics of progress that will help us develop increasingly intelligent systems: metrics that will point in the right direction and that will give us an actionable feedback signal. As long as we set our goal to be “create a model that solves task X,” the shortcut rule will apply, and we’ll end up with a model that does X, period.

In my view, intelligence can be precisely quantified as an efficiency ratio: the conversion ratio between the amount of relevant information you have available about the world (which could be either past experience or innate prior knowledge) and your future operating area, the set of novel situations where you will be able to produce appropriate behavior (you can view this as your skill set). A more intelligent agent will be able to handle a broader set of future tasks and situations using a smaller amount of past experience. To measure such a ratio, you just need to fix the inf

### PDF p.615 (printed 591) — 19 The future of AI › 19.4 The missing ingredients: Search and symbols › 
broader than what can be represented with current deep learning models, and some of these programs could achieve superior generalization power. Importantly, such programs will not be differentiable end to end, although specific modules will remain differentiable, and thus will need to be generated via a combination of discrete program search and gradient descent.

We’ll move away from having, on one hand, hardcoded algorithmic intelligence (handcrafted software) and, on the other hand, learned geometric intelligence (deep learning). Instead, we’ll have a blend of formal algorithmic modules that provide reasoning and abstraction capabilities and geometric modules that provide informal intuition and pattern-recognition capabilities (figure 19.15). The entire system will be learned with little or no human involvement. This should dramatically expand the scope of problems that can be solved with machine learning—the space of programs that we can generate automatically, given appropriate training data. Systems like AlphaGo—or even RNNs—can be seen as a prehistoric ancestor of such hybrid algorithmic-geometric models.

Figure 19.15 A learned program relying on both geometric primitives (

### PDF p.603 (printed 579) — 19 The future of AI › 19.3 How to build intelligence › 
So far, the driving success metric of the field of AI has been to solve specific tasks, from chess

to Go, from MNIST classification to ImageNet, from high school math tests to the bar exam. Consequently, the history of the field has been defined by a series of “successes” where we figured out how to solve these tasks without featuring any intelligence.

If that sounds like a surprising statement, keep in mind that human-like intelligence isn’t characterized by skill at any particular task—rather, it is the ability to adapt to novelty to efficiently acquire new skills and master never-before-seen tasks. By fixing the task, you make it possible to provide an arbitrarily precise description of what needs to be done—either via hardcoding human-provided knowledge or by supplying humongous amounts of data. You make it possible for engineers to “buy” more skill for their AI by just adding data or adding hardcoded knowledge, without increasing the generalization power of the AI (see figure 19.8). If you have near-infinite training data, even a very crude algorithm like nearest-neighbor search can play video games with superhuman skill. Likewise, if you have a near-infinite amount of human

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

### PDF p.37 (printed 13) — 1 What is deep learning? › 1.10 Beware of the short-term hype › 
If a large language model can output a sufficiently human-sounding answer when asked a question, does it matter if it possesses true cognitive autonomy? The key difference is adaptability. Intelligence is the ability to face the unknown, adapt to it, and learn from it. Automation, even at its best, can only handle situations it’s been trained on or programmed for. That’s why creating robust automation is so challenging—it requires accounting for every possible scenario. So don’t worry about AI suddenly becoming self-aware and taking over humanity. Today’s technology simply isn’t headed in that direction. Even with significant
