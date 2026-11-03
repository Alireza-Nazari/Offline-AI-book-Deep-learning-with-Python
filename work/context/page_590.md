# Generation bundle for page_590  (PDF page 590, printed page 566)

Page kind: content
Chapter: 19 The future of AI
Section: 19.1 The limitations of deep learning
Subsection: 19.1.1 Deep learning models struggle to adapt to novelty
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_590.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_590.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p590_e03] (figure_caption) Figure 19.1 An easy yet novel puzzle
[p590_e04] (body) model would just repeat the answer it had memorized without paying any attention to the actual numbers in the query, or what the query really meant. Similarly, LLMs struggle to adapt to variations of the Monty Hall problem (see figure 19.2) and will tend to always output the canonical answer to the puzzle, which they’ve seen many times during training, regardless of whether it makes sense in context.
[p590_e05] (figure_caption) Figure 19.2 A variation of the Monty Hall problem
[p590_e06] (body) To note, these specific prompts were patched later on by special-casing them. Today, there are over 25,000 people who are employed full time to provide training data for LLMs by reviewing failure cases and suggesting better answers. LLM maintenance is a constant game of whack-a-mole where failing prompts are patched one at a time,

## Figures
- Figure 19.1: An easy yet novel puzzle  | labels inside the figure: (none - image)
- Figure 19.2: A variation of the Monty Hall problem  | labels inside the figure: (none - image)

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

## Previous page (PDF 589) text, for continuity
19.1.1 Deep learning models struggle to adapt to novelty

Deep learning models are big, parametric curves fitted to large datasets. That’s the source of their power—they’re easy to train, and they scale really well, both in terms of model size and dataset size. But that’s also a source of significant weaknesses. Curve fitting has inherent limitations.

First and foremost, a parametric curve is only capable of information storage—it’s a kind of database. Recall our discussion of Transformers as an “interpolative database” from chapter 15? Second, crucially, this database is static. The model’s parameters are determined during a distinct “training time” phase. Afterward, these parameters are frozen, and this fixed version is used during “inference time” for making predictions on new data. The only thing you can do with a static database is information retrieval. And that’s exactly what deep learning models excel at: recognizing or generating patterns highly similar to those encountered during training. The flip side is that they are inherently poor at adaptation. The database is backward-looking—it fits past data but can’t handle a changing future. At inference time, you’d better hope that the situations the model faces are part of the training data distribution, because otherwise, the model will break down. A model trained on ImageNet will classify a leopard-print sofa as an actual leopard, for instance—sofas were not part of its training data. This also applies to the largest of generative models. In recent years, the rise of large language models (LLMs) and their application to programming assistance and reasoning-like problems has provided extensive empirical proof of this. Despite frequent claims that LLMs can perform in-context learning to pick up new skills from ju …

## Next page (PDF 591) text, for continuity
without addressing the more general underlying issue. Even already patched prompts will still fail if you make small changes to them!

19.1.2 Deep learning models are highly sensitive to phrasing and other distractors

A closely related problem is the extreme sensitivity of deep learning models to how their input is presented. For instance, image models are affected by adversarial examples, which are samples fed to a deep learning network that are designed to trick the model into misclassifying them. You’re already aware that it’s possible to do gradient ascent in input space to generate inputs that maximize the activation of some ConvNet filter—this is the basis of the filter visualization technique introduced in chapter 10.

Similarly, through gradient ascent, you can slightly modify an image to maximize the class prediction for a given class. By taking a picture of a panda and adding to it a gibbon gradient, we can get a neural network to classify the panda as a gibbon (see figure 19.3). This evidences both the brittleness of these models and the deep difference between their input-to-output mapping and our human perception.

Figure 19.3 An adversarial example: imperceptible changes in an image can upend a model’s classification of the image.

Similarly, LLMs suffer from an extremely high sensitivity to minor details in their prompts. Innocuous prompt modifications, such as changing place and people’s

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.592 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.528 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.526 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.593 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.530 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.529 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.527 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.519 — 16 Text generation › 16.4 Going further with LLMs

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.629 (printed 605) — 20 Conclusions › 20.2 Limitations of deep learning › 20.2 Limitations of deep learning
Building deep learning models is like playing with LEGO bricks: layers can be plugged together to map essentially anything to anything, given that you have appropriate training data available and that the mapping is achievable via a continuous geometric transformation of reasonable complexity.

Here’s the catch, though—this mapping is often not learnable in a way that will generalize. Deep learning models operate like vast, interpolative databases of patterns. Their pattern-matching strength is also their core weakness:

They fundamentally struggle to adapt to novelty. Because their parameters are fixed after training, they can only retrieve or replicate patterns similar to their training data. Faced with inputs significantly outside this familiar distribution—no matter how simple the underlying task—their performance degrades drastically, as they lack mechanisms for fluid generalization beyond their memorized experience. This explains why even large models fail on novel tasks or simple variations of familiar problems—like ARC-AGI tasks.

They’re very sensitive to phrasing and other distractors. Deep learning models exhibit high sensitivity to superficial variations in input presen

### PDF p.593 (printed 569) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.3 Deep learning models struggle to learn generalizable programs
The problem with deep learning models isn’t just that they’re limited to blindly reapplying patterns they’ve memorized at training time or that they’re highly sensitive to how their input is presented. Even if you just need to query and apply a well-known program, and you know exactly how to address this program in latent space, you still face a major issue: the programs memorized by deep learning models often don’t generalize well. They will work for some input values and fail for some other input values. This is especially true for programs that encode any kind of discrete logic.

### PDF p.593 (printed 569) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.3 Deep learning models struggle to learn generalizable programs
Consider the problem of adding two numbers, represented as character sequences— like “4 3 5 7 + 8 9 3 6.” Try training a Transformer on hundreds of thousands of such digit pairs: you will reach a very high accuracy. Very high, but not 100%—you will keep regularly seeing incorrect answers, because the Transformer doesn’t manage to encode the exact addition algorithm (you know, the one you learned in primary school). It is instead guessing the output by interpolating between the data points it has seen at training time. This applies to state-of-the-art LLMs, too—at least those that weren’t explicitly hardcoded to execute snippets like “4357 + 8936” in Python to provide the right answer. They’ve seen enough examples of digit addition that they can add numbers, but they only have about 70% accuracy—quite underwhelming. Further, their accuracy is strongly dependent on which digits are being added, with more common digits leading to higher accuracy. The reason why a deep learning model does not end up learning an exact addition algorithm even after seeing millions of examples is that it is just a static chain of simple, continuous geometric transformations mapping one vector space into a

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › 
We can note that while the total number of tokens used for pretraining has climbed steadily and massively, model parameter counts have varied substantially since GPT-3. In part, this is because we now know GPT-3 was undertrained, but it is also for a more practical reason. When deploying a model, it’s often worth it to sacrifice performance for a smaller model that fits on cheaper hardware. A really good model won’t help very much if it’s prohibitively expensive to run. There’s another reason we might not be able to just scale up these models thoughtlessly: we are starting to run out of pretraining data! Tech companies are starting to have trouble finding more high-quality, public, human-written content to throw at pretraining. Models are even starting to “eat their own tail” by training on a significant portion of content created by other LLMs, which runs into a whole other host of concerns. This is one of the reasons reinforcement learning is getting a lot of attention recently. If you can create a difficult, self-contained environment that generates new problems for an LLM to attempt, you will have found a way to continue training using the model’s own output—no need to scrounge

### PDF p.595 (printed 571) — 19 The future of AI › 19.2 Scale isn’t all you need › 
been released, and it was essentially a scaled-up version of GPT-3: more parameters, more training data. Its significantly improved performance seemed to suggest that you could just keep going—that there could be a GPT-5 that would simply be more of the same and from which artificial general intelligence (AGI) would spontaneously emerge.

Proponents of this view would point to “scaling laws” as evidence. Scaling laws are an empirical relationship observed between the size of a deep learning model (as well as the size of its training dataset) and its performance on specific tasks. They suggest that increasing the size of a model reliably leads to better performance in a predictable manner. But the key thing that scaling law enthusiasts are missing is that the benchmarks they’re using to measure “performance” are effectively memorization tests, the kind we like to give university students. LLMs perform well on these tests by memorizing the answers, and naturally, cramming more questions and more answers into the models improves their performance accordingly. The reality is that scaling up our models hasn’t led to any progress on the issues I’ve listed so far in these pages—inability 

### PDF p.528 (printed 504) — 16 Text generation › 16.5 Where are LLMs heading next? › 
In the first attempt, our model was hung up on the superfluous detail that each letter has two pages. In the second attempt, the model gets the problem right. This instruction-tuned Gemma model we are working with has already been tuned on math problems like this; you would not get nearly as good results from the “untuned” Gemma model from the last section.

We could extend this idea to a very simple form of chain-of-thought training: 1 Collect a bunch of basic math and reasoning problems and desired answers. 2 Generate (with some randomness) a number of responses. 3 Find all the responses with a correct answer via string parsing. You can prompt

the model to use a specific text marker for the final answer as we did previously. 4 Run supervised fine-tuning on correct responses, including all the intermediate output. 5 Repeat! The previously described process is a reinforcement learning algorithm. Our answer checking acts as the environment, and the generated outputs are the actions the model uses to learn. As with RLHF, in practice you would use a more complex gradient update step to use information from all responses (even the incorrect ones), but the basic principle is the same. 

### PDF p.526 (printed 502) — 16 Text generation › 16.4 Going further with LLMs › 16.4.4 “Reasoning” models
For years since the first LLMs, researchers have struggled with the well-known fact that these models were abysmal at math problems and logic puzzles. A model might give a perfect response to a problem directly in its training data, but substitute a few names or numbers in the prompt, and it would become evident that the model had no grasp on what it was trying to solve. For many problems in natural language processing, LLMs gave an easy recipe for progress: increase the amount of training data, increase some benchmark score. Grade school math problems, however, defied progress.

In 2023, researchers from Google noticed that if you prompted the model with a few examples of “showing your work” on a math problem—as in literally writing out the steps like you would on a homework assignment—the model would start to do the same. As the model mimicked writing out intermediate steps, it would actually do far better at reaching the correct solution by attending to its own output. They called this “chain-of-thought” prompting, and the name stuck. Another group of researchers noticed that

### PDF p.529 (printed 505) — 16 Text generation › 16.5 Where are LLMs heading next? › 
better model performance. So recently, model sizes have trended flatter while data sizes have trended up.

This doesn’t mean that scaling will stop—more computing power does generally lead to better LLM performance, and we have yet to see signs of an asymptote where next token prediction performance levels off. Companies are continuing to invest billions of dollars in scaling LLMs and seeing what new capabilities emerge. Figure 16.8 shows details for some of the major LLMs released from 2018 to 2025.

Figure 16.8 LLM parameter counts (top) and pretraining dataset sizes (bottom) over time. Many recent proprietary LLMs (e.g., GPT-4 and Gemini) are not included because model details have not been disclosed.
