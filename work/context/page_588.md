# Generation bundle for page_588  (PDF page 588, printed page 564)

Page kind: chapter_opener
Chapter: 19 The future of AI
Section: 19.1 The limitations of deep learning
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_588.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_588.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p588_e01] (chapter_number) 19
[p588_e02] (chapter_title) The future of AI
[p588_e03] (chapter_covers) This chapter covers
[p588_e04] (bullet) The limitations of deep learning
[p588_e05] (bullet) The nature of intelligence
[p588_e06] (bullet) What’s missing from current approaches
[p588_e07] (bullet) What the future might look like
[p588_e08] (body) To use a tool appropriately, you should not only understand what it can do but also be aware of what it can’t do. I’m going to present an overview of some key limitations of deep learning. Then, I’ll offer some speculative thoughts about the future evolution of AI and what it would take to get to human-level general intelligence. This should be especially interesting to you if you’d like to get into fundamental research.
[p588_e09] (heading_section) 19.1 The limitations of deep learning
[p588_e10] (body) There are infinitely many things you can do with deep learning. But deep learning can’t do everything. To use a tool well, you should be aware of its limitations, not just its strengths. So where does deep learning fall short?

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

## Previous page (PDF 587) text, for continuity
Summary

You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.

## Next page (PDF 589) text, for continuity
19.1.1 Deep learning models struggle to adapt to novelty

Deep learning models are big, parametric curves fitted to large datasets. That’s the source of their power—they’re easy to train, and they scale really well, both in terms of model size and dataset size. But that’s also a source of significant weaknesses. Curve fitting has inherent limitations.

First and foremost, a parametric curve is only capable of information storage—it’s a kind of database. Recall our discussion of Transformers as an “interpolative database” from chapter 15? Second, crucially, this database is static. The model’s parameters are determined during a distinct “training time” phase. Afterward, these parameters are frozen, and this fixed version is used during “inference time” for making predictions on new data. The only thing you can do with a static database is information retrieval. And that’s exactly what deep learning models excel at: recognizing or generating patterns highly similar to those encountered during training. The flip side is that they are inherently poor at adaptation. The database is backward-looking—it fits past data but can’t handle a changing future. At inference time, you’d better hope that the situations the model faces are part of the training data distribution, because otherwise, the model will break down. A model trained on ImageNet will classify a leopard-print sofa as an actual leopard, for instance—sofas were not part of its training data. This also applies to the largest of generative models. In recent years, the rise of large language models (LLMs) and their application to programming assistance and reasoning-like problems has provided extensive empirical proof of this. Despite frequent claims that LLMs can perform in-context learning to pick up new skills from ju …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.26 — 1 What is deep learning? › 1.2 Artificial intelligence
- PDF p.594 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.595 — 19 The future of AI › 19.2 Scale isn’t all you need
- PDF p.621 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.620 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.25 — 1 What is deep learning? › 
- PDF p.600 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.630 — 20 Conclusions › 20.3 What might lie ahead

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.629 (printed 605) — 20 Conclusions › 20.2 Limitations of deep learning › 20.2 Limitations of deep learning
You should always resist the temptation to anthropomorphize deep learning models. Their performance is built on pointwise statistical patterns rather than human-like experiential grounding, making it brittle when encountering deviations from training data.

The narrative that simply scaling up model size and training data would lead to general intelligence has proven insufficient. While scaling enhances performance on

### PDF p.25 (printed 1) — 1 What is deep learning? ›  › This chapter covers
High-level definitions of fundamental concepts

A soft introduction to the principles behind

machine learning

Deep learning’s rising popularity and future

potential

Over the past decade, artificial intelligence (AI) has been a subject of intense media hype. Machine learning, deep learning, and AI come up in countless articles, often outside of technology-minded publications. We’re promised a future of intelligent chatbots, self-driving cars, and virtual assistants—a future sometimes painted in a grim light and other times as utopian, where human jobs will be scarce and most economic activity will be handled by robots or AI agents. For a practitioner of machine learning, it’s important to be able to recognize the signal amid the noise, so that you can tell world-changing developments from overhyped press releases. Our future is at stake, and it’s one in which you have an active role to play: after reading this book, you’ll be one of those who can develop these AI systems. So let’s

### PDF p.621 (printed 597) — 20 Conclusions › 20.1 Key concepts in review › 
to extract from perceptual data almost anything a human could. Hence, it’s sometimes said that deep learning has “solved perception”—although that’s true only for a fairly narrow definition of perception.

Due to its unprecedented technical successes, deep learning has singlehandedly brought about the third and by far the largest AI summer: a period of intense interest, investment, and hype in the field of AI. As this book is being written, we’re in the middle of it. Whether this period will end in the near future and what happens after it ends are topics of debate. One thing is certain: in stark contrast with previous AI summers, deep learning has provided enormous business value to both large and small technology companies and has become a huge consumer success, enabling human-level speech recognition, chatbot assistants, photorealistic image generation, human-level machine translation, and more. The hype may (and likely will) recede, but the sustained economic and technological impact of deep learning will remain. In that sense, deep learning could be analogous to the internet: it may be overly hyped up for a few years, but in the longer term, it will still be a major revolution

### PDF p.594 (printed 570) — 19 The future of AI › 19.2 Scale isn’t all you need › 19.2 Scale isn’t all you need
Could we just keep scaling our models to overcome the limitations of deep learning? Is scale all we need? This has long been the prevailing narrative in the field, one that was especially prominent in early 2023, during peak LLM hype. Back then, GPT-4 had just

### PDF p.595 (printed 571) — 19 The future of AI › 19.2 Scale isn’t all you need › 
Deep learning models are limited in what they can represent, and most of the programs you may wish to learn can’t be expressed as a continuous geometric morphing of a data manifold. This is true in particular of algorithmic reasoning tasks.

Let’s take a closer look at what separates biological intelligence from the deep learning approach.

### PDF p.8 (printed viii) — contents ›  › contents
preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

### PDF p.600 (printed 576) — 19 The future of AI › 19.3 How to build intelligence › 
However, the deep learning paradigm has remained limited to cognitive automation: The “intelligence” label in “artificial intelligence” has been a category error. It would be more accurate to call our field “artificial cognition,” with “cognitive automation” and “artificial intelligence” being two nearly independent subfields within it. In this subdivision, AI would be a greenfield where almost everything remains to be discovered. Now, I don’t mean to diminish the achievements of deep learning. Cognitive automation is incredibly useful, and the way deep learning models are capable of automating tasks from exposure to data alone represents an especially powerful form of cognitive automation, far more practical and versatile than explicit programming. Doing this well is a game changer for essentially every industry. But it’s still a long way from human (or animal) intelligence. Our models, so far, can only perform local generalization: they map space X to space Y via a smooth geometric transform learned from a dense sampling of X-to-Y data points, and any disruption within spaces X or Y invalidates this mapping. They can only generalize to new situations that stay similar to past dat

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 20.3 What might lie ahead
Solving human-like fluid intelligence (and ARC-AGI-2) requires moving beyond the limitations inherent in current approaches. While deep learning excels at value-centric abstraction, which enables pattern recognition and intuition, it fundamentally lacks capabilities for program-centric abstraction, which underpins discrete reasoning, planning, and causal understanding. Human intelligence seamlessly integrates both— future AI must do the same.

Future key developments may include

Hybrid models—Future models will likely integrate learned algorithmic modules (providing reasoning and symbolic manipulation) with deep learning modules (providing perception and intuition). These systems might learn to use programming primitives like control flow, variables, recursion, and complex data structures dynamically.

Deep-learning guided program search—Program synthesis—automatically discovering executable code that meets specifications—offers a route to program-centric abstraction. However, its reliance on inefficient discrete search is a major bottleneck. A crucial advance will be using deep learning to guide this search, utilizing learned intuition about program structure to navigate vast, co
