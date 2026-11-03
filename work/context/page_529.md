# Generation bundle for page_529  (PDF page 529, printed page 505)

Page kind: content
Chapter: 16 Text generation
Section: 16.5 Where are LLMs heading next?
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_529.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_529.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p529_e03] (body) better model performance. So recently, model sizes have trended flatter while data sizes have trended up.
[p529_e04] (body) This doesn’t mean that scaling will stop—more computing power does generally lead to better LLM performance, and we have yet to see signs of an asymptote where next token prediction performance levels off. Companies are continuing to invest billions of dollars in scaling LLMs and seeing what new capabilities emerge. Figure 16.8 shows details for some of the major LLMs released from 2018 to 2025.
[p529_e05] (figure_caption) Figure 16.8 LLM parameter counts (top) and pretraining dataset sizes (bottom) over time. Many recent proprietary LLMs (e.g., GPT-4 and Gemini) are not included because model details have not been disclosed.

## Figures
- Figure 16.8: LLM parameter counts (top) and pretraining dataset sizes (bottom) over time. Many recent proprietary LLMs (e.g., GPT-4 and Gemini) are not included because model details have not been disclosed.  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 16 Text generation (PDF pages 490-531)
  - 16.1 A brief history of sequence generation (p.492)
  - 16.2 Training a mini-GPT (p.494)
      - 16.2.1 Building the model (p.497)
      - 16.2.2 Pretraining the model (p.500)
      - 16.2.3 Generative decoding (p.502)
      - 16.2.4 Sampling strategies (p.504)
  - 16.3 Using a pretrained LLM (p.508)
      - 16.3.1 Text generation with the Gemma model (p.509)
      - 16.3.2 Instruction fine-tuning (p.512)
      - 16.3.3 Low-Rank Adaptation (LoRA) (p.514)
  - 16.4 Going further with LLMs (p.519)
      - 16.4.1 Reinforcement Learning with Human Feedback (RLHF) (p.519)
      - 16.4.2 Multimodal LLMs (p.522)
      - 16.4.3 Retrieval Augmented Generation (RAG) (p.525)
      - 16.4.4 “Reasoning” models (p.526)
  - 16.5 Where are LLMs heading next? (p.528)

## Previous page (PDF 528) text, for continuity
In the first attempt, our model was hung up on the superfluous detail that each letter has two pages. In the second attempt, the model gets the problem right. This instruction-tuned Gemma model we are working with has already been tuned on math problems like this; you would not get nearly as good results from the “untuned” Gemma model from the last section.

We could extend this idea to a very simple form of chain-of-thought training: 1 Collect a bunch of basic math and reasoning problems and desired answers. 2 Generate (with some randomness) a number of responses. 3 Find all the responses with a correct answer via string parsing. You can prompt

the model to use a specific text marker for the final answer as we did previously. 4 Run supervised fine-tuning on correct responses, including all the intermediate output. 5 Repeat! The previously described process is a reinforcement learning algorithm. Our answer checking acts as the environment, and the generated outputs are the actions the model uses to learn. As with RLHF, in practice you would use a more complex gradient update step to use information from all responses (even the incorrect ones), but the basic principle is the same. The same idea is being used to improve LLM performance in other domains that have obvious, verifiable answers to text prompts. Coding is an important one—you can prompt the LLM to output code and then actually run the code to test the quality of the response. In all these domains, one trend is clear—as a model learns to solve more difficult questions, the model will spend more and more time “showing its work” before reaching a final answer. You can think of this as the model learning to search over its own output of potential solutions. We will discuss this idea further in the final chapter of …

## Next page (PDF 530) text, for continuity
We can note that while the total number of tokens used for pretraining has climbed steadily and massively, model parameter counts have varied substantially since GPT-3. In part, this is because we now know GPT-3 was undertrained, but it is also for a more practical reason. When deploying a model, it’s often worth it to sacrifice performance for a smaller model that fits on cheaper hardware. A really good model won’t help very much if it’s prohibitively expensive to run. There’s another reason we might not be able to just scale up these models thoughtlessly: we are starting to run out of pretraining data! Tech companies are starting to have trouble finding more high-quality, public, human-written content to throw at pretraining. Models are even starting to “eat their own tail” by training on a significant portion of content created by other LLMs, which runs into a whole other host of concerns. This is one of the reasons reinforcement learning is getting a lot of attention recently. If you can create a difficult, self-contained environment that generates new problems for an LLM to attempt, you will have found a way to continue training using the model’s own output—no need to scrounge the web for more morsels of quality text. None of the solutions we touched on will be a silver bullet for the issues facing LLMs. At the end of the day, the fundamental problem remains that LLMs are wildly inefficient at learning compared to humans. Model capabilities only come from training on many orders of magnitude more text than people will read in their lifetimes. As scaling LLMs continues, so too will more fundamental research in how to make models that can learn quickly with limited data. Still, LLMs represent the ability to build fluent natural language interfaces, and that alone wil …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.589 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.606 — 19 The future of AI › 19.3 How to build intelligence
- PDF p.508 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.525 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.517 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.629 — 20 Conclusions › 20.2 Limitations of deep learning
- PDF p.425 — 14 Text classification › 14.4 Set models
- PDF p.595 — 19 The future of AI › 19.2 Scale isn’t all you need

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.595 (printed 571) — 19 The future of AI › 19.2 Scale isn’t all you need › 
been released, and it was essentially a scaled-up version of GPT-3: more parameters, more training data. Its significantly improved performance seemed to suggest that you could just keep going—that there could be a GPT-5 that would simply be more of the same and from which artificial general intelligence (AGI) would spontaneously emerge.

Proponents of this view would point to “scaling laws” as evidence. Scaling laws are an empirical relationship observed between the size of a deep learning model (as well as the size of its training dataset) and its performance on specific tasks. They suggest that increasing the size of a model reliably leads to better performance in a predictable manner. But the key thing that scaling law enthusiasts are missing is that the benchmarks they’re using to measure “performance” are effectively memorization tests, the kind we like to give university students. LLMs perform well on these tests by memorizing the answers, and naturally, cramming more questions and more answers into the models improves their performance accordingly. The reality is that scaling up our models hasn’t led to any progress on the issues I’ve listed so far in these pages—inability 

### PDF p.606 (printed 582) — 19 The future of AI › 19.3 How to build intelligence › 
The ARC-AGI benchmark has proven remarkably resistant to the prevailing deep learning scaling paradigm. Most other benchmarks have saturated quickly in the age of LLMs. That’s because they can be hacked via memorization, whereas ARC-AGI is designed to be resistant to it. From 2019, when ARC-AGI was first released, to 2025, base LLMs underwent a roughly 50,000× scale-up—from, say, GPT-2 (2019) to GPT-4.5 (2025), but their performance on the 2019 version of ARC-AGI only went from 0% to around 10%. Given that you, reader, would easily score above 95%, that isn’t very good. If you scale up your system by 50,000× and you’re still not making meaningful progress, that’s like a big warning sign telling you that you need to try new ideas. Simply making models bigger or training them on more data has not unlocked the kind of fluid intelligence that ARC-AGI requires. ARC-AGI was clearly showing that on-the-fly recombination capabilities are necessary to tackle reasoning.

### PDF p.589 (printed 565) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.1 Deep learning models struggle to adapt to novelty
Despite frequent claims that LLMs can perform in-context learning to pick up new skills from just a few examples, there is overwhelming evidence that what they’re actually doing is fetching vector functions they’ve memorized during training and reapplying them to the task at hand. By learning to do next-token prediction across a web-sized text dataset, an LLM has collected millions of potentially useful mini text-processing programs, and it can easily be prompted into reusing them on a new problem. But show it something that has no direct equivalent in its training data, and it’s helpless. Take a look at the puzzle in figure 19.1. Did you figure out the solution? Good. It’s not very hard, is it? But today, no state-of-the-art LLM or vision-language model can do this because this particular problem doesn’t directly map to anything they’ve seen at training time—even after having been trained on the entire internet and then some. An LLM’s ability to solve a given problem has nothing to do with problem complexity, and everything to do with familiarity—they will break their teeth on any sufficiently novel problem, no matter how simple. This failure mode even applies to tiny variations o

### PDF p.590 (printed 566) — 19 The future of AI › 19.1 The limitations of deep learning › 
Figure 19.1 An easy yet novel puzzle

model would just repeat the answer it had memorized without paying any attention to the actual numbers in the query, or what the query really meant. Similarly, LLMs struggle to adapt to variations of the Monty Hall problem (see figure 19.2) and will tend to always output the canonical answer to the puzzle, which they’ve seen many times during training, regardless of whether it makes sense in context.

Figure 19.2 A variation of the Monty Hall problem

To note, these specific prompts were patched later on by special-casing them. Today, there are over 25,000 people who are employed full time to provide training data for LLMs by reviewing failure cases and suggesting better answers. LLM maintenance is a constant game of whack-a-mole where failing prompts are patched one at a time,

### PDF p.508 (printed 484) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3 Using a pretrained LLM
Now that we’ve trained a mini-language model from scratch, let’s try using a billion-parameter pretrained model and see what it can do. Given how prohibitively expensive pretraining a Transformer can be, most of the industry has centered around using pretrained models developed by a relatively short list of companies. This is not purely a cost concern but also an environmental one—generative model training is now making up a large percentage of the total data center power consumption of large tech companies.

Meta published some environmental data on Llama 2, an LLM it published in 2023. It’s a good bit smaller than GPT-3, but it needed an estimated 1.3 million kilowatt hours of electricity to train—the daily power usage of about 45,000 American households. If every organization using an LLM ran pretraining themselves, the scale of energy use would be a noticeable percentage of global energy consumption. Let’s play around with a pretrained generative model from Google called Gemma. We will use the third version of the Gemma model, which was released to the public in 2025. To keep the examples in this book accessible, we will use the smallest variation of

### PDF p.531 (printed 507) — 16 Text generation › 16.5 Where are LLMs heading next? › 
LLM output is unreliable, and all LLMs will occasionally hallucinate factually incorrect information.

LLMs can be fine-tuned to follow instructions in a chat dialog. This type of

fine-tuning is called instruction fine-tuning:

– The simplest form of instruction fine-tuning involves directly training the

model on instruction and response pairs. – More advanced forms of instruction fine-tuning involve reinforcement learning.

The most common resource bottleneck when working with LLMs is accelerator memory.

LoRA is a technique to reduce memory usage by freezing most Transformer parameters and only updating a low-rank decomposition of attention projection weights.

LLMs can input or output data from different modalities if you can figure out how to frame these inputs or outputs as sequences in a sequence prediction problem.

A foundation model is a general term for models of any modality trained using self-supervision for a wide range of downstream tasks.

### PDF p.630 (printed 606) — 20 Conclusions › 20.3 What might lie ahead › 
benchmarks that amount to memorization tests, it fails to address the fundamental limitations of deep learning, which stem from the core paradigm of fitting static, interpolative curves to data. Five years of exponential scaling of base LLMs haven’t overcome these constraints because the underlying approach remains unchanged.

By 2024, this realization spurred a transition toward test-time adaptation (TTA), where models perform search or fine-tuning during the inference phase to adapt to novel problems. While TTA methods have yielded major breakthroughs, such as OpenAI’s o3 surpassing human baseline on ARC-AGI-1 in late 2024, this performance has come at an extreme computational cost. Efficient, human-like adaptation is still a completely open problem, and the slightly harder ARC-AGI-2 benchmark remains completely unsolved as of today. We still need further conceptual advances beyond mere scaling or brute-force search.

### PDF p.629 (printed 605) — 20 Conclusions › 20.2 Limitations of deep learning › 20.2 Limitations of deep learning
You should always resist the temptation to anthropomorphize deep learning models. Their performance is built on pointwise statistical patterns rather than human-like experiential grounding, making it brittle when encountering deviations from training data.

The narrative that simply scaling up model size and training data would lead to general intelligence has proven insufficient. While scaling enhances performance on
