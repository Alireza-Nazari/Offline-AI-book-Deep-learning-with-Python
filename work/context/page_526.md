# Generation bundle for page_526  (PDF page 526, printed page 502)

Page kind: content
Chapter: 16 Text generation
Section: 16.4 Going further with LLMs
Subsection: 16.4.4 “Reasoning” models
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_526.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_526.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p526_e03] (body) RAG works by taking an initial user question and doing some form of a query to pull in additional text context. This query can be to a database, a search engine, or anything that can give further information on the question asked by a user. This extra information is then added straight into the prompt. For example, you might construct a prompt like this:
[p526_e04] (sidebar) Use the following pieces of context to answer the question. Question: What are some good ways to improve sleep? Context: {text from a medical journal on improving sleep} Answer:
[p526_e05] (body) A common approach for looking up relevant information is to use a vector database. To build a vector database, you can use an LLM, or any model, to embed a series of source documents as vectors. The document text will be stored in the database, with the embedding vector used as a key. During retrieval, an LLM can again be used to embed the user query as a vector. The vector database is responsible for searching for key vectors close to the query vector and for surfacing the corresponding text. This might sound a lot like the attention mechanism itself—recall that the terms “query,” “key,” and “value” actually came from database systems.
[p526_e06] (body) Surfacing information to assist with generation does a few things:
[p526_e07] (bullet) It gives you an obvious way to work around the cutoff date of the model.
[p526_e08] (bullet) It allows the model to access private data. Companies might want to use an LLM trained on public data to serve as an interface to information stored privately.
[p526_e09] (bullet) It can help factually ground the model. There is no silver bullet that stops hallucinations entirely, but an LLM is much less likely to make up facts on a topic if presented with correct context about the subject in a prompt.
[p526_e10] (heading_subsection) 16.4.4 “Reasoning” models
[p526_e11] (body) For years since the first LLMs, researchers have struggled with the well-known fact that these models were abysmal at math problems and logic puzzles. A model might give a perfect response to a problem directly in its training data, but substitute a few names or numbers in the prompt, and it would become evident that the model had no grasp on what it was trying to solve. For many problems in natural language processing, LLMs gave an easy recipe for progress: increase the amount of training data, increase some benchmark score. Grade school math problems, however, defied progress.
[p526_e12] (body) In 2023, researchers from Google noticed that if you prompted the model with a few examples of “showing your work” on a math problem—as in literally writing out the steps like you would on a homework assignment—the model would start to do the same. As the model mimicked writing out intermediate steps, it would actually do far better at reaching the correct solution by attending to its own output. They called this “chain-of-thought” prompting, and the name stuck. Another group of researchers noticed that

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

## Previous page (PDF 525) text, for continuity
Foundation models

As LLMs venture into different modalities, the “large language model” moniker can become a bit misleading. They do model language, but also images, audio, maybe even structured data. In the next chapter, we will see a distinct architecture, called diffusion models, that works quite differently in terms of underlying structure but has a similar feel—they too are trained on massive amounts of data at “internet scale” with a self-supervised loss.

An umbrella term for models like this is foundation models. More specifically, a foundation model is any model that is trained on broad data (generally using self-supervision at scale) that can be fine-tuned to a wide range of downstream tasks. In general, you can think of a foundation model as learning to reconstruct data pulled from large swaths of the internet, given a partial representation of it. While LLMs are the first and best-known of these models, there are many others. The hallmarks of a foundation model are the self-supervised learning objective (a reconstruction loss) and the fact that these models are not specialized to a single task and can be used for a number of downstream purposes. This is an important and striking shift that has happened quite recently in the long history of machine learning. Rather than training a model from scratch on your individual dataset, you will often be better off using a foundation model to get a rich representation of your input (whether it’s images, text, or something else) and then specialize that model for your final downstream task. Of course, this comes with the downside of needing to run large models with billions of parameters, so it’s hardly a fit for all real-world applications of machine learning.

16.4.3 Retrieval Augmented Generation (RAG)

Sticking ext …

## Next page (PDF 527) text, for continuity
you didn’t even need examples; you could simply prompt the model with the phrase “Let’s think step by step” and get better output.

Since these discoveries, there has been heavy interest in directly training LLMs to get better at chain-of-thought reasoning. Models like OpenAI’s o1 and DeepSeek’s r1 have made headlines by showing significant strides in math and coding problems by training a model to “think out loud” on difficult questions. The approach for this chain-of-thought fine-tuning is very similar to RLHF. We will first train the model on a few supervised examples of “showing your work” on a math problem and arriving at a correct answer. Next, we will prompt the model with a new math question and check whether the model got the final answer correct. Finally, we use these new generated outputs to further tune the model’s weights. Let’s try this out with the Gemma model. We can write out our own word problem and turn on random sampling so we get a somewhat random response each time:

Let’s try generating a couple of responses:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.528 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.592 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.589 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.530 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.511 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.512 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.590 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.521 — 16 Text generation › 16.4 Going further with LLMs

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.528 (printed 504) — 16 Text generation › 16.5 Where are LLMs heading next? › 
In the first attempt, our model was hung up on the superfluous detail that each letter has two pages. In the second attempt, the model gets the problem right. This instruction-tuned Gemma model we are working with has already been tuned on math problems like this; you would not get nearly as good results from the “untuned” Gemma model from the last section.

We could extend this idea to a very simple form of chain-of-thought training: 1 Collect a bunch of basic math and reasoning problems and desired answers. 2 Generate (with some randomness) a number of responses. 3 Find all the responses with a correct answer via string parsing. You can prompt

the model to use a specific text marker for the final answer as we did previously. 4 Run supervised fine-tuning on correct responses, including all the intermediate output. 5 Repeat! The previously described process is a reinforcement learning algorithm. Our answer checking acts as the environment, and the generated outputs are the actions the model uses to learn. As with RLHF, in practice you would use a more complex gradient update step to use information from all responses (even the incorrect ones), but the basic principle is the same. 

### PDF p.589 (printed 565) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.1 Deep learning models struggle to adapt to novelty
Despite frequent claims that LLMs can perform in-context learning to pick up new skills from just a few examples, there is overwhelming evidence that what they’re actually doing is fetching vector functions they’ve memorized during training and reapplying them to the task at hand. By learning to do next-token prediction across a web-sized text dataset, an LLM has collected millions of potentially useful mini text-processing programs, and it can easily be prompted into reusing them on a new problem. But show it something that has no direct equivalent in its training data, and it’s helpless. Take a look at the puzzle in figure 19.1. Did you figure out the solution? Good. It’s not very hard, is it? But today, no state-of-the-art LLM or vision-language model can do this because this particular problem doesn’t directly map to anything they’ve seen at training time—even after having been trained on the entire internet and then some. An LLM’s ability to solve a given problem has nothing to do with problem complexity, and everything to do with familiarity—they will break their teeth on any sufficiently novel problem, no matter how simple. This failure mode even applies to tiny variations o

### PDF p.465 (printed 441) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
We can make the hypothesis space of this attention mechanism much richer if we give the model parameters to control the attention score. If we project both source and target vectors with `Dense` layers, the model can find a good shared space where source vectors are close to target vectors if they help the overall prediction quality. Similarly, we should allow the model to project the source vectors into an entirely separate space before they are combined and once again after the summation.

Figure 15.5 When both target and source are

We can also adopt a slightly dif

sequences, attention scores are a 2D matrix. Each

ferent naming for inputs that has

row shows the attention scores for the word we are

become standard in the field. What

trying to predict (in green).

we just wrote is roughly summarized as `sum(score(target,` `source)` `* source)`. We will write this equivalently with different input names as `sum(score(query, key) * value)`. This three-argument version is more general—in rare cases, you might not want to use the same vector to score your source inputs as you use to sum your source inputs.

The terminology comes from search engines and recommender systems. Imagin

### PDF p.466 (printed 442) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Figure 15.6 Retrieving images from a database: the query is compared to a set of keys, and the match scores are used to rank values (images).

This block is a perfectly functional attention mechanism! We just wrote a function that will allow the model to pull information from anywhere in the source sequence, contextually, depending on the target word we are decoding.

The “Attention is all you need” authors made two more changes to our mechanism through trial and error. The first is a simple scaling factor. When input vectors get long, the dot-product scores can get quite large, which can affect the stability of our softmax gradients. The fix is simple: we can scale down our softmax scores slightly. Scaling by the square root of the vector length works well for any vector size. The other has to do with the expressivity of the attention mechanism. The softmax sum we are doing is powerful—it allows a direct connection across distant parts of a sequence. But the summation is also blunt: if the model tries to attend to too many

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › 
We can note that while the total number of tokens used for pretraining has climbed steadily and massively, model parameter counts have varied substantially since GPT-3. In part, this is because we now know GPT-3 was undertrained, but it is also for a more practical reason. When deploying a model, it’s often worth it to sacrifice performance for a smaller model that fits on cheaper hardware. A really good model won’t help very much if it’s prohibitively expensive to run. There’s another reason we might not be able to just scale up these models thoughtlessly: we are starting to run out of pretraining data! Tech companies are starting to have trouble finding more high-quality, public, human-written content to throw at pretraining. Models are even starting to “eat their own tail” by training on a significant portion of content created by other LLMs, which runs into a whole other host of concerns. This is one of the reasons reinforcement learning is getting a lot of attention recently. If you can create a difficult, self-contained environment that generates new problems for an LLM to attempt, you will have found a way to continue training using the model’s own output—no need to scrounge

### PDF p.590 (printed 566) — 19 The future of AI › 19.1 The limitations of deep learning › 
Figure 19.1 An easy yet novel puzzle

model would just repeat the answer it had memorized without paying any attention to the actual numbers in the query, or what the query really meant. Similarly, LLMs struggle to adapt to variations of the Monty Hall problem (see figure 19.2) and will tend to always output the canonical answer to the puzzle, which they’ve seen many times during training, regardless of whether it makes sense in context.

Figure 19.2 A variation of the Monty Hall problem

To note, these specific prompts were patched later on by special-casing them. Today, there are over 25,000 people who are employed full time to provide training data for LLMs by reviewing failure cases and suggesting better answers. LLM maintenance is a constant game of whack-a-mole where failing prompts are patched one at a time,

### PDF p.589 (printed 565) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.1 Deep learning models struggle to adapt to novelty
First and foremost, a parametric curve is only capable of information storage—it’s a kind of database. Recall our discussion of Transformers as an “interpolative database” from chapter 15? Second, crucially, this database is static. The model’s parameters are determined during a distinct “training time” phase. Afterward, these parameters are frozen, and this fixed version is used during “inference time” for making predictions on new data. The only thing you can do with a static database is information retrieval. And that’s exactly what deep learning models excel at: recognizing or generating patterns highly similar to those encountered during training. The flip side is that they are inherently poor at adaptation. The database is backward-looking—it fits past data but can’t handle a changing future. At inference time, you’d better hope that the situations the model faces are part of the training data distribution, because otherwise, the model will break down. A model trained on ImageNet will classify a leopard-print sofa as an actual leopard, for instance—sofas were not part of its training data. This also applies to the largest of generative models. In recent years, the rise of lar

### PDF p.511 (printed 487) — 16 Text generation › 16.3 Using a pretrained LLM › 
Though it’s tempting when working with a model that can “talk” to imagine it interpreting our prompt in some sort of human, conversational way, nothing of the sort is going on here. We have just constructed a prompt for which an actual brownie recipe is a more likely continuation than mimicking someone posting on a forum asking for baking help.

You can go much further in constructing prompts. You might prompt a model with some natural language instructions of the role it is supposed to fill, for example, `"You` `are a large language model that gives short, helpful answers to people's questions."` Or you might feed the model a prompt containing a long list of harmful topics that should not be included in any generated responses. If this all sounds a bit hand-wavy and hard to control, that’s a good assessment. Attempting to visit different parts of a model’s distribution through prompting is often useful, but predicting how a model will respond to a given prompt is very difficult. Another well-documented problem faced by LLMs is hallucinations. A model will always say something—there is always a most-likely next token to a given sequence. Finding locations in our LLM distribution th
