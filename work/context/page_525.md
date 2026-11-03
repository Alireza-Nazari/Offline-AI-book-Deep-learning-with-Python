# Generation bundle for page_525  (PDF page 525, printed page 501)

Page kind: content
Chapter: 16 Text generation
Section: 16.4 Going further with LLMs
Subsection: 16.4.3 Retrieval Augmented Generation (RAG)
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_525.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_525.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p525_e03] (heading_subsubsection) Foundation models
[p525_e04] (body) As LLMs venture into different modalities, the “large language model” moniker can become a bit misleading. They do model language, but also images, audio, maybe even structured data. In the next chapter, we will see a distinct architecture, called diffusion models, that works quite differently in terms of underlying structure but has a similar feel—they too are trained on massive amounts of data at “internet scale” with a self-supervised loss.
[p525_e05] (body) An umbrella term for models like this is foundation models. More specifically, a foundation model is any model that is trained on broad data (generally using self-supervision at scale) that can be fine-tuned to a wide range of downstream tasks. In general, you can think of a foundation model as learning to reconstruct data pulled from large swaths of the internet, given a partial representation of it. While LLMs are the first and best-known of these models, there are many others. The hallmarks of a foundation model are the self-supervised learning objective (a reconstruction loss) and the fact that these models are not specialized to a single task and can be used for a number of downstream purposes. This is an important and striking shift that has happened quite recently in the long history of machine learning. Rather than training a model from scratch on your individual dataset, you will often be better off using a foundation model to get a rich representation of your input (whether it’s images, text, or something else) and then specialize that model for your final downstream task. Of course, this comes with the downside of needing to run large models with billions of parameters, so it’s hardly a fit for all real-world applications of machine learning.
[p525_e06] (heading_subsection) 16.4.3 Retrieval Augmented Generation (RAG)
[p525_e07] (body) Sticking extra information in the prompt is not just helpful in handling image data; it can be a general way to extend the capabilities of an LLM. One notable example is when using an LLM for search. If we naively compare an LLM to a search engine, it has a couple of fatal flaws:
[p525_e08] (bullet) An LLM will occasionally make things up. It will output false “facts” that were not present in the training data but could be interpolated from the training data. This information can range from misleading to dangerous.
[p525_e09] (bullet) An LLM’s knowledge of the world has a cutoff date—at best, the date the model was pretrained. Training an LLM is quite expensive, and it is not feasible to train continuously on new data. So at some arbitrary point in time, an LLM’s knowledge of the world will just stop.
[p525_e10] (body) No one wants to use a search engine that can only tell you about things that happened six months ago. But if we think of an LLM as more like “conversational software” that can handle any sequence data in a prompt, what if we instead used the model as the interface to information retrieved by more traditional search? That’s the idea behind retrieval-augmented generation or RAG.

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

## Previous page (PDF 524) text, for continuity
We can use Gemma to ask some questions about this image:

Each of our input prompts contains the special token `<start_of_image>`. This is turned into 256 placeholder values in our input sequence, which, in turn, is replaced with the soft tokens representing our image.

Training for a multimodal model like this is quite similar to regular LLM pretraining and fine-tuning. Usually, you would want to first pretrain your image encoder separately, like we first did in Chapter 8 of this book. Then you can simply do the same basic “guess the next word” pretraining and also feed in mixed image and text content combined into a single sequence. Our transformer would not be trained to output image soft tokens; we would simply zero the loss at these image token locations. It might seem almost magical that we can simply add image data to an LLM, but when we consider the power of the sequence model we’re working with, it’s really quite an expected result. We’ve taken a Transformer, recast our image input as sequence data, and done a lot of extra training. The model can preserve the original language model’s ability to ingest and produce text while learning to also embed images in the Transformer’s latent space.

## Next page (PDF 526) text, for continuity
RAG works by taking an initial user question and doing some form of a query to pull in additional text context. This query can be to a database, a search engine, or anything that can give further information on the question asked by a user. This extra information is then added straight into the prompt. For example, you might construct a prompt like this:

Use the following pieces of context to answer the question. Question: What are some good ways to improve sleep? Context: {text from a medical journal on improving sleep} Answer:

A common approach for looking up relevant information is to use a vector database. To build a vector database, you can use an LLM, or any model, to embed a series of source documents as vectors. The document text will be stored in the database, with the embedding vector used as a key. During retrieval, an LLM can again be used to embed the user query as a vector. The vector database is responsible for searching for key vectors close to the query vector and for surfacing the corresponding text. This might sound a lot like the attention mechanism itself—recall that the terms “query,” “key,” and “value” actually came from database systems.

Surfacing information to assist with generation does a few things:

It gives you an obvious way to work around the cutoff date of the model.

It allows the model to access private data. Companies might want to use an LLM trained on public data to serve as an interface to information stored privately.

It can help factually ground the model. There is no silver bullet that stops hallucinations entirely, but an LLM is much less likely to make up facts on a topic if presented with correct context about the subject in a prompt.

16.4.4 “Reasoning” models

For years since the first LLMs, researchers have struggled w …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.530 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.589 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.508 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.531 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.519 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.528 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.592 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.443 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › Summary
Large language models, or LLMs, are the combination of a few key ingredients: – The Transformer architecture – A language modeling task (predicting the next token based on past tokens) – A large amount of unlabeled text data

An LLM learns a probability distribution for predicting individual tokens. This can be combined with a sampling strategy to generate a long string of text. There are many popular ways to sample text: – Greedy search takes the most likely predicted token at each generation step. – Random sampling directly samples the predicted categorical distribution over all tokens. – Top-K sampling restricts the categorical distribution to the top set of K candidates.

LLMs use billions of parameters and are trained on trillions of words of text.

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › 
We can note that while the total number of tokens used for pretraining has climbed steadily and massively, model parameter counts have varied substantially since GPT-3. In part, this is because we now know GPT-3 was undertrained, but it is also for a more practical reason. When deploying a model, it’s often worth it to sacrifice performance for a smaller model that fits on cheaper hardware. A really good model won’t help very much if it’s prohibitively expensive to run. There’s another reason we might not be able to just scale up these models thoughtlessly: we are starting to run out of pretraining data! Tech companies are starting to have trouble finding more high-quality, public, human-written content to throw at pretraining. Models are even starting to “eat their own tail” by training on a significant portion of content created by other LLMs, which runs into a whole other host of concerns. This is one of the reasons reinforcement learning is getting a lot of attention recently. If you can create a difficult, self-contained environment that generates new problems for an LLM to attempt, you will have found a way to continue training using the model’s own output—no need to scrounge

### PDF p.589 (printed 565) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.1 Deep learning models struggle to adapt to novelty
Despite frequent claims that LLMs can perform in-context learning to pick up new skills from just a few examples, there is overwhelming evidence that what they’re actually doing is fetching vector functions they’ve memorized during training and reapplying them to the task at hand. By learning to do next-token prediction across a web-sized text dataset, an LLM has collected millions of potentially useful mini text-processing programs, and it can easily be prompted into reusing them on a new problem. But show it something that has no direct equivalent in its training data, and it’s helpless. Take a look at the puzzle in figure 19.1. Did you figure out the solution? Good. It’s not very hard, is it? But today, no state-of-the-art LLM or vision-language model can do this because this particular problem doesn’t directly map to anything they’ve seen at training time—even after having been trained on the entire internet and then some. An LLM’s ability to solve a given problem has nothing to do with problem complexity, and everything to do with familiarity—they will break their teeth on any sufficiently novel problem, no matter how simple. This failure mode even applies to tiny variations o

### PDF p.14 (printed xiv) — contents ›  › 
14.5

Sequence models 405

Training a recurrent model 406 ■ Understanding word embeddings 409 ■ Using a word embedding 410 ■ Pretraining a word embedding 414 ■ Using the pretrained embedding for classification 418

15

Language models and the Transformer 421 15.1

The language model 421

Training a Shakespeare language model 422 ■ Generating Shakespeare 426

15.2

Sequence-to-sequence learning 428

English-to-Spanish translation 430 ■ Sequence-to-sequence learning with RNNs 432

15.3

The Transformer architecture 437

Dot-product attention 439 ■ Transformer encoder block 444 Transformer decoder block 446 ■ Sequence-to-sequence learning with a Transformer 448 ■ Embedding positional information 451

15.4

Classification with a pretrained Transformer 454

Pretraining a Transformer encoder 454 ■ Loading a pretrained Transformer 455 ■ Preprocessing IMDb movie reviews 458 Fine-tuning a pretrained Transformer 460

15.5

What makes the Transformer effective? 461 16 Text generation 466 16.1 A brief history of sequence generation 468 16.2 Training a mini-GPT 470

Building the model 473 ■ Pretraining the model 476 Generative decoding 478 ■ Sampling strategies 480

16.3

Using a pretrained LLM

### PDF p.519 (printed 495) — 16 Text generation › 16.4 Going further with LLMs › 16.4 Going further with LLMs
We have now trained a GPT model from scratch and fine-tuned a language model into our very own chatbot. However, we are just scratching the surface of LLM research today. In this section, we will cover a non-exhaustive list of extensions and improvements to the basic “autocomplete the internet” language modeling setup.

### PDF p.643 (printed 619) — index ›  › 
stemming, defined 386–387

TensorFlow Serving 185 step fusing 556 tensor operations, derivatives of 42–43 stochastic gradient descent 43–46 tensors, defined 63 strided convolutions 239 test images, preparing 321 subclassing Model class 202 test-time training 582 rewriting previous example as subclassed text classification 381 model 202

augmenting text data 413 what subclassed models don't support 204 NLP (natural language processing) 382–384 subgraph isomorphism 586 preparing text data 384–394 subword tokenization 390–394 pretraining word embeddings 414 summary() method 193 sequence models 405 supervised sets vs. sequences 395–398 fine-tuning 495 setting models 398–404 learning 214 training recurrent model 406 symbolic AI 3 using pretrained embedding for symbolic tensor 196

classification 418 symbols 584–594 word embeddings 409, 410 blending deep learning and program text-conditioning 509 synthesis 590–592 text_dataset_from_directory utility 397 cognition as combination of both kinds of abstraction 587 text_embeddings input 531 deep learning and 588 text generation 466 long-term vision 594 history of sequence generation 468–470 modular component recombination and lifelong LLMs (l

### PDF p.508 (printed 484) — 16 Text generation › 16.3 Using a pretrained LLM › 
A sampling strategy is an important control when generating text, and there are many more approaches. For example, beam search is a technique that heuristically explores multiple chains of predicted tokens by keeping a fixed number of “beams” (different chains of predicted tokens) to explore at each timestep.

With top-K sampling, our model generates something closer to plausible English text, but there is little apparent utility to such output. This fits with the results of GPT-1. For the initial GPT paper, the generated output was more of a curiosity, and state-of-the-art results were only achieved by fine-tuning classification models. Our mini-GPT is far less trained than GPT-1. To reach the scale of generative LLMs today, we’d need to increase our parameter count by at least 100× and our train step count by at least 1,000×. If we did, we would see the same leaps in quality observed by OpenAI with GPT. And we could do it! The training recipe we used previously is the exact blueprint used by everyone training LLMs today. The only missing pieces are a very large compute budget and some tricks for training across multiple machines that we will cover in chapter 18. For a more practi
