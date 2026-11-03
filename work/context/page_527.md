# Generation bundle for page_527  (PDF page 527, printed page 503)

Page kind: content
Chapter: 16 Text generation
Section: 16.4 Going further with LLMs
Subsection: 16.4.4 “Reasoning” models
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_527.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_527.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p527_e03] (body) you didn’t even need examples; you could simply prompt the model with the phrase “Let’s think step by step” and get better output.
[p527_e04] (body) Since these discoveries, there has been heavy interest in directly training LLMs to get better at chain-of-thought reasoning. Models like OpenAI’s o1 and DeepSeek’s r1 have made headlines by showing significant strides in math and coding problems by training a model to “think out loud” on difficult questions. The approach for this chain-of-thought fine-tuning is very similar to RLHF. We will first train the model on a few supervised examples of “showing your work” on a math problem and arriving at a correct answer. Next, we will prompt the model with a new math question and check whether the model got the final answer correct. Finally, we use these new generated outputs to further tune the model’s weights. Let’s try this out with the Gemma model. We can write out our own word problem and turn on random sampling so we get a somewhat random response each time:
[p527_e05] (code)
```
prompt = """Judy wrote a 2-page letter to 3 friends twice a week for 3 months.
How many letters did she write?
Be brief, and add "ANSWER:" before your final answer."""
gemma_lm.compile(sampler="random")
```
[p527_e06] (body) Let’s try generating a couple of responses:
[p527_e07] (code)
```
>>> gemma_lm.generate(PROMPT_TEMPLATE.format(prompt))
<start_of_turn>user
Judy wrote a 2-page letter to 3 friends twice a week for 3 months.
How many letters did she write?
Be brief, and add "ANSWER:" before your final answer.<end_of_turn>
<start_of_turn>model
Here's how to solve the problem:
* **Letters per week:** 3 friends * 2 letters/week = 6 letters/week
* **Letters per month:** 6 letters/week * 4 weeks/month = 24 letters/month
* **Letters in 3 months:** 24 letters/month * 3 months = 72 letters
* **Total letters:** 72 letters * 2 = 144 letters
ANSWER: 144<end_of_turn>
>>> gemma_lm.generate(PROMPT_TEMPLATE.format(prompt))
<start_of_turn>user
Judy wrote a 2-page letter to 3 friends twice a week for 3 months.
How many letters did she write?
Be brief, and add "ANSWER:" before your final answer.<end_of_turn>
<start_of_turn>model
Here's how to solve the problem:
* **Letters per week:** 3 friends * 2 letters/week = 6 letters/week
```

## Code annotations printed next to listings (callouts, in order)
- Turns on random sampling to
- get a diverse range of outputs

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

## Previous page (PDF 526) text, for continuity
RAG works by taking an initial user question and doing some form of a query to pull in additional text context. This query can be to a database, a search engine, or anything that can give further information on the question asked by a user. This extra information is then added straight into the prompt. For example, you might construct a prompt like this:

Use the following pieces of context to answer the question. Question: What are some good ways to improve sleep? Context: {text from a medical journal on improving sleep} Answer:

A common approach for looking up relevant information is to use a vector database. To build a vector database, you can use an LLM, or any model, to embed a series of source documents as vectors. The document text will be stored in the database, with the embedding vector used as a key. During retrieval, an LLM can again be used to embed the user query as a vector. The vector database is responsible for searching for key vectors close to the query vector and for surfacing the corresponding text. This might sound a lot like the attention mechanism itself—recall that the terms “query,” “key,” and “value” actually came from database systems.

Surfacing information to assist with generation does a few things:

It gives you an obvious way to work around the cutoff date of the model.

It allows the model to access private data. Companies might want to use an LLM trained on public data to serve as an interface to information stored privately.

It can help factually ground the model. There is no silver bullet that stops hallucinations entirely, but an LLM is much less likely to make up facts on a topic if presented with correct context about the subject in a prompt.

16.4.4 “Reasoning” models

For years since the first LLMs, researchers have struggled w …

## Next page (PDF 528) text, for continuity
In the first attempt, our model was hung up on the superfluous detail that each letter has two pages. In the second attempt, the model gets the problem right. This instruction-tuned Gemma model we are working with has already been tuned on math problems like this; you would not get nearly as good results from the “untuned” Gemma model from the last section.

We could extend this idea to a very simple form of chain-of-thought training: 1 Collect a bunch of basic math and reasoning problems and desired answers. 2 Generate (with some randomness) a number of responses. 3 Find all the responses with a correct answer via string parsing. You can prompt

the model to use a specific text marker for the final answer as we did previously. 4 Run supervised fine-tuning on correct responses, including all the intermediate output. 5 Repeat! The previously described process is a reinforcement learning algorithm. Our answer checking acts as the environment, and the generated outputs are the actions the model uses to learn. As with RLHF, in practice you would use a more complex gradient update step to use information from all responses (even the incorrect ones), but the basic principle is the same. The same idea is being used to improve LLM performance in other domains that have obvious, verifiable answers to text prompts. Coding is an important one—you can prompt the LLM to output code and then actually run the code to test the quality of the response. In all these domains, one trend is clear—as a model learns to solve more difficult questions, the model will spend more and more time “showing its work” before reaching a final answer. You can think of this as the model learning to search over its own output of potential solutions. We will discuss this idea further in the final chapter of …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.521 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.524 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.518 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.522 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.511 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.513 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.418 — 14 Text classification › 14.2 Preparing text data
- PDF p.508 — 16 Text generation › 16.3 Using a pretrained LLM

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.519 (printed 495) — 16 Text generation › 16.4 Going further with LLMs › 16.4.1 Reinforcement Learning with Human Feedback (RLHF)
Any need to manually write text examples will almost always become a bottleneck— such data is slow and expensive to come by. Moreover, this approach will be limited by the human performance ceiling on the instruction-following task. If we want to do better than human performance in a chatbot-like experience, we cannot rely on manually written output to supervise LLM training. The real problem we are trying to optimize is our preference for certain responses over others. With a large enough sample of people, this preference problem is perfectly defined, but figuring out how to translate from “our preferences” to a loss function we could use to compute gradients is quite tricky. This is what Reinforcement Learning with Human Feedback, or RLHF, attempts to solve. The first step in RLHF fine-tuning is exactly what we did in the last section—supervised fine-tuning with handwritten prompts and responses. This gets us to a good baseline performance; we now need to improve on this baseline. To this end, we will build a reward model that can act as a proxy for human preference. We can gather a large number of prompts and responses to these prompts. Some of these responses can be handwritten

### PDF p.521 (printed 497) — 16 Text generation › 16.4 Going further with LLMs › Choosing a dtype for large models
This 4-billion-parameter model was first pretrained on 14 trillion tokens of text and then extensively fine-tuned to make it more helpful when answering questions. Some of this tuning was done with supervised fine-tuning like we did in the previous section, some with RLHF as we covered in this section, and some with still other techniques— like using an even larger model as a “teacher” to guide training. The increase in ability to do question-answering is easily noticeable.

Let’s try this model on the prompt that has been giving us trouble with hallucinations:

### PDF p.520 (printed 496) — 16 Text generation › 16.4 Going further with LLMs › 
In this simple example, we filter our generated responses with a reward cutoff, and simply treat the “good” output as new training data for more supervised fine-tuning like we just did in the last section. In practice, you will usually not discard your bad responses but rather use specialized gradient update algorithms to steer your model’s parameters using all responses and rewards. After all, a bad response gives a good signal on what not to do. OpenAI originally described RLHF in a 2022 paper5 and used this training setup to go from GPT-3’s initial pretrained parameters to the first version of ChatGPT.

An advantage of this setup is that it can be iterative. You can take this newly trained model, generate new and improved responses to prompts, rank these responses by human preference, and train a new and improved reward model.

### PDF p.512 (printed 488) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3.2 Instruction fine-tuning
Instruction fine-tuning involves feeding the model input/output pairs—a user instruction followed by a model response. We combine these into a single sequence that becomes new training data for the model. To make it clear during training when an instruction or response ends, we can add special markers like `"[instruction]"` and `"[response]"` directly to the combined sequence. The precise markup will not matter much as long as it is consistent.

We can use the combined sequence as regular training data, with the same “guess the next word” loss we used to pretrain an LLM. By doing further training with examples containing desired responses, we are essentially bending the model’s output in the direction we want. We won’t be learning a latent space for language here; that’s already been done over trillions of tokens of pretraining. We are simply nudging the learned representation a bit to control the tone and content of the output. To begin, we will need a dataset of instruction-response pairs. Training chatbots is a hot topic, so there are many datasets made specifically for this purpose. We will use a dataset made public by the company Databricks. Employees contributed to a dataset 

### PDF p.452 (printed 428) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
We have yet to produce the next great tragedy, but this is not terrible for two minutes of training on a minimal dataset. The goal of this toy example is to show the power of the language model setup. We trained the model on the narrow problem of guessing a single character at a time but still use it for a much broader problem, generating an open-ended, Shakespearean-like text response.

It’s important to notice that this training setup only works because a recurrent neural network only passes information forward in the sequence. If you’d like, try replacing the `GRU` layer with a `Bidirectional(GRU(...))`. The training accuracy will zoom to above 99% immediately, and generation will stop working entirely. During training, our model sees the entire sequence each train step. If we “cheat” by letting information from the next token in the sequence affect the current token’s prediction, we’ve made our problem trivially easy. This language modeling setup is fundamental to countless problems in the text domain. It is also somewhat unique compared to other modeling problems we have seen so far in this book. We cannot simply call `model.predict()` to get the desired output. There is an en

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

### PDF p.512 (printed 488) — 16 Text generation › 16.3 Using a pretrained LLM › 
Of course, this is utter nonsense, but the model could not find a more likely way to complete this prompt.

Hallucinations and uncontrollable output are fundamental problems with language models. If there is a silver bullet, we have yet to find it. However, one approach that helps immensely is to further fine-tune a model with examples of the specific types of generative outputs you would like. In the specific case of wanting to build a chatbot that can follow instructions, this type of training is called instruction fine-tuning. Let’s try some instruction fine-tuning with Gemma to make it a lot more useful as a conversation partner.

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
