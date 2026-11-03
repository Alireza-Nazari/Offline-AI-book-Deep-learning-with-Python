# Generation bundle for page_519  (PDF page 519, printed page 495)

Page kind: content
Chapter: 16 Text generation
Section: 16.4 Going further with LLMs
Subsection: 16.4.1 Reinforcement Learning with Human Feedback (RLHF)
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_519.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_519.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p519_e03] (heading_section) 16.4 Going further with LLMs
[p519_e04] (body) We have now trained a GPT model from scratch and fine-tuned a language model into our very own chatbot. However, we are just scratching the surface of LLM research today. In this section, we will cover a non-exhaustive list of extensions and improvements to the basic “autocomplete the internet” language modeling setup.
[p519_e05] (heading_subsection) 16.4.1 Reinforcement Learning with Human Feedback (RLHF)
[p519_e06] (body) The type of instruction fine-tuning we just did is often called supervised fine-tuning. It is supervised because we are curating, by hand, a list of example prompts and responses we want from the model.
[p519_e07] (body) Any need to manually write text examples will almost always become a bottleneck— such data is slow and expensive to come by. Moreover, this approach will be limited by the human performance ceiling on the instruction-following task. If we want to do better than human performance in a chatbot-like experience, we cannot rely on manually written output to supervise LLM training. The real problem we are trying to optimize is our preference for certain responses over others. With a large enough sample of people, this preference problem is perfectly defined, but figuring out how to translate from “our preferences” to a loss function we could use to compute gradients is quite tricky. This is what Reinforcement Learning with Human Feedback, or RLHF, attempts to solve. The first step in RLHF fine-tuning is exactly what we did in the last section—supervised fine-tuning with handwritten prompts and responses. This gets us to a good baseline performance; we now need to improve on this baseline. To this end, we will build a reward model that can act as a proxy for human preference. We can gather a large number of prompts and responses to these prompts. Some of these responses can be handwritten; the model can write others. Responses could even be written by other chatbot LLMs. We then need to get human evaluators to rank these responses by preference. Given a prompt and several potential responses, an evaluator’s task is to rank them from most helpful to least helpful. Such data collection is expensive and slow, but still faster than writing all the desired responses by hand. We can use this ranked preference dataset to build the reward model, which takes in a prompt-response pair and outputs a single floating-point value. The higher the value, the better the response. This reward model is usually another, smaller Transformer. Instead of predicting the next token, it reads a whole sequence and outputs a single float—a rating for a given response. We can then use this reward model to tune our model further, using a reinforcement learning setup. We won’t get too deep into the details of reinforcement learning in this book, but don’t be too intimidated by the term—it refers to any training setup where a deep learning model learns by making predictions (called actions) and getting feedback on that output (called rewards). In short, a model’s own predictions become its training data.

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

## Previous page (PDF 518) text, for continuity
Did our fine-tuning make our model better at following directions? Let’s give it a try:

Much better. Our model will now respond to questions, instead of trying to simply carry on the thought of the prompt text.

Have we solved the hallucination problem?

Not at all. However, we could still use instruction tuning to make some inroads here. A common technique is to train the model on a lot of instruction/response pairs where the desired response is `"I` `don't` `know"` or `"As` `a` `language` `model,` `I` `cannot` `help` `you` `with ` `that"`. This can train the model to avoid attempting to answer specific topics where it would often give poor-quality results.

## Next page (PDF 520) text, for continuity
In our case, the action is simply generating a response to an input prompt, like we have been doing above with the `generate()` function. The reward is simply applying a separate regression model to that string output. Here’s a simple example in pseudocode.

Listing 16.15 Pseudocode for the simplest possible RLHF algorithm

In this simple example, we filter our generated responses with a reward cutoff, and simply treat the “good” output as new training data for more supervised fine-tuning like we just did in the last section. In practice, you will usually not discard your bad responses but rather use specialized gradient update algorithms to steer your model’s parameters using all responses and rewards. After all, a bad response gives a good signal on what not to do. OpenAI originally described RLHF in a 2022 paper5 and used this training setup to go from GPT-3’s initial pretrained parameters to the first version of ChatGPT.

An advantage of this setup is that it can be iterative. You can take this newly trained model, generate new and improved responses to prompts, rank these responses by human preference, and train a new and improved reward model.

4.1.1. Using a chatbot trained with RLHF

We can make this more concrete by trying a model trained with this form of iterative preference tuning. Since building chatbots is the “killer app” for large Transformer models, it is common practice for companies that release pretrained models like Gemma to release specialized “instruction-tuned” versions, built just for chat. Let’s try loading one now. This will be a 4-billion-parameter model, quadruple the size of the model we just loaded and the largest model we will use in this book.

Listing 16.16 Loading an instruction-tuned Gemma variant

5 Ouyang et al., “Training Language  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.512 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.528 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.521 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.530 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.513 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.531 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.494 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.525 — 16 Text generation › 16.4 Going further with LLMs

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.512 (printed 488) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3.2 Instruction fine-tuning
Instruction fine-tuning involves feeding the model input/output pairs—a user instruction followed by a model response. We combine these into a single sequence that becomes new training data for the model. To make it clear during training when an instruction or response ends, we can add special markers like `"[instruction]"` and `"[response]"` directly to the combined sequence. The precise markup will not matter much as long as it is consistent.

We can use the combined sequence as regular training data, with the same “guess the next word” loss we used to pretrain an LLM. By doing further training with examples containing desired responses, we are essentially bending the model’s output in the direction we want. We won’t be learning a latent space for language here; that’s already been done over trillions of tokens of pretraining. We are simply nudging the learned representation a bit to control the tone and content of the output. To begin, we will need a dataset of instruction-response pairs. Training chatbots is a hot topic, so there are many datasets made specifically for this purpose. We will use a dataset made public by the company Databricks. Employees contributed to a dataset 

### PDF p.528 (printed 504) — 16 Text generation › 16.5 Where are LLMs heading next? › 
In the first attempt, our model was hung up on the superfluous detail that each letter has two pages. In the second attempt, the model gets the problem right. This instruction-tuned Gemma model we are working with has already been tuned on math problems like this; you would not get nearly as good results from the “untuned” Gemma model from the last section.

We could extend this idea to a very simple form of chain-of-thought training: 1 Collect a bunch of basic math and reasoning problems and desired answers. 2 Generate (with some randomness) a number of responses. 3 Find all the responses with a correct answer via string parsing. You can prompt

the model to use a specific text marker for the final answer as we did previously. 4 Run supervised fine-tuning on correct responses, including all the intermediate output. 5 Repeat! The previously described process is a reinforcement learning algorithm. Our answer checking acts as the environment, and the generated outputs are the actions the model uses to learn. As with RLHF, in practice you would use a more complex gradient update step to use information from all responses (even the incorrect ones), but the basic principle is the same. 

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

### PDF p.527 (printed 503) — 16 Text generation › 16.4 Going further with LLMs › 
you didn’t even need examples; you could simply prompt the model with the phrase “Let’s think step by step” and get better output.

Since these discoveries, there has been heavy interest in directly training LLMs to get better at chain-of-thought reasoning. Models like OpenAI’s o1 and DeepSeek’s r1 have made headlines by showing significant strides in math and coding problems by training a model to “think out loud” on difficult questions. The approach for this chain-of-thought fine-tuning is very similar to RLHF. We will first train the model on a few supervised examples of “showing your work” on a math problem and arriving at a correct answer. Next, we will prompt the model with a new math question and check whether the model got the final answer correct. Finally, we use these new generated outputs to further tune the model’s weights. Let’s try this out with the Gemma model. We can write out our own word problem and turn on random sampling so we get a somewhat random response each time:

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

### PDF p.521 (printed 497) — 16 Text generation › 16.4 Going further with LLMs › Choosing a dtype for large models
This 4-billion-parameter model was first pretrained on 14 trillion tokens of text and then extensively fine-tuned to make it more helpful when answering questions. Some of this tuning was done with supervised fine-tuning like we did in the previous section, some with RLHF as we covered in this section, and some with still other techniques— like using an even larger model as a “teacher” to guide training. The increase in ability to do question-answering is easily noticeable.

Let’s try this model on the prompt that has been giving us trouble with hallucinations:

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › 
We can note that while the total number of tokens used for pretraining has climbed steadily and massively, model parameter counts have varied substantially since GPT-3. In part, this is because we now know GPT-3 was undertrained, but it is also for a more practical reason. When deploying a model, it’s often worth it to sacrifice performance for a smaller model that fits on cheaper hardware. A really good model won’t help very much if it’s prohibitively expensive to run. There’s another reason we might not be able to just scale up these models thoughtlessly: we are starting to run out of pretraining data! Tech companies are starting to have trouble finding more high-quality, public, human-written content to throw at pretraining. Models are even starting to “eat their own tail” by training on a significant portion of content created by other LLMs, which runs into a whole other host of concerns. This is one of the reasons reinforcement learning is getting a lot of attention recently. If you can create a difficult, self-contained environment that generates new problems for an LLM to attempt, you will have found a way to continue training using the model’s own output—no need to scrounge

### PDF p.512 (printed 488) — 16 Text generation › 16.3 Using a pretrained LLM › 
Of course, this is utter nonsense, but the model could not find a more likely way to complete this prompt.

Hallucinations and uncontrollable output are fundamental problems with language models. If there is a silver bullet, we have yet to find it. However, one approach that helps immensely is to further fine-tune a model with examples of the specific types of generative outputs you would like. In the specific case of wanting to build a chatbot that can follow instructions, this type of training is called instruction fine-tuning. Let’s try some instruction fine-tuning with Gemma to make it a lot more useful as a conversation partner.
