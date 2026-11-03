# Generation bundle for page_511  (PDF page 511, printed page 487)

Page kind: content
Chapter: 16 Text generation
Section: 16.3 Using a pretrained LLM
Subsection: 16.3.1 Text generation with the Gemma model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_511.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_511.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p511_e03] (body) As we saw with the mini-GPT example, this is not so much a bug as a consequence of our pretraining objective. The Gemma model was trained with the same “guess the next word” objective we used for mini-GPT, which means it’s effectively a fancy autocomplete for the internet. It will just keep rattling off the most probable word in its single sequence as if your prompt was a snippet of text found in a random document on the web. One way to change our output is to prompt the model with a longer input that makes it obvious which type of output we are looking for. For example, if we prompt the Gemma model with the beginning two sentences of a brownie recipe, we get more helpful output:
[p511_e04] (code)
```
>>> gemma_lm.generate(
>>>     "The following brownie recipe is easy to make in just a few "
>>>     "steps.\n\nYou can start by",
>>>     max_length=40,
>>> )
The following brownie recipe is easy to make in just a few steps.
You can start by melting the butter and sugar in a saucepan over medium heat.
Then add the eggs and vanilla extract
```
[p511_e05] (body) Though it’s tempting when working with a model that can “talk” to imagine it interpreting our prompt in some sort of human, conversational way, nothing of the sort is going on here. We have just constructed a prompt for which an actual brownie recipe is a more likely continuation than mimicking someone posting on a forum asking for baking help.
[p511_e06] (body) You can go much further in constructing prompts. You might prompt a model with some natural language instructions of the role it is supposed to fill, for example, `"You` `are a large language model that gives short, helpful answers to people's questions."` Or you might feed the model a prompt containing a long list of harmful topics that should not be included in any generated responses. If this all sounds a bit hand-wavy and hard to control, that’s a good assessment. Attempting to visit different parts of a model’s distribution through prompting is often useful, but predicting how a model will respond to a given prompt is very difficult. Another well-documented problem faced by LLMs is hallucinations. A model will always say something—there is always a most-likely next token to a given sequence. Finding locations in our LLM distribution that have no grounding in actual fact is easy:
[p511_e07] (code)
```
>>> gemma_lm.generate(
>>>     "Tell me about the 542nd president of the United States.",
>>>     max_length=40,
>>> )
Tell me about the 542nd president of the United States.
The 542nd president of the United States was James A. Garfield.
```

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

## Previous page (PDF 510) text, for continuity
Rather than implementing a generation routine ourselves, we can simplify our lives by using the `generate()` function that comes as part of the `CausalLM` class. This `generate()` function can be compiled with different sampling strategies, as we explored in section 16.2:

We can notice a few things right off the bat. First, the output is much more coherent than our mini-GPT model. It would be hard to distinguish this text from much of the training data in the C4 dataset. Second, the output is still not that useful. The model will generate vaguely plausible text, but what you could do with it is unclear.

## Next page (PDF 512) text, for continuity
Of course, this is utter nonsense, but the model could not find a more likely way to complete this prompt.

Hallucinations and uncontrollable output are fundamental problems with language models. If there is a silver bullet, we have yet to find it. However, one approach that helps immensely is to further fine-tune a model with examples of the specific types of generative outputs you would like. In the specific case of wanting to build a chatbot that can follow instructions, this type of training is called instruction fine-tuning. Let’s try some instruction fine-tuning with Gemma to make it a lot more useful as a conversation partner.

16.3.2 Instruction fine-tuning

Instruction fine-tuning involves feeding the model input/output pairs—a user instruction followed by a model response. We combine these into a single sequence that becomes new training data for the model. To make it clear during training when an instruction or response ends, we can add special markers like `"[instruction]"` and `"[response]"` directly to the combined sequence. The precise markup will not matter much as long as it is consistent.

We can use the combined sequence as regular training data, with the same “guess the next word” loss we used to pretrain an LLM. By doing further training with examples containing desired responses, we are essentially bending the model’s output in the direction we want. We won’t be learning a latent space for language here; that’s already been done over trillions of tokens of pretraining. We are simply nudging the learned representation a bit to control the tone and content of the output. To begin, we will need a dataset of instruction-response pairs. Training chatbots is a hot topic, so there are many datasets made specifically for this purpose. We will use a dataset …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.518 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.522 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.502 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.505 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.503 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.513 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.494 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.451 — 15 Language models and the Transformer › 15.1 The language model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.528 (printed 504) — 16 Text generation › 16.5 Where are LLMs heading next? › 
In the first attempt, our model was hung up on the superfluous detail that each letter has two pages. In the second attempt, the model gets the problem right. This instruction-tuned Gemma model we are working with has already been tuned on math problems like this; you would not get nearly as good results from the “untuned” Gemma model from the last section.

We could extend this idea to a very simple form of chain-of-thought training: 1 Collect a bunch of basic math and reasoning problems and desired answers. 2 Generate (with some randomness) a number of responses. 3 Find all the responses with a correct answer via string parsing. You can prompt

the model to use a specific text marker for the final answer as we did previously. 4 Run supervised fine-tuning on correct responses, including all the intermediate output. 5 Repeat! The previously described process is a reinforcement learning algorithm. Our answer checking acts as the environment, and the generated outputs are the actions the model uses to learn. As with RLHF, in practice you would use a more complex gradient update step to use information from all responses (even the incorrect ones), but the basic principle is the same. 

### PDF p.502 (printed 478) — 16 Text generation › 16.2 Training a mini-GPT › 16.2.3 Generative decoding
To sample some output from our model, we can follow the approach we used to generate Shakespeare or Spanish translations in chapter 15. We feed a prompt of fixed tokens into the model. For each position in the input sequence, the model outputs a probability distribution over the entire vocabulary for the next token. By selecting the most likely next token at the last location, adding it to our sequence, and then repeating this process, we are able to generate a new sequence, one token at a time.

Listing 16.9 A simple generation function for the mini-GPT model

### PDF p.505 (printed 481) — 16 Text generation › 16.2 Training a mini-GPT › 
This isn’t so much a bug as it’s a direct consequence of our training objective. Our model is trying to predict the most likely next token in a sequence across about a billion words on many, many topics. If there’s no obvious choice for where a sequence of text should head next, an effective strategy is to guess common words or repeated patterns of words. Unsurprisingly, our model learns to do this during training almost immediately. If you were to stop training our model very early on, it would likely generate the word `"the"` incessantly, as `"the"` is the most common word in the English language. During our generative loop, we have always chosen the most likely predicted token in our model’s output. But our output is not just a single predicted token; it is a probability distribution across all 32,000 tokens in our vocabulary. Using the most likely output at each generation step is called greedy search. It’s the most straightforward approach to using model predictions, but it is hardly the only one. If we instead add some randomness to the process, we can explore the probability distribution learned by the model more broadly. This can keep us from getting stuck in loops of high-

### PDF p.513 (printed 489) — 16 Text generation › 16.3 Using a pretrained LLM › 
Our prompt template gives our examples a predictable structure. Although Gemma is not a sequence-to-sequence model like our English-to-Spanish translator, we can still use it in a sequence-to-sequence setting by training on prompts like this and only generating the output after the `"[response]"` marker.

Let’s make a `tf.data.Dataset` and split some validation data:

### PDF p.527 (printed 503) — 16 Text generation › 16.4 Going further with LLMs › 
you didn’t even need examples; you could simply prompt the model with the phrase “Let’s think step by step” and get better output.

Since these discoveries, there has been heavy interest in directly training LLMs to get better at chain-of-thought reasoning. Models like OpenAI’s o1 and DeepSeek’s r1 have made headlines by showing significant strides in math and coding problems by training a model to “think out loud” on difficult questions. The approach for this chain-of-thought fine-tuning is very similar to RLHF. We will first train the model on a few supervised examples of “showing your work” on a math problem and arriving at a correct answer. Next, we will prompt the model with a new math question and check whether the model got the final answer correct. Finally, we use these new generated outputs to further tune the model’s weights. Let’s try this out with the Gemma model. We can write out our own word problem and turn on random sampling so we get a somewhat random response each time:

### PDF p.508 (printed 484) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3 Using a pretrained LLM
Now that we’ve trained a mini-language model from scratch, let’s try using a billion-parameter pretrained model and see what it can do. Given how prohibitively expensive pretraining a Transformer can be, most of the industry has centered around using pretrained models developed by a relatively short list of companies. This is not purely a cost concern but also an environmental one—generative model training is now making up a large percentage of the total data center power consumption of large tech companies.

Meta published some environmental data on Llama 2, an LLM it published in 2023. It’s a good bit smaller than GPT-3, but it needed an estimated 1.3 million kilowatt hours of electricity to train—the daily power usage of about 45,000 American households. If every organization using an LLM ran pretraining themselves, the scale of energy use would be a noticeable percentage of global energy consumption. Let’s play around with a pretrained generative model from Google called Gemma. We will use the third version of the Gemma model, which was released to the public in 2025. To keep the examples in this book accessible, we will use the smallest variation of

### PDF p.494 (printed 470) — 16 Text generation › 16.2 Training a mini-GPT › 
GPT-3 was still plagued by fundamental issues that have yet to be solved. LLMs “hallucinate” often—their output can veer from accurate to completely false with zero indication. They’re extremely sensitive to prompt phrasing, with seemingly minor prompt rewording triggering large jumps up or down in performance. And they cannot adapt to problems that weren’t extensively featured in their training data.

However, the generative output from GPT-3 was good enough that the model became the basis for ChatGPT—the first widespread, consumer-facing generative model. In the months and years since, ChatGPT has sparked a deluge of investment and interest in building LLMs and finding new use cases for them. In the next section, we will make a miniature GPT model of our own to better understand how such a model works, what it can do, and where it fails.

### PDF p.451 (printed 427) — 15 Language models and the Transformer › 15.1 The language model › 
To begin generation, we first need to “prime” the internal state of the GRU with our prompt. To do this, we will feed the prompt into the model one token at a time. This will compute the exact RNN state the model would see if this prompt had been encountered during training.

When we feed the very last character of the prompt into the model, our state output will capture information about the entire prompt sequence. We can save the final output prediction to later select the first character of our generated response.

Listing 15.7 Using a fixed prompt to compute a language model’s starting state
