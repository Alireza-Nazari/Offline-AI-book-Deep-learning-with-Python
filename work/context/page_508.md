# Generation bundle for page_508  (PDF page 508, printed page 484)

Page kind: content
Chapter: 16 Text generation
Section: 16.3 Using a pretrained LLM
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_508.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_508.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p508_e03] (body) zeros out the probability of anything outside the K candidates. You can combine the two, for example, sampling the top five candidates with a temperature of 0.5:
[p508_e04] (code)
```
>>> compiled_generate(prompt, partial(top_k, k=5, temperature=0.5))
A piece of advice that you can use to get rid of the problem.
The first thing you need to do is to get the job done. It is important that you
have a plan that will help you get rid of it.
The first thing you need to do is to get rid of the problem yourself.
```
[p508_e05] (body) A sampling strategy is an important control when generating text, and there are many more approaches. For example, beam search is a technique that heuristically explores multiple chains of predicted tokens by keeping a fixed number of “beams” (different chains of predicted tokens) to explore at each timestep.
[p508_e06] (body) With top-K sampling, our model generates something closer to plausible English text, but there is little apparent utility to such output. This fits with the results of GPT-1. For the initial GPT paper, the generated output was more of a curiosity, and state-of-the-art results were only achieved by fine-tuning classification models. Our mini-GPT is far less trained than GPT-1. To reach the scale of generative LLMs today, we’d need to increase our parameter count by at least 100× and our train step count by at least 1,000×. If we did, we would see the same leaps in quality observed by OpenAI with GPT. And we could do it! The training recipe we used previously is the exact blueprint used by everyone training LLMs today. The only missing pieces are a very large compute budget and some tricks for training across multiple machines that we will cover in chapter 18. For a more practical approach, we will transition to using a pretrained model. This will allow us to explore the behavior of an LLM at today’s scale.
[p508_e07] (heading_section) 16.3 Using a pretrained LLM
[p508_e08] (body) Now that we’ve trained a mini-language model from scratch, let’s try using a billion-parameter pretrained model and see what it can do. Given how prohibitively expensive pretraining a Transformer can be, most of the industry has centered around using pretrained models developed by a relatively short list of companies. This is not purely a cost concern but also an environmental one—generative model training is now making up a large percentage of the total data center power consumption of large tech companies.
[p508_e09] (body) Meta published some environmental data on Llama 2, an LLM it published in 2023. It’s a good bit smaller than GPT-3, but it needed an estimated 1.3 million kilowatt hours of electricity to train—the daily power usage of about 45,000 American households. If every organization using an LLM ran pretraining themselves, the scale of energy use would be a noticeable percentage of global energy consumption. Let’s play around with a pretrained generative model from Google called Gemma. We will use the third version of the Gemma model, which was released to the public in 2025. To keep the examples in this book accessible, we will use the smallest variation of

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

## Previous page (PDF 507) text, for continuity
Figure 16.3 Greedy, top-K, and random sampling strategies shown on the same probability distribution

Let’s try this out in code. We can use `keras.ops.top_k` to find the top K elements of an array:

We can try a few different variations of top-K to see how it affects sampling:

Passing a top-K cutoff is different than temperature sampling. Passing a low temperature makes likely tokens more likely, but it does not rule any token out. Top-K sampling

## Next page (PDF 509) text, for continuity
Gemma available, which clocks in at almost exactly 1 billion parameters. This “small” model was trained on roughly 2 trillion tokens of pretraining data—2,000 times more tokens than the mini-GPT we just trained!

16.3.1 Text generation with the Gemma model

To load this pretrained model, we can use KerasHub, as we have done in previous chapters.

Accessing Gemma weights

If you are running the code for this chapter yourself, you will need to accept a Terms of Use for the Gemma models before you can download the weights. The model weights are stored on Kaggle, and we can use the `kagglehub` API to log in as we did in chapter 8. Before we do, you will need to do two things: 1 Go to https://www.kaggle.com/models/keras/gemma3, and accept the Gemma Terms of Use at the top of the page. 2 Go to https://www.kaggle.com/settings, and generate a Kaggle API Key (if you did not already do this in chapter 8). With that, we can use our API key to authenticate with Kaggle from our notebook:

As LLMs become increasingly powerful, terms of service like this are becoming more and more common. The Gemma terms of use prohibit using the model for things like generating spam or hate speech.

Listing 16.11 Instantiating a pretrained LLM with KerasHub

`CausalLM` is another example of the high-level task API, much like the `ImageClassifier` and `ImageSegmenter` tasks we used earlier in the book. The `CausalLM` task will combine a tokenizer and correctly initialized architecture into a single Keras model. KerasHub will load the Gemma weights into a correctly initialized architecture and load a matching tokenizer for the pretrained weights.

Let’s take a look at the Gemma model summary:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.506 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.530 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.494 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.511 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.505 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.493 — 16 Text generation › 16.1 A brief history of sequence generation
- PDF p.495 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.521 — 16 Text generation › 16.4 Going further with LLMs

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.493 (printed 469) — 16 Text generation › 16.1 A brief history of sequence generation › 
In 2018, a year after the “Attention Is All You Need” paper we discussed in the last chapter, a group of researchers at an organization called OpenAI put out a new paper “Improving Language Understanding by Generative Pre-Training.”3 They combined a few ingredients:

Unsupervised pretraining of a language model—essentially training a model to “guess the next token” in a sequence, as we did with our Shakespeare generator in chapter 15

The Transformer architecture

Textual data on various topics via thousands of self-published books

The authors showed that such a pretrained model could be fine-tuned to achieve state-of-the-art performance on a wide array of text classification tasks—from gauging the similarity of two sentences to answering a multiple-choice question. They called the pretrained model GPT, short for Generative Pretrained Transformer.

GPT didn’t come with any modeling or training advancements. What was interesting about the results was that such a general training setup could beat out more involved techniques across a number of tasks. There was no complex text normalization, no need to customize the model architecture or training data per benchmark, just a lot of pre

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › 
We can note that while the total number of tokens used for pretraining has climbed steadily and massively, model parameter counts have varied substantially since GPT-3. In part, this is because we now know GPT-3 was undertrained, but it is also for a more practical reason. When deploying a model, it’s often worth it to sacrifice performance for a smaller model that fits on cheaper hardware. A really good model won’t help very much if it’s prohibitively expensive to run. There’s another reason we might not be able to just scale up these models thoughtlessly: we are starting to run out of pretraining data! Tech companies are starting to have trouble finding more high-quality, public, human-written content to throw at pretraining. Models are even starting to “eat their own tail” by training on a significant portion of content created by other LLMs, which runs into a whole other host of concerns. This is one of the reasons reinforcement learning is getting a lot of attention recently. If you can create a difficult, self-contained environment that generates new problems for an LLM to attempt, you will have found a way to continue training using the model’s own output—no need to scrounge

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › Summary
Large language models, or LLMs, are the combination of a few key ingredients: – The Transformer architecture – A language modeling task (predicting the next token based on past tokens) – A large amount of unlabeled text data

An LLM learns a probability distribution for predicting individual tokens. This can be combined with a sampling strategy to generate a long string of text. There are many popular ways to sample text: – Greedy search takes the most likely predicted token at each generation step. – Random sampling directly samples the predicted categorical distribution over all tokens. – Top-K sampling restricts the categorical distribution to the top set of K candidates.

LLMs use billions of parameters and are trained on trillions of words of text.

### PDF p.493 (printed 469) — 16 Text generation › 16.1 A brief history of sequence generation › 
The language modeling setup enabled each of these models to generate text, and the developers at OpenAI noticed that with each leap in scale, the quality of this generative output shot up substantially.

With GPT-1, the model’s generative capabilities were mostly a by-product of its pretraining and not the primary focus. They evaluated the model by fine-tuning it with an extra dense layer for classification, as we did with RoBERTa in the last chapter. With GPT-2, the authors noticed that you could prompt the model with a few examples of a task and generate quality output without any fine-tuning. For instance, you could prompt the model with the following to receive a French translation of the word cheese:

3 Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever, “Improving Language Understanding by Generative Pre-Training,” (2018), https://mng.bz/GweD.

### PDF p.501 (printed 477) — 16 Text generation › 16.2 Training a mini-GPT › 
Figure 16.2 Warmup makes our updates to model parameters smaller at the beginning of training and can help with stability.

We will train our model using one pass over our 1 billion tokens, split across eight epochs so we can occasionally check our validation set loss and accuracy.

We are training a miniature version of GPT, using 3× fewer parameters than GPT-1 and 100× fewer overall training steps. But despite this being two orders of magnitude cheaper to train than the smallest GPT model, this call to `fit()` will be the most computationally expensive training run in the entire book. If you are running the code as you read, set things off and take a breather!

Listing 16.8 Training the mini-GPT model

### PDF p.506 (printed 482) — 16 Text generation › 16.2 Training a mini-GPT › 
At a high temperature, our outputs no longer resemble English, settling on seemingly random tokens. At a low temperature, our model behavior starts to resemble greedy search, repeating certain patterns of text over and over.

Another popular technique for shaping our distribution is restricting our sampling to a set of the most likely tokens. This is called top-K sampling, where K is the number of candidates you should explore. Figure 16.3 shows how top-K sampling strikes a middle ground between greedy and random approaches.

### PDF p.529 (printed 505) — 16 Text generation › 16.5 Where are LLMs heading next? › 
better model performance. So recently, model sizes have trended flatter while data sizes have trended up.

This doesn’t mean that scaling will stop—more computing power does generally lead to better LLM performance, and we have yet to see signs of an asymptote where next token prediction performance levels off. Companies are continuing to invest billions of dollars in scaling LLMs and seeing what new capabilities emerge. Figure 16.8 shows details for some of the major LLMs released from 2018 to 2025.

Figure 16.8 LLM parameter counts (top) and pretraining dataset sizes (bottom) over time. Many recent proprietary LLMs (e.g., GPT-4 and Gemini) are not included because model details have not been disclosed.

### PDF p.519 (printed 495) — 16 Text generation › 16.4 Going further with LLMs › 16.4 Going further with LLMs
We have now trained a GPT model from scratch and fine-tuned a language model into our very own chatbot. However, we are just scratching the surface of LLM research today. In this section, we will cover a non-exhaustive list of extensions and improvements to the basic “autocomplete the internet” language modeling setup.
