# Generation bundle for page_524  (PDF page 524, printed page 500)

Page kind: content
Chapter: 16 Text generation
Section: 16.4 Going further with LLMs
Subsection: 16.4.2 Multimodal LLMs
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_524.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_524.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p524_e03] (body) We can use Gemma to ask some questions about this image:
[p524_e04] (code)
```
>>> gemma_lm.preprocessor.max_images_per_prompt = 1
>>> gemma_lm.preprocessor.sequence_length = 512
>>> prompt = "What is going on in this image? Be concise!<start_of_image>"
>>> gemma_lm.generate({
...     "prompts": PROMPT_TEMPLATE.format(prompt),
...     "images": [image],
... })
<start_of_turn>user
What is going on in this image? Be concise!
<start_of_image>
<end_of_turn>
<start_of_turn>model
A snake wearing glasses is sitting in a leather armchair, surrounded by a large
bookshelf, and reading a book. It's a whimsical, slightly surreal image.
<end_of_turn>
>>> prompt = "What is the snake wearing?<start_of_image>"
>>> gemma_lm.generate({
...     "prompts": PROMPT_TEMPLATE.format(prompt),
...     "images": [image],
... })
<start_of_turn>user
What is the snake wearing?
<start_of_image>
<end_of_turn>
<start_of_turn>model
The snake is wearing a pair of glasses! They are red-framed and perched on its
head.<end_of_turn>
```
[p524_e05] (body) Each of our input prompts contains the special token `<start_of_image>`. This is turned into 256 placeholder values in our input sequence, which, in turn, is replaced with the soft tokens representing our image.
[p524_e06] (body) Training for a multimodal model like this is quite similar to regular LLM pretraining and fine-tuning. Usually, you would want to first pretrain your image encoder separately, like we first did in Chapter 8 of this book. Then you can simply do the same basic “guess the next word” pretraining and also feed in mixed image and text content combined into a single sequence. Our transformer would not be trained to output image soft tokens; we would simply zero the loss at these image token locations. It might seem almost magical that we can simply add image data to an LLM, but when we consider the power of the sequence model we’re working with, it’s really quite an expected result. We’ve taken a Transformer, recast our image input as sequence data, and done a lot of extra training. The model can preserve the original language model’s ability to ingest and produce text while learning to also embed images in the Transformer’s latent space.

## Code annotations printed next to listings (callouts, in order)
- Limits the maximum
- input size of the model

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

## Previous page (PDF 523) text, for continuity
Figure 16.6 Handling image input by splicing text tokens and soft image tokens together Figure 16.7 A test image for the Gemma model

## Next page (PDF 525) text, for continuity
Foundation models

As LLMs venture into different modalities, the “large language model” moniker can become a bit misleading. They do model language, but also images, audio, maybe even structured data. In the next chapter, we will see a distinct architecture, called diffusion models, that works quite differently in terms of underlying structure but has a similar feel—they too are trained on massive amounts of data at “internet scale” with a self-supervised loss.

An umbrella term for models like this is foundation models. More specifically, a foundation model is any model that is trained on broad data (generally using self-supervision at scale) that can be fine-tuned to a wide range of downstream tasks. In general, you can think of a foundation model as learning to reconstruct data pulled from large swaths of the internet, given a partial representation of it. While LLMs are the first and best-known of these models, there are many others. The hallmarks of a foundation model are the self-supervised learning objective (a reconstruction loss) and the fact that these models are not specialized to a single task and can be used for a number of downstream purposes. This is an important and striking shift that has happened quite recently in the long history of machine learning. Rather than training a model from scratch on your individual dataset, you will often be better off using a foundation model to get a rich representation of your input (whether it’s images, text, or something else) and then specialize that model for your final downstream task. Of course, this comes with the downside of needing to run large models with billions of parameters, so it’s hardly a fit for all real-world applications of machine learning.

16.4.3 Retrieval Augmented Generation (RAG)

Sticking ext …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.522 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.527 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.521 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.556 — 17 Image generation › 17.3 Text-to-image models
- PDF p.513 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.558 — 17 Image generation › 17.3 Text-to-image models
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models
- PDF p.511 — 16 Text generation › 16.3 Using a pretrained LLM

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.522 (printed 498) — 16 Text generation › 16.4 Going further with LLMs › 16.4.2 Multimodal LLMs
One obvious chatbot extension is the ability to handle new modalities of input. An assistant that can respond to audio input and process images would be far more useful than one that can only operate on text.

Extending a Transformer to different modalities can be done in a conceptually simple way. The Transformer is not a text-specific model; it’s a highly effective model for learning patterns in sequence data. If we can figure out how to coerce other data types into a sequence representation, we can feed this sequence into a Transformer and train with it. In fact, the Gemma model we just loaded does just that. The model comes with a separate 420-million-parameter image encoder that cuts an input image into 256 patches and encodes each patch as a vector with the same dimensionality as Gemma’s hidden transformer dimension. Each image will be embedded as a `(256,` `2560)` sequence. Because 2560 is the hidden dimensionality of the Gemma Transformer model, this image representation can simply be spliced into our text sequence after the token embedding layer. You can think of it like 256 special tokens representing the image, where each `(1,` `2560)` vector is sometimes called a “soft 

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

### PDF p.479 (printed 455) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.512 (printed 488) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3.2 Instruction fine-tuning
Instruction fine-tuning involves feeding the model input/output pairs—a user instruction followed by a model response. We combine these into a single sequence that becomes new training data for the model. To make it clear during training when an instruction or response ends, we can add special markers like `"[instruction]"` and `"[response]"` directly to the combined sequence. The precise markup will not matter much as long as it is consistent.

We can use the combined sequence as regular training data, with the same “guess the next word” loss we used to pretrain an LLM. By doing further training with examples containing desired responses, we are essentially bending the model’s output in the direction we want. We won’t be learning a latent space for language here; that’s already been done over trillions of tokens of pretraining. We are simply nudging the learned representation a bit to control the tone and content of the output. To begin, we will need a dataset of instruction-response pairs. Training chatbots is a hot topic, so there are many datasets made specifically for this purpose. We will use a dataset made public by the company Databricks. Employees contributed to a dataset 

### PDF p.429 (printed 405) — 14 Text classification › 14.5 Sequence models › 14.5 Sequence models
Our last two models indicated that sequence information is important. We improved a basic linear model by adding features with some info on local word order.

However, this was done by manually engineering input features, and we can see how the approach will only scale up to a local ordering of just a few words. As is often the case in deep learning, rather than attempting to build these features ourselves, we should expose the model to the raw word sequence and let it directly learn positional dependencies between tokens. Models that ingest a complete token sequence are called, simply enough, sequence models. We have a few choices for architecture here. We could build an RNN model as we just did for timeseries modeling. We could build a 1D ConvNet, similar to our image processing models, but convolving filters over a single sequence dimension. And as we will dig into in the next chapter, we can build a Transformer. Before taking on any of these approaches, we must preprocess our inputs into ordered sequences. We want an integer sequence of token IDs, as we saw in the tokenization portion of this chapter, but with one additional wrinkle to handle. When we run computations on a batc

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › Summary
Large language models, or LLMs, are the combination of a few key ingredients: – The Transformer architecture – A language modeling task (predicting the next token based on past tokens) – A large amount of unlabeled text data

An LLM learns a probability distribution for predicting individual tokens. This can be combined with a sampling strategy to generate a long string of text. There are many popular ways to sample text: – Greedy search takes the most likely predicted token at each generation step. – Random sampling directly samples the predicted categorical distribution over all tokens. – Top-K sampling restricts the categorical distribution to the top set of K candidates.

LLMs use billions of parameters and are trained on trillions of words of text.
