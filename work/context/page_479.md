# Generation bundle for page_479  (PDF page 479, printed page 455)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.4 Classification with a pretrained Transformer
Subsection: 15.4.2 Loading a pretrained Transformer
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_479.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_479.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p479_e03] (body) model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.
[p479_e04] (body) BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’t need any labels about the text you feed in; for any text sequence, you can easily choose some random tokens and mask them out. That made it easy for the authors to find a large amount of text data needed to train models of this size. For the most part, they pulled from Wikipedia as a source. Using pretrained word embeddings was already common practice when BERT was released—we saw this ourselves in the last chapter. But pretraining an entire Transformer gave something much more powerful—the ability to compute a word embedding for a word in the context of the words around it. And the Transformer allowed doing this at a scale and quality that were unheard of at the time. The authors of BERT took this model, pretrained on a huge amount of text, and specialized it to achieve state-of-the-art results on several NLP benchmarks at the time. This marked a distinctive shift in the field toward using very large, pretrained models, often with only a small amount of fine-tuning. Let’s try this out.
[p479_e05] (heading_subsection) 15.4.2 Loading a pretrained Transformer
[p479_e06] (body) Instead of using BERT here, let’s use a follow-up model called RoBERTa,3 short for Robustly Optimized BERT. RoBERTa made some minor simplifications to BERT’s architecture, but most notably used more training data to improve performance. BERT used 16 GB of English language text, mainly from Wikipedia. The RoBERTa authors used 160 GB of text from all over the web. It’s estimated that RoBERTa cost a few hundred thousand dollars to train at the time. Because of this extra training data, the model performs noticeably better for an equivalent overall parameter count.
[p479_e07] (body) To use a pretrained model we will need a few things:
[p479_e08] (bullet) A matching tokenizer—Used with the pretrained model itself. Any text must be tokenized in the same way as during pretraining. If the words of our IMDb
[p479_e09] (footnote) 3 Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach” (2019), https://arxiv.org/ abs/1907.11692.

## Chapter outline
Chapter: 15 Language models and the Transformer (PDF pages 445-489)
  - 15.1 The language model (p.445)
      - 15.1.1 Training a Shakespeare language model (p.446)
      - 15.1.2 Generating Shakespeare (p.450)
  - 15.2 Sequence-to-sequence learning (p.452)
      - 15.2.1 English-to-Spanish translation (p.454)
      - 15.2.2 Sequence-to-sequence learning with RNNs (p.456)
  - 15.3 The Transformer architecture (p.461)
      - 15.3.1 Dot-product attention (p.463)
      - 15.3.2 Transformer encoder block (p.468)
      - 15.3.3 Transformer decoder block (p.470)
      - 15.3.4 Sequence-to-sequence learning with a Transformer (p.472)
      - 15.3.5 Embedding positional information (p.475)
  - 15.4 Classification with a pretrained Transformer (p.478)
      - 15.4.1 Pretraining a Transformer encoder (p.478)
      - 15.4.2 Loading a pretrained Transformer (p.479)
      - 15.4.3 Preprocessing IMDb movie reviews (p.482)
      - 15.4.4 Fine-tuning a pretrained Transformer (p.484)
  - 15.5 What makes the Transformer effective? (p.485)

## Previous page (PDF 478) text, for continuity
Subjectively, the Transformer performs significantly better than the GRU-based translation model. It’s still a toy model, but it’s a better toy model.

The Transformer is a powerful architecture that has laid the basis for an explosion of interest in text-processing models. It’s also fairly complex, as deep learning models go. After seeing all of these implementation details, one might reasonably protest that this all seems quite arbitrary. There are so many small details to take on faith. How could we possibly know this choice and configuration of layers is optimal? The answer is simple—it’s not. Over the years, a number of improvements have been proposed to the Transformer architecture by making changes to attention, normalization, and positional embeddings. Many new models in research today are replacing attention altogether with something less computationally complex as sequence lengths get very long. Eventually, perhaps by the time you are reading this book, something will have supplanted the Transformer as the dominant architecture used for language modeling. There’s a lot we can learn from the Transformer that will stand the test of time. At the end of this chapter, we will discuss what makes the Transformer so effective. But it’s worth remembering that, as a whole, the field of machine learning moves empirically. Attention grew out of an attempt to augment RNNs, and after years of guessing and checking by a ton of people, it gave rise to the Transformer. There is little reason to think this process is done playing out.

15.4 Classification with a pretrained Transformer

After “Attention is all you need,” people started to notice how far Transformer training could scale, especially compared to models that had come before. As we just mentioned, one big plus was th …

## Next page (PDF 480) text, for continuity
reviews map to different token indices than they would have during pretraining, we cannot use the learned representations of each token in the model.

A matching model architecture—To use the pretrained model, we need to recreate the math used internally by the model for pretraining exactly.

The pretrained weights—These weights were created by training the model for about a day on 1,024 GPUs and billions of input words.

Recreating the tokenizer and architecture code ourselves would not be too hard. The model internals almost exactly match the `TransformerEncoder` we built previously. However, matching a model implementation is a time-consuming process, and as we have done earlier in this book, we can instead use the KerasHub library to access pretrained model implementations for Keras.

Let’s use KerasHub to load a RoBERTa tokenizer and model. We can use the special constructor `from_preset()` to load a pretrained model’s weights, configuration, and tokenizer assets from disk. We will load RoBERTa’s base model, which is the smallest of the few pretrained checkpoints released with the RoBERTa paper.

Listing 15.20 Loading the RoBERTa pretrained model with KerasHub

The `Tokenizer` maps from text to sequences of integers, as we would expect. Remember the `SubWordTokenizer` we built in the last chapter? RoBERTa’s tokenizer is almost the same as that tokenizer, with minor tweaks to handle Unicode characters from any language.

Given the size of RoBERTa’s pretraining dataset, subword tokenization is a must. Using a character-level tokenizer would make input sequences way too long, making the model much more expensive to train. Using a word-level tokenizer would require a massive vocabulary to attempt to cover all the distinct words in the millions of documents of text used …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.488 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.484 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.14 — contents › 
- PDF p.438 — 14 Text classification › 14.5 Sequence models
- PDF p.493 — 16 Text generation › 16.1 A brief history of sequence generation
- PDF p.446 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.452 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.483 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.484 (printed 460) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › Where does pretraining data come from?
Transformers are data-hungry models. They perform better the more input you throw at them at a scale previously unseen in deep learning. The original Transformer was trained on translation data with many millions of tokens. Not long after, Transformers were trained with billions, and today, trillions of tokens. That’s a lot of written words. So where does all this data come from? The answer has changed over time, but the short answer is the internet. Another answer is that this has increasingly become a secret. Companies will often not release the exact data they have used to train a model or describe the precise mixture of data sources used during training. We can take a look at some pretraining datasets over time:

The very first Transformer trained on a well-known English-German translation

dataset with 4 million sentence pairs.

BERT used a dump of English Wikipedia plus a dataset containing 7,000

self-published books.

GPT2, a precursor to ChatGPT, scraped a dataset by following outgoing links

from Reddit.

The latest version of Llama, a pretrained Transformer released by Meta, is

trained on “15 trillion tokens of data from publicly available sources.” It is becoming incre

### PDF p.488 (printed 464) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › Summary
A language model is a model that learns a specific probability distribution— `p(token|past tokens)`: – Language models have broad applications, but the most important is that you can generate text by calling them in a loop, where the output token at one time step becomes the input token in the next. – A masked language model learns a related probability distribution `p(tokens|surrounding` `tokens)` and can be helpful for classifying text and individual tokens. – A sequence-to-sequence language model learns to predict the next token given both past tokens in a target sequence and an entirely separate, fixed source sequence. Sequence-to-sequence models are useful for problems like translation and question answering. – A sequence-to-sequence model usually has two separate components. An encoder computes a representation of the source sequence, and a decoder takes this representation as input and predicts the next token in a target sequence based on past tokens.

Attention is a mechanism that allows a model to pull information from anywhere in a sequence selectively based on the context of the token currently being processed: – Attention avoids the problems RNNs have with long-range de

### PDF p.493 (printed 469) — 16 Text generation › 16.1 A brief history of sequence generation › 
In 2018, a year after the “Attention Is All You Need” paper we discussed in the last chapter, a group of researchers at an organization called OpenAI put out a new paper “Improving Language Understanding by Generative Pre-Training.”3 They combined a few ingredients:

Unsupervised pretraining of a language model—essentially training a model to “guess the next token” in a sequence, as we did with our Shakespeare generator in chapter 15

The Transformer architecture

Textual data on various topics via thousands of self-published books

The authors showed that such a pretrained model could be fine-tuned to achieve state-of-the-art performance on a wide array of text classification tasks—from gauging the similarity of two sentences to answering a multiple-choice question. They called the pretrained model GPT, short for Generative Pretrained Transformer.

GPT didn’t come with any modeling or training advancements. What was interesting about the results was that such a general training setup could beat out more involved techniques across a number of tasks. There was no complex text normalization, no need to customize the model architecture or training data per benchmark, just a lot of pre

### PDF p.489 (printed 465) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The Transformer is a sequence modeling architecture that uses attention as the only mechanism to pass information across a sequence: – The Transformer works by stacking blocks of alternating attention and two-layer feedforward networks. – The Transformer can scale to many parameters and lots of training data while still improving accuracy in the language modeling problem. – Unlike RNNs, the Transformer involves no sequence-length loops at training time, making the model much easier to train in parallel across many machines. – A Transformer encoder uses bidirectional attention to build a rich representation of a sequence. – A Transformer decoder uses causal attention to predict the next word in a language model setup.

### PDF p.438 (printed 414) — 14 Text classification › 14.5 Sequence models › 14.5.4 Pretraining a word embedding
The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models with millions or even billions of parameters, text models become incredibly data-hungry. We are usually limited by our ability to find labeled examples for a particular problem in the text domain.

The idea is to devise an unsupervised task to train model parameters that do not need labeled data. Pretraining data can be text in a similar domain to our final task, or even just arbitrary text in the languages we are interested in working with. Pretraining allows us to learn general patterns in language, effectively priming our model before we specialize it to the final task we are interested in. Word embeddings were one of the first big successes with text pretraining, and we will show how to pretrain a word embedding in this section. Remember the `unsup/` directory we ignored in our IMDb dataset preparation? It contains another 25,000 reviews— the same size as our training data. We will combine all our training data together and show how to pretrain the parameters of an `Emb

### PDF p.524 (printed 500) — 16 Text generation › 16.4 Going further with LLMs › 
Each of our input prompts contains the special token `<start_of_image>`. This is turned into 256 placeholder values in our input sequence, which, in turn, is replaced with the soft tokens representing our image.

Training for a multimodal model like this is quite similar to regular LLM pretraining and fine-tuning. Usually, you would want to first pretrain your image encoder separately, like we first did in Chapter 8 of this book. Then you can simply do the same basic “guess the next word” pretraining and also feed in mixed image and text content combined into a single sequence. Our transformer would not be trained to output image soft tokens; we would simply zero the loss at these image token locations. It might seem almost magical that we can simply add image data to an LLM, but when we consider the power of the sequence model we’re working with, it’s really quite an expected result. We’ve taken a Transformer, recast our image input as sequence data, and done a lot of extra training. The model can preserve the original language model’s ability to ingest and produce text while learning to also embed images in the Transformer’s latent space.

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences
