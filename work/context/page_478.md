# Generation bundle for page_478  (PDF page 478, printed page 454)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.4 Classification with a pretrained Transformer
Subsection: 15.4.1 Pretraining a Transformer encoder
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_478.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_478.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p478_e03] (body) Subjectively, the Transformer performs significantly better than the GRU-based translation model. It’s still a toy model, but it’s a better toy model.
[p478_e04] (body) The Transformer is a powerful architecture that has laid the basis for an explosion of interest in text-processing models. It’s also fairly complex, as deep learning models go. After seeing all of these implementation details, one might reasonably protest that this all seems quite arbitrary. There are so many small details to take on faith. How could we possibly know this choice and configuration of layers is optimal? The answer is simple—it’s not. Over the years, a number of improvements have been proposed to the Transformer architecture by making changes to attention, normalization, and positional embeddings. Many new models in research today are replacing attention altogether with something less computationally complex as sequence lengths get very long. Eventually, perhaps by the time you are reading this book, something will have supplanted the Transformer as the dominant architecture used for language modeling. There’s a lot we can learn from the Transformer that will stand the test of time. At the end of this chapter, we will discuss what makes the Transformer so effective. But it’s worth remembering that, as a whole, the field of machine learning moves empirically. Attention grew out of an attempt to augment RNNs, and after years of guessing and checking by a ton of people, it gave rise to the Transformer. There is little reason to think this process is done playing out.
[p478_e05] (heading_section) 15.4 Classification with a pretrained Transformer
[p478_e06] (body) After “Attention is all you need,” people started to notice how far Transformer training could scale, especially compared to models that had come before. As we just mentioned, one big plus was that the model is faster to train than RNNs. No more loops during training, which is always good when working with a GPU or TPU.
[p478_e07] (body) It is also a very data hungry model architecture. We actually got a little taste of this in the last section. While our RNN translation model plateaued in validation performance after 5 or so epochs, the Transformer model was still improving its validation score after 30 epochs of training. These observations prompted many to try scaling up the Transformer with more data, layers, and parameters—with great results. This caused a distinctive shift in the field toward large pretrained models that can cost millions to train but perform noticeably better on a wide range of problems in the text domain. For our last code example in the text section, we will revisit our IMDb text-classification problem, this time with a pretrained Transformer model.
[p478_e08] (heading_subsection) 15.4.1 Pretraining a Transformer encoder
[p478_e09] (body) One of the first pretrained Transformers to become popular in NLP was called BERT, short for Bidirectional Encoder Representations from Transformers.2 The paper and
[p478_e10] (footnote) 2 Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (2019), https://arxiv.org/abs/1810.04805.

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

## Previous page (PDF 477) text, for continuity
Let’s rerun generation with our newly trained `Transformer`. We can use the same code as we did for our RNN sampling.

Listing 15.19 Generating translations with a Transformer

Running the generation code, we get the following output:

## Next page (PDF 479) text, for continuity
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’t need any labels about the text you feed in; for any text sequence, you can easily choose some random tokens and mask them out. That made it easy for the authors to find a large amount of text data needed to train models of this size. For the most part, they pulled from Wikipedia as a source. Using pretrained word embeddings was already common practice when BERT was released—we saw this ourselves in the last chapter. But pretraining an entire Transformer gave something much more powerful—the ability to compute a word embedding for a word in the context of the words around it. And the Transfor …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.489 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.461 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.474 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.476 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.484 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.445 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.487 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.407 — 14 Text classification › 14.1 A brief history of natural language processing

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.489 (printed 465) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The Transformer is a sequence modeling architecture that uses attention as the only mechanism to pass information across a sequence: – The Transformer works by stacking blocks of alternating attention and two-layer feedforward networks. – The Transformer can scale to many parameters and lots of training data while still improving accuracy in the language modeling problem. – Unlike RNNs, the Transformer involves no sequence-length loops at training time, making the model much easier to train in parallel across many machines. – A Transformer encoder uses bidirectional attention to build a rich representation of a sequence. – A Transformer decoder uses causal attention to predict the next word in a language model setup.

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

### PDF p.461 (printed 437) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3 The Transformer architecture
In 2017, Vaswani et al. introduced the Transformer architecture in the seminal paper “Attention Is All You Need.”1 The authors were working on translation systems like the one we just built, and the critical discovery is right in the title. As it turned out, a simple mechanism called attention can be used to construct powerful sequence models that don’t feature recurrent layers at all. The idea of attention was not new and had been used in NLP systems for a couple of years when they published. But the idea that attention was so useful it could be the only mechanism used to pass information across a sequence was quite surprising at the time.

This finding unleashed nothing short of a revolution in natural language processing—and beyond. Attention has fast become one of the most influential ideas in deep learning. In this section, you’ll get an in-depth explanation of how it works and why it has proven so effective for sequence modeling. We’ll then use attention to rebuild our English-to-Spanish translation model. So, with all that as build-up, what exactly is attention? And how does it offer a replacement for the recurrent neural networks we have used so far? Attention was actually 

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

### PDF p.407 (printed 383) — 14 Text classification › 14.1 A brief history of natural language processing › 
In the last decade, NLP researchers and practitioners have discovered just how shockingly effective it can be to learn the answer to narrow statistical questions about text. In the 2010s, researchers began applying LSTM models to text, dramatically increasing the number of parameters in NLP models and the compute resources required to train them. The results were encouraging—LSTMs could generalize to unseen examples with far greater accuracy than previous approaches, but they eventually hit limits. LSTMs struggled to track dependencies in long chains of text with many sentences and paragraphs, and compared to computer vision models, they were slow and unwieldy to train. Toward the end of the 2010s, researchers at Google discovered a new architecture called the Transformer that solved many scalability issues plaguing LSTMs. As long as you increased the size of a model and its training data together, Transformers appeared to perform more and more accurately. Better yet, the computations needed for training a Transformer could be effectively parallelized, even for long sequences. If you doubled the number of machines doing training, you could roughly halve the time you need to wait fo

### PDF p.476 (printed 452) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
With positional information back in the model, things went much better. We achieved a 67% accuracy when guessing the next word. It’s a noticeable improvement from the `GRU` model, and that’s all the more impressive when you consider that this model has half the parameters of the GRU counterpart.

There’s one other important thing to notice about this training run. Training is noticeably faster than the RNN—each epoch takes about a third of the time. This would be true even if we matched parameter count with the RNN model, and it is a side effect of getting rid of the looped state passing of our `GRU` layers. With attention, there is no looping computation to handle during training, meaning that on a GPU or TPU, we can handle the entire attention computation in one go. This makes the `Transformer` quicker to train on accelerators.

### PDF p.474 (printed 450) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
After training, we get to about 58% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 58% of the time. Something is off here. Training is worse than the RNN model by 7 percentage points. Either this Transformer architecture is not what it was hyped up to be, or we missed something in our implementation. Can you spot what it is?

This section is ostensibly about sequence models. In the previous chapter, we saw how vital word order could be to meaning. And yet, the Transformer we just built wasn’t a sequence model at all. Did you notice? It’s composed of dense layers that process sequence tokens independently of each other and an attention layer that looks at the tokens as a set. You could change the order of the tokens in a sequence, and you’d get identical pairwise attention scores and the same context-aware representations. If you were to rearrange every word in every English source sentence completely, the model wouldn’t notice, and you’d still get the same accuracy. Attention is a set-processing mechanism, focused on the relationships between pairs of sequence elements—it’s blind to whether these elements occur at the beginning, at the end,

### PDF p.445 (printed 421) — 15 Language models and the Transformer › 15.1 The language model › This chapter covers
How to generate text with a deep learning model

Training a model to translate from English to Spanish

The Transformer, a powerful architecture for text

modeling problems

With the basics of text preprocessing and modeling covered in the previous chapter, this chapter will tackle some more involved language problems such as machine translation. We will build up a solid intuition for the Transformer model that powers products like ChatGPT and has helped trigger a wave of investment in natural language processing (NLP).
