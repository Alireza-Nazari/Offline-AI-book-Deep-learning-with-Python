# Generation bundle for page_445  (PDF page 445, printed page 421)

Page kind: chapter_opener
Chapter: 15 Language models and the Transformer
Section: 15.1 The language model
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_445.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_445.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p445_e01] (chapter_number) 15
[p445_e02] (chapter_title) Language models and
[p445_e03] (chapter_title) the Transformer
[p445_e04] (chapter_covers) This chapter covers
[p445_e05] (bullet) How to generate text with a deep learning model
[p445_e06] (bullet) Training a model to translate from English to Spanish
[p445_e07] (bullet) The Transformer, a powerful architecture for text
[p445_e08] (bullet) modeling problems
[p445_e09] (body) With the basics of text preprocessing and modeling covered in the previous chapter, this chapter will tackle some more involved language problems such as machine translation. We will build up a solid intuition for the Transformer model that powers products like ChatGPT and has helped trigger a wave of investment in natural language processing (NLP).
[p445_e10] (heading_section) 15.1 The language model
[p445_e11] (body) In the previous chapter, we learned how to convert text data to numeric inputs, and we used this numeric representation to classify movie reviews. However, text classification is, in many ways, a uniquely simple problem. We only need to output a single floating-point number for binary classification and, at worst, N numbers for N-way classification.

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

## Previous page (PDF 444) text, for continuity
An embedding is an efficient way to transform token IDs into a learned latent space. Embeddings can be trained normally with gradient descent.

Pretraining is critical for sequence models as a way to get around the data-hungry nature of these models. During pretraining, an unsupervised task allows models to learn from large amounts of unlabeled text data. The learned parameters can then be transferred to a downstream task.

## Next page (PDF 446) text, for continuity
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences. Attempting to represent every output sequence as a unique classifier output would overwhelm compute resources no matter how we design our model. A practical approach for making such a prediction problem feasible is to build a model that only predicts a single token output at a time. A language model is a model that, in its simplest form, learns a straightforward but deep probability distribution: `p(token|past tokens)`. Given a sequence of all tokens observed up to a point, a language model will attempt to output a probability distribution over all possible tokens that could come next. A 20, …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.452 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.478 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.407 — 14 Text classification › 14.1 A brief history of natural language processing
- PDF p.405 — 14 Text classification › 14.1 A brief history of natural language processing
- PDF p.461 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.408 — 14 Text classification › 14.2 Preparing text data
- PDF p.488 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.448 — 15 Language models and the Transformer › 15.1 The language model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.478 (printed 454) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
Subjectively, the Transformer performs significantly better than the GRU-based translation model. It’s still a toy model, but it’s a better toy model.

The Transformer is a powerful architecture that has laid the basis for an explosion of interest in text-processing models. It’s also fairly complex, as deep learning models go. After seeing all of these implementation details, one might reasonably protest that this all seems quite arbitrary. There are so many small details to take on faith. How could we possibly know this choice and configuration of layers is optimal? The answer is simple—it’s not. Over the years, a number of improvements have been proposed to the Transformer architecture by making changes to attention, normalization, and positional embeddings. Many new models in research today are replacing attention altogether with something less computationally complex as sequence lengths get very long. Eventually, perhaps by the time you are reading this book, something will have supplanted the Transformer as the dominant architecture used for language modeling. There’s a lot we can learn from the Transformer that will stand the test of time. At the end of this chapter, we will 

### PDF p.405 (printed 381) — 14 Text classification › 14.1 A brief history of natural language processing › This chapter covers
An introduction to the field of natural language

processing (NLP)

Preprocessing text input into numeric input

Building simple text classification models

This chapter will lay the foundation for working with text input that we will build on in the next two chapters of this book. By the end of this chapter, you will be able to build a simple text classifier in a number of different ways. This will set the stage for building more complicated models, like the Transformer, in the next chapter.

### PDF p.452 (printed 428) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 15.2 Sequence-to-sequence learning
Let’s take the language model idea and extend it to tackle an important problem— machine translation. Translation belongs to a class of modeling problems often called sequence-to-sequence modeling (or seq2seq if you are trying to save keystrokes). We seek to build a model that can take in a source text as a fixed input sequence and generate the translated text sequence as a result. Question answering is another classic sequence-to-sequence problem.

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

### PDF p.479 (printed 455) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’

### PDF p.407 (printed 383) — 14 Text classification › 14.1 A brief history of natural language processing › 
In the last decade, NLP researchers and practitioners have discovered just how shockingly effective it can be to learn the answer to narrow statistical questions about text. In the 2010s, researchers began applying LSTM models to text, dramatically increasing the number of parameters in NLP models and the compute resources required to train them. The results were encouraging—LSTMs could generalize to unseen examples with far greater accuracy than previous approaches, but they eventually hit limits. LSTMs struggled to track dependencies in long chains of text with many sentences and paragraphs, and compared to computer vision models, they were slow and unwieldy to train. Toward the end of the 2010s, researchers at Google discovered a new architecture called the Transformer that solved many scalability issues plaguing LSTMs. As long as you increased the size of a model and its training data together, Transformers appeared to perform more and more accurately. Better yet, the computations needed for training a Transformer could be effectively parallelized, even for long sequences. If you doubled the number of machines doing training, you could roughly halve the time you need to wait fo

### PDF p.452 (printed 428) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
We have yet to produce the next great tragedy, but this is not terrible for two minutes of training on a minimal dataset. The goal of this toy example is to show the power of the language model setup. We trained the model on the narrow problem of guessing a single character at a time but still use it for a much broader problem, generating an open-ended, Shakespearean-like text response.

It’s important to notice that this training setup only works because a recurrent neural network only passes information forward in the sequence. If you’d like, try replacing the `GRU` layer with a `Bidirectional(GRU(...))`. The training accuracy will zoom to above 99% immediately, and generation will stop working entirely. During training, our model sees the entire sequence each train step. If we “cheat” by letting information from the next token in the sequence affect the current token’s prediction, we’ve made our problem trivially easy. This language modeling setup is fundamental to countless problems in the text domain. It is also somewhat unique compared to other modeling problems we have seen so far in this book. We cannot simply call `model.predict()` to get the desired output. There is an en
