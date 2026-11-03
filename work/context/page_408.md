# Generation bundle for page_408  (PDF page 408, printed page 384)

Page kind: content
Chapter: 14 Text classification
Section: 14.2 Preparing text data
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_408.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_408.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p408_e03] (body) recognition machines. Despite our persistent human tendency to find intelligence in “things that talk,” these models copy and synthesize training data in a way that is wholly distinct (and much less efficient!) than human intelligence. However, it is also fair to say that the emergence of complex behaviors from incredibly simple “guess the missing word” training setups has been one of the most shocking empirical results in the last decade of machine learning.
[p408_e04] (body) In the following three chapters, we will look at a range of techniques for machine learning with text data. We will skip discussion of the hardcoded linguistic features that prevailed until the 1990s, but we will look at everything from running logistic regressions for classifying text to training LSTMs for machine translation. We will closely examine the Transformer model and discuss what makes it so scalable and effective at generalizing in the text domain. Let’s dig in.
[p408_e05] (heading_section) 14.2 Preparing text data
[p408_e06] (body) Let’s consider an English sentence:
[p408_e07] (code)
```
The quick brown fox jumped over the lazy dog.
```
[p408_e08] (body) There is an obvious blocker before we can start applying any of the deep learning techniques of previous chapters—our input is not numeric! Before beginning any modeling, we need to translate the written word into tensors of numbers. Unlike images, which have a relatively natural numeric representation, you could build a numeric representation of text in several ways.
[p408_e09] (body) A simple approach would be to borrow from standard text file formats for text and use something like an ASCII encoding. We could chop the input into a sequence of characters and assign each a unique index. Another intuitive approach would be building a representation based on words, first breaking sentences apart on all spaces and punctuation and then mapping each word to a unique numeric representation. These are both good approaches to try, and in general, all text preprocessing will include a splitting step, where text is split into small individual units, called tokens. A powerful tool for splitting text is regular expressions, which can flexibly match patterns of characters in text. Let’s look at how to use a regular expression to split a string into a sequence of characters. The most basic regex we can apply is `"."`, which matches any character in the input text:
[p408_e10] (code)
```
import regex as re
def split_chars(text):
    return re.findall(r".", text)
```
[p408_e11] (body) We can apply the function to our example input string:

## Chapter outline
Chapter: 14 Text classification (PDF pages 405-444)
  - 14.1 A brief history of natural language processing (p.405)
  - 14.2 Preparing text data (p.408)
      - 14.2.1 Character and word tokenization (p.411)
      - 14.2.2 Subword tokenization (p.414)
  - 14.3 Sets vs. sequences (p.419)
      - 14.3.1 Loading the IMDb classification dataset (p.420)
  - 14.4 Set models (p.422)
      - 14.4.1 Training a bag-of-words model (p.423)
      - 14.4.2 Training a bigram model (p.427)
  - 14.5 Sequence models (p.429)
      - 14.5.1 Training a recurrent model (p.430)
      - 14.5.2 Understanding word embeddings (p.433)
      - 14.5.3 Using a word embedding (p.434)
      - 14.5.4 Pretraining a word embedding (p.438)
      - 14.5.5 Using the pretrained embedding for classification (p.442)

## Previous page (PDF 407) text, for continuity
automate the development of the kind of if/then/else rules of hardcoded language systems. Then, statistical approaches started gaining speed, starting with logistic regression. Over time, learned parametric models took over, and linguistics came to be seen by some as a hindrance when baked directly into a model. Frederick Jelinek, an early speech recognition researcher, joked in the 1990s, “Every time I fire a linguist, the performance of the speech recognizer goes up.”

Much as computer vision is pattern recognition applied to pixels, the modern field of NLP is all about pattern recognition applied to words in text. There’s no shortage of practical applications:

Given the text of an email, what is the probability that it is spam? (text classification)

Given an English sentence, what is the most likely Russian translation?

(translation)

Given an incomplete sentence, what word will likely come next? (language

modeling)

The text-processing models you will train in this book won’t possess a human-like understanding of language; rather, they simply look for statistical regularities in their input data, which turns out to be sufficient to perform well on a wide array of real-world tasks.

In the last decade, NLP researchers and practitioners have discovered just how shockingly effective it can be to learn the answer to narrow statistical questions about text. In the 2010s, researchers began applying LSTM models to text, dramatically increasing the number of parameters in NLP models and the compute resources required to train them. The results were encouraging—LSTMs could generalize to unseen examples with far greater accuracy than previous approaches, but they eventually hit limits. LSTMs struggled to track dependencies in long chains of text with many sentences and pa …

## Next page (PDF 409) text, for continuity
Regex can easily be applied to split our text into words instead. The `"[\w]+"` regular expression will grab consecutive non-whitespace characters, and the `"[.,!?;]"` can match the punctuation marks between the brackets. We can combine the two to achieve a regular expression that splits each word and punctuation mark into a token:

Here’s what it does to a test sentence:

Splitting takes us from a single string to a token sequence, but we still need to transform our string tokens into numeric inputs. By far the most common approach is to map each token to a unique integer index, often called indexing our input. This is a flexible and reversible representation of our tokenized input that can work with a wide range of modeling approaches. Later on, we can decide how to map from token indices into a latent space ingested by the model.

For character tokens, we could use ASCII lookups to index each token—for example, `ord('A') `→` 65` and `ord('z') `→` 122`. However, this can scale poorly when you start to consider other languages—there are over a million characters in the Unicode specification! A more robust technique is to build a mapping from specific tokens in our training data to indices that occur in the data we care about, which in NLP is called a vocabulary. This has the nice property of working for word-level tokens as easily as for character-level tokens. Let’s take a look at how we might use a vocabulary to transform text. We will build a simple Python dictionary that maps tokens to indices, split our input into tokens, and finally index our tokens:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.415 — 14 Text classification › 14.2 Preparing text data
- PDF p.410 — 14 Text classification › 14.2 Preparing text data
- PDF p.437 — 14 Text classification › 14.5 Sequence models
- PDF p.405 — 14 Text classification › 14.1 A brief history of natural language processing
- PDF p.426 — 14 Text classification › 14.4 Set models
- PDF p.438 — 14 Text classification › 14.5 Sequence models
- PDF p.412 — 14 Text classification › 14.2 Preparing text data
- PDF p.418 — 14 Text classification › 14.2 Preparing text data

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences

### PDF p.405 (printed 381) — 14 Text classification › 14.1 A brief history of natural language processing › This chapter covers
An introduction to the field of natural language

processing (NLP)

Preprocessing text input into numeric input

Building simple text classification models

This chapter will lay the foundation for working with text input that we will build on in the next two chapters of this book. By the end of this chapter, you will be able to build a simple text classifier in a number of different ways. This will set the stage for building more complicated models, like the Transformer, in the next chapter.

### PDF p.438 (printed 414) — 14 Text classification › 14.5 Sequence models › 14.5.4 Pretraining a word embedding
The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models with millions or even billions of parameters, text models become incredibly data-hungry. We are usually limited by our ability to find labeled examples for a particular problem in the text domain.

The idea is to devise an unsupervised task to train model parameters that do not need labeled data. Pretraining data can be text in a similar domain to our final task, or even just arbitrary text in the languages we are interested in working with. Pretraining allows us to learn general patterns in language, effectively priming our model before we specialize it to the final task we are interested in. Word embeddings were one of the first big successes with text pretraining, and we will show how to pretrain a word embedding in this section. Remember the `unsup/` directory we ignored in our IMDb dataset preparation? It contains another 25,000 reviews— the same size as our training data. We will combine all our training data together and show how to pretrain the parameters of an `Emb

### PDF p.447 (printed 423) — 15 Language models and the Transformer › 15.1 The language model › 
To build a language model from this input, we will need to massage our source text. First, we will split our data into equal-length chunks that we can batch and use for model training, much as we did for weather measurements in the timeseries chapters. Because we will be using a character-level tokenizer here, we can do this chunking directly on the string input. A 100-character string will map to a 100-integer sequence.

We will also split each input into two separate feature and label sequences, with each label sequence simply being the input sequence offset by a single character.

Listing 15.2 Splitting text into chunks for language model training

### PDF p.445 (printed 421) — 15 Language models and the Transformer › 15.1 The language model › 15.1 The language model
In the previous chapter, we learned how to convert text data to numeric inputs, and we used this numeric representation to classify movie reviews. However, text classification is, in many ways, a uniquely simple problem. We only need to output a single floating-point number for binary classification and, at worst, N numbers for N-way classification.

### PDF p.479 (printed 455) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

### PDF p.643 (printed 619) — index ›  › 
stemming, defined 386–387

TensorFlow Serving 185 step fusing 556 tensor operations, derivatives of 42–43 stochastic gradient descent 43–46 tensors, defined 63 strided convolutions 239 test images, preparing 321 subclassing Model class 202 test-time training 582 rewriting previous example as subclassed text classification 381 model 202

augmenting text data 413 what subclassed models don't support 204 NLP (natural language processing) 382–384 subgraph isomorphism 586 preparing text data 384–394 subword tokenization 390–394 pretraining word embeddings 414 summary() method 193 sequence models 405 supervised sets vs. sequences 395–398 fine-tuning 495 setting models 398–404 learning 214 training recurrent model 406 symbolic AI 3 using pretrained embedding for symbolic tensor 196

classification 418 symbols 584–594 word embeddings 409, 410 blending deep learning and program text-conditioning 509 synthesis 590–592 text_dataset_from_directory utility 397 cognition as combination of both kinds of abstraction 587 text_embeddings input 531 deep learning and 588 text generation 466 long-term vision 594 history of sequence generation 468–470 modular component recombination and lifelong LLMs (l
