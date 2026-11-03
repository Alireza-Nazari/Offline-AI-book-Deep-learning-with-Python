# Generation bundle for page_405  (PDF page 405, printed page 381)

Page kind: chapter_opener
Chapter: 14 Text classification
Section: 14.1 A brief history of natural language processing
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_405.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_405.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p405_e01] (chapter_number) 14
[p405_e02] (chapter_title) Text classification
[p405_e03] (chapter_covers) This chapter covers
[p405_e04] (bullet) An introduction to the field of natural language
[p405_e05] (bullet) processing (NLP)
[p405_e06] (bullet) Preprocessing text input into numeric input
[p405_e07] (bullet) Building simple text classification models
[p405_e08] (body) This chapter will lay the foundation for working with text input that we will build on in the next two chapters of this book. By the end of this chapter, you will be able to build a simple text classifier in a number of different ways. This will set the stage for building more complicated models, like the Transformer, in the next chapter.
[p405_e09] (heading_section) 14.1 A brief history of natural language processing
[p405_e10] (body) In computer science, we refer to human languages, like English or Mandarin, as “natural” languages to distinguish them from languages that were designed for machines, like LISP, Assembly, and XML. Every machine language was designed: its starting point was an engineer writing down a set of formal rules to describe what statements you can make and what they mean. The rules came first, and people only started using the language once the rule set was complete. With human language,

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

## Previous page (PDF 404) text, for continuity
Summary

As you first learned in chapter 6, when approaching a new problem, it’s good to first establish commonsense baselines for your metric of choice. If you don’t have a baseline to beat, you can’t tell whether you’re making real progress.

Try simple models before expensive ones, to justify the additional expense. Sometimes a simple model will turn out to be your best option.

When you have data where ordering matters—in particular, for timeseries data— recurrent networks are a great fit and easily outperform models that first flatten the temporal data. The two essential RNN layers available in Keras are the `LSTM` layer and the `GRU` layer.

To use dropout with recurrent networks, you should use a time-constant dropout mask and recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the `recurrent_dropout` arguments of recurrent layers.

Stacked RNNs provide more representational power than a single RNN layer. They’re also much more expensive and thus not always worth it. Although they offer clear gains on complex problems (such as machine translation), they may not always be relevant to smaller, simpler problems.

## Next page (PDF 406) text, for continuity
it’s the reverse: usage comes first, and rules arise later. Natural language was shaped by an evolutionary process, much like biological organisms—that’s what makes it “natural.” Its “rules,” like the grammar of English, were formalized after the fact and are often ignored or broken by its users. As a result, while machine-readable language is highly structured and rigorous, natural language is messy—ambiguous, chaotic, sprawling, and constantly in flux.

Computer scientists have long fixated on the potential of systems that can ingest or produce natural language. Language, particularly written text, underpins most of our communications and cultural production. Centuries of human knowledge are stored via text; the internet is mostly text, and even our thoughts are based on language! The practice of using computers to interpret and manipulate language is called natural language processing, or NLP for short. It was first proposed as a field of study immediately following World War II, where some thought we could view understanding language as a form of “code cracking,” where natural language is the “code” used to transmit information. In the early days of the field, many people naively thought that you could write down the “rule set of English,” much like one can write down the rule set of LISP. In the early 1950s, researchers at IBM and Georgetown demonstrated a system that could translate Russian into English. The system used a grammar with six hardcoded rules and a lookup table with a couple of hundred elements (words and suffixes) to translate 60 handpicked Russian sentences accurately. The goal was to drum up excitement and funding for machine translation, and in that sense, it was a huge success. Despite the limited nature of the demo, the authors claimed that withi …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.407 — 14 Text classification › 14.1 A brief history of natural language processing
- PDF p.445 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.408 — 14 Text classification › 14.2 Preparing text data
- PDF p.447 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.443 — 14 Text classification › 14.5 Sequence models
- PDF p.411 — 14 Text classification › 14.2 Preparing text data
- PDF p.419 — 14 Text classification › 14.3 Sets vs. sequences
- PDF p.488 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.407 (printed 383) — 14 Text classification › 14.1 A brief history of natural language processing › 
automate the development of the kind of if/then/else rules of hardcoded language systems. Then, statistical approaches started gaining speed, starting with logistic regression. Over time, learned parametric models took over, and linguistics came to be seen by some as a hindrance when baked directly into a model. Frederick Jelinek, an early speech recognition researcher, joked in the 1990s, “Every time I fire a linguist, the performance of the speech recognizer goes up.”

Much as computer vision is pattern recognition applied to pixels, the modern field of NLP is all about pattern recognition applied to words in text. There’s no shortage of practical applications:

Given the text of an email, what is the probability that it is spam? (text classification)

Given an English sentence, what is the most likely Russian translation?

(translation)

Given an incomplete sentence, what word will likely come next? (language

modeling)

The text-processing models you will train in this book won’t possess a human-like understanding of language; rather, they simply look for statistical regularities in their input data, which turns out to be sufficient to perform well on a wide array of real-world

### PDF p.445 (printed 421) — 15 Language models and the Transformer › 15.1 The language model › 15.1 The language model
In the previous chapter, we learned how to convert text data to numeric inputs, and we used this numeric representation to classify movie reviews. However, text classification is, in many ways, a uniquely simple problem. We only need to output a single floating-point number for binary classification and, at worst, N numbers for N-way classification.

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences

### PDF p.408 (printed 384) — 14 Text classification › 14.2 Preparing text data › 14.2 Preparing text data
There is an obvious blocker before we can start applying any of the deep learning techniques of previous chapters—our input is not numeric! Before beginning any modeling, we need to translate the written word into tensors of numbers. Unlike images, which have a relatively natural numeric representation, you could build a numeric representation of text in several ways.

A simple approach would be to borrow from standard text file formats for text and use something like an ASCII encoding. We could chop the input into a sequence of characters and assign each a unique index. Another intuitive approach would be building a representation based on words, first breaking sentences apart on all spaces and punctuation and then mapping each word to a unique numeric representation. These are both good approaches to try, and in general, all text preprocessing will include a splitting step, where text is split into small individual units, called tokens. A powerful tool for splitting text is regular expressions, which can flexibly match patterns of characters in text. Let’s look at how to use a regular expression to split a string into a sequence of characters. The most basic regex we can apply is

### PDF p.445 (printed 421) — 15 Language models and the Transformer › 15.1 The language model › This chapter covers
How to generate text with a deep learning model

Training a model to translate from English to Spanish

The Transformer, a powerful architecture for text

modeling problems

With the basics of text preprocessing and modeling covered in the previous chapter, this chapter will tackle some more involved language problems such as machine translation. We will build up a solid intuition for the Transformer model that powers products like ChatGPT and has helped trigger a wave of investment in natural language processing (NLP).

### PDF p.411 (printed 387) — 14 Text classification › 14.2 Preparing text data › 
as models have grown more expressive, this type of standardization tends to do more harm than good. The tense and plurality of a word are necessary signals to its meaning. For the larger models used today, most standardization is as light as possible—for example, converting all inputs to a standard character encoding before further processing.

With standardization, we have now seen three distinct stages for preprocessing text (figure 14.1):

Figure 14.1 The text preprocessing pipeline

1 Standardization—Where we nor

malize input with basic text-to-text transformations 2 Splitting—Where we split our text into sequences of tokens 3 Indexing—Where we map our tokens to indices using a vocabulary People often refer to the entire process as tokenization, and to an object that maps text to sequence of token indices as a tokenizer. Let’s try building a few.

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

### PDF p.407 (printed 383) — 14 Text classification › 14.1 A brief history of natural language processing › 
In the last decade, NLP researchers and practitioners have discovered just how shockingly effective it can be to learn the answer to narrow statistical questions about text. In the 2010s, researchers began applying LSTM models to text, dramatically increasing the number of parameters in NLP models and the compute resources required to train them. The results were encouraging—LSTMs could generalize to unseen examples with far greater accuracy than previous approaches, but they eventually hit limits. LSTMs struggled to track dependencies in long chains of text with many sentences and paragraphs, and compared to computer vision models, they were slow and unwieldy to train. Toward the end of the 2010s, researchers at Google discovered a new architecture called the Transformer that solved many scalability issues plaguing LSTMs. As long as you increased the size of a model and its training data together, Transformers appeared to perform more and more accurately. Better yet, the computations needed for training a Transformer could be effectively parallelized, even for long sequences. If you doubled the number of machines doing training, you could roughly halve the time you need to wait fo
