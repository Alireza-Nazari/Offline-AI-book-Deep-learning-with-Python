# Generation bundle for page_410  (PDF page 410, printed page 386)

Page kind: content
Chapter: 14 Text classification
Section: 14.2 Preparing text data
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_410.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_410.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p410_e03] (code)
```
    "over": 6,
    "dog": 7,
    ".": 8,
}
words = split_words("The quick brown fox jumped over the lazy dog.")
indices = [vocabulary.get(word, 0) for word in words]
```
[p410_e04] (body) This outputs the following:
[p410_e05] (code)
```
[0, 2, 3, 4, 5, 6, 1, 0, 7, 8]
```
[p410_e06] (body) We introduce a special token called `"[UNK]"` to our vocabulary, which represents a token that is unknown to the vocabulary. This way, we can index all input we come across, even if some terms only occur in our test data. In the previous example `"lazy"` maps to the `"[UNK]"` index 0, as it was not included in our vocabulary.
[p410_e07] (body) With these simple text transformations, we are well on our way to building a text preprocessing pipeline. However, there is one more common type of text manipulation we should consider—standardization. Consider these two sentences:
[p410_e08] (bullet) “sunset came. i was staring at the Mexico sky. Isnt nature splendid??”
[p410_e09] (bullet) “Sunset came; I stared at the México sky. Isn’t nature splendid?”
[p410_e10] (body) They are very similar—in fact, they are almost identical. Yet, if you were to convert them to indices as previously described, you would end up with very different representations because “i” and “I” are two distinct characters, “Mexico” and “México” are two distinct words, “isnt” isn’t “isn’t,” and so on. A machine learning model doesn’t know a priori that “i” and “I” are the same letter, that “é” is an “e” with an accent, or that “staring” and “stared” are two forms of the same verb. Standardizing text is a basic form of feature engineering that aims to erase encoding differences that you don’t want your model to have to deal with. It’s not exclusive to machine learning, either—you’d have to do the same thing if you were building a search engine.
[p410_e11] (body) One simple and widespread standardization scheme is to convert to lowercase and remove punctuation characters. Our two sentences would become
[p410_e12] (bullet) “sunset came i was staring at the mexico sky isnt nature splendid”
[p410_e13] (bullet) “sunset came i stared at the méxico sky isnt nature splendid”
[p410_e14] (body) Much closer already. We could get even closer if we removed accent marks on all characters.
[p410_e15] (body) There’s a lot you can do with standardization, and it used to be one of the most critical areas to improve model performance. For many decades in NLP, it was common practice to use regular expressions to attempt to map words to a common root (e.g. “tired” → “tire” and “trophies” → “trophy”), called stemming or lemmatization. But

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

## Previous page (PDF 409) text, for continuity
Regex can easily be applied to split our text into words instead. The `"[\w]+"` regular expression will grab consecutive non-whitespace characters, and the `"[.,!?;]"` can match the punctuation marks between the brackets. We can combine the two to achieve a regular expression that splits each word and punctuation mark into a token:

Here’s what it does to a test sentence:

Splitting takes us from a single string to a token sequence, but we still need to transform our string tokens into numeric inputs. By far the most common approach is to map each token to a unique integer index, often called indexing our input. This is a flexible and reversible representation of our tokenized input that can work with a wide range of modeling approaches. Later on, we can decide how to map from token indices into a latent space ingested by the model.

For character tokens, we could use ASCII lookups to index each token—for example, `ord('A') `→` 65` and `ord('z') `→` 122`. However, this can scale poorly when you start to consider other languages—there are over a million characters in the Unicode specification! A more robust technique is to build a mapping from specific tokens in our training data to indices that occur in the data we care about, which in NLP is called a vocabulary. This has the nice property of working for word-level tokens as easily as for character-level tokens. Let’s take a look at how we might use a vocabulary to transform text. We will build a simple Python dictionary that maps tokens to indices, split our input into tokens, and finally index our tokens:

## Next page (PDF 411) text, for continuity
as models have grown more expressive, this type of standardization tends to do more harm than good. The tense and plurality of a word are necessary signals to its meaning. For the larger models used today, most standardization is as light as possible—for example, converting all inputs to a standard character encoding before further processing.

With standardization, we have now seen three distinct stages for preprocessing text (figure 14.1):

Figure 14.1 The text preprocessing pipeline

1 Standardization—Where we nor

malize input with basic text-to-text transformations 2 Splitting—Where we split our text into sequences of tokens 3 Indexing—Where we map our tokens to indices using a vocabulary People often refer to the entire process as tokenization, and to an object that maps text to sequence of token indices as a tokenizer. Let’s try building a few.

14.2.1 Character and word tokenization

To start, let’s build a character-level tokenizer that maps each character in an input string to an integer. To keep things simple, we will use only one standardization step— we lowercase all input.

Listing 14.1 A basic character-level tokenizer

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.408 — 14 Text classification › 14.2 Preparing text data
- PDF p.416 — 14 Text classification › 14.2 Preparing text data
- PDF p.415 — 14 Text classification › 14.2 Preparing text data
- PDF p.414 — 14 Text classification › 14.2 Preparing text data
- PDF p.412 — 14 Text classification › 14.2 Preparing text data
- PDF p.418 — 14 Text classification › 14.2 Preparing text data
- PDF p.422 — 14 Text classification › 14.4 Set models
- PDF p.429 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.455 (printed 431) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
We need to preserve the `"[start]"` and `"[end]"` tokens that we’ve inserted. By default, the characters `[` and `]` would be stripped, but we want to keep them around so we can distinguish the word `"start"` from the start token `"[start]"`.

Punctuation is different from language to language! In the Spanish `Text` `Vectorization` layer, if we’re going to strip punctuation characters, we need to also strip the character ¿.

Note that for a non-toy translation model, we would treat punctuation characters as separate tokens rather than stripping them since we would want to be able to generate correctly punctuated sentences. In our case, for simplicity, we’ll get rid of all punctuation.

Listing 15.9 Learning token vocabularies for English and Spanish text

### PDF p.408 (printed 384) — 14 Text classification › 14.2 Preparing text data › 14.2 Preparing text data
There is an obvious blocker before we can start applying any of the deep learning techniques of previous chapters—our input is not numeric! Before beginning any modeling, we need to translate the written word into tensors of numbers. Unlike images, which have a relatively natural numeric representation, you could build a numeric representation of text in several ways.

A simple approach would be to borrow from standard text file formats for text and use something like an ASCII encoding. We could chop the input into a sequence of characters and assign each a unique index. Another intuitive approach would be building a representation based on words, first breaking sentences apart on all spaces and punctuation and then mapping each word to a unique numeric representation. These are both good approaches to try, and in general, all text preprocessing will include a splitting step, where text is split into small individual units, called tokens. A powerful tool for splitting text is regular expressions, which can flexibly match patterns of characters in text. Let’s look at how to use a regular expression to split a string into a sequence of characters. The most basic regex we can apply is

### PDF p.412 (printed 388) — 14 Text classification › 14.2 Preparing text data › 
Pretty simple. Before using this, we also need to build a function that computes a vocabulary of tokens based on some input text. Rather than simply mapping all characters to a unique index, let’s give ourselves the ability to limit our vocabulary size to only the most common tokens in our input data. When we get into the modeling side of things, limiting the vocabulary size will be an important way to limit the number of parameters in a model.

Listing 14.2 Computing a character-level vocabulary

### PDF p.443 (printed 419) — 14 Text classification › 14.5 Sequence models › Summary
All text modeling problems involve a preprocessing step where text is broken up and transformed to integer data, called tokenization.

Tokenization can be divided into three steps: standardization, splitting, and indexing. Standardization normalizes text, splitting breaks text up into tokens, and indexing assigns each token a unique integer ID.

There are three main types of tokenization: character, word, and subword tokenization. With an expressive-enough model and sufficient training data, the subword tokenization is usually the most effective.

NLP models differ primarily in handling the order of input tokens: – Set models discard most order information and learn simple and fast models based solely on the presence or absence of tokens in the input. Bigram or trigram models consider the presence or absence of two or three consecutive tokens. Set models are incredibly fast to train and deploy. – Sequence models attempt to learn with the ordered sequence of tokens in the input data. Sequence models need large amounts of data to learn effectively.

### PDF p.480 (printed 456) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
The `Tokenizer` maps from text to sequences of integers, as we would expect. Remember the `SubWordTokenizer` we built in the last chapter? RoBERTa’s tokenizer is almost the same as that tokenizer, with minor tweaks to handle Unicode characters from any language.

Given the size of RoBERTa’s pretraining dataset, subword tokenization is a must. Using a character-level tokenizer would make input sequences way too long, making the model much more expensive to train. Using a word-level tokenizer would require a massive vocabulary to attempt to cover all the distinct words in the millions of documents of text used from across the web. Getting good coverage of words would blow up our vocabulary size and make the `Embedding` layer at the front of the Transformer unworkably large. Using a subword tokenizer allows the model to handle any word with only a 50,000-term vocabulary:

### PDF p.427 (printed 403) — 14 Text classification › 14.4 Set models › 14.4.2 Training a bigram model
Of course, we can intuitively guess that discarding all word order is very reductive because even atomic concepts can be expressed via multiple words: the term “United States” conveys a concept that is quite distinct from the meaning of the words “states” and “united” taken separately. A movie that is “not bad” and a movie that is “bad” should probably get different sentiment scores.

Therefore, it is usually a good idea to inject some knowledge of local word ordering into a model, even for these simple set-based models we are currently building. One easy way to do that is to consider bigrams—a term for two tokens that appear consecutively in the input text. Given our example “this movie made me cry,” `{"this",` `"movie",` `"made",` `"me",` `"cry"}` is the set of all word unigrams in the input, and `{"this` `movie",` `"movie` `made",` `"made me",` `"me cry"}` is the set of all bigrams. The bag-of-words model we just trained could equivalently be called a unigram model, and the term n-gram refers to an ordered sequence of n tokens for any n. To add bigrams to our model, we want to consider the frequency of all bigrams while building our vocabulary. We could do this in two ways: by c

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences

### PDF p.417 (printed 393) — 14 Text classification › 14.2 Preparing text data › 
We can now extend this to a full function for computing a byte-pair encoding vocabulary. We start our vocabulary with all characters found in the input text, and we will progressively add merged symbols (larger and larger subwords) to our vocabulary until it reaches our desired length. We also keep a separate dictionary of our merge rules, including a rank order in which we applied them. Next, we will see how to use these merge rules to tokenize new input text.

Listing 14.7 Computing a byte-pair encoding vocabulary
