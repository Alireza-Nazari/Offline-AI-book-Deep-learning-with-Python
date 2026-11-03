# Generation bundle for page_414  (PDF page 414, printed page 390)

Page kind: content
Chapter: 14 Text classification
Section: 14.2 Preparing text data
Subsection: 14.2.2 Subword tokenization
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_414.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_414.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p414_e03] (code)
```
vocabulary = compute_word_vocabulary(moby_dick, max_size=2_000)
word_tokenizer = WordTokenizer(vocabulary)
```
[p414_e04] (body) We can print out the same data for our word-level tokenizer:
[p414_e05] (code)
```
>>> print("Vocabulary length:", len(vocabulary))
Vocabulary length: 2000
>>> print("Vocabulary start:", list(vocabulary.keys())[:5])
Vocabulary start: ["[UNK]", ",", "the", ".", "of"]
>>> print("Vocabulary end:", list(vocabulary.keys())[-5:])
Vocabulary end: ["tambourine", "subtle", "perseus", "elevated", "repose"]
>>> print("Line length:", len(word_tokenizer(
...    "Call me Ishmael. Some years ago--never mind how long precisely."
... )))
Line length: 13
```
[p414_e06] (body) We can already see some of the strengths and weaknesses of both tokenization techniques. A character-level tokenizer needs only 64 vocabulary terms to cover the entire book but will encode each input as a very long sequence. A word-level tokenizer quickly fills a 2,000-term vocabulary (you would need a dictionary with 17,000 terms to index every word in the book!), but the outputs of the word-level tokenizer are much shorter.
[p414_e07] (body) As machine learning practitioners have scaled models up with more and more data and parameters, the downsides of both word and character tokenization have become apparent. The “compression” offered by word-level tokenization turns out to be very important—it allows feeding longer sequences into a model. However, if you attempt to build a word-level vocabulary for a large dataset (today, you might see a dataset with trillions of words), you would have an unworkably large vocabulary with hundreds of millions of terms. If you aggressively restrict your word-level vocabulary size, you will encode a lot of text to the `"[UNK]"` token, throwing out valuable information. These issues have led to the rise in popularity of a third type of tokenization, called subword tokenization, which attempts to bridge the gap between word- and character-level approaches.
[p414_e08] (heading_subsection) 14.2.2 Subword tokenization
[p414_e09] (body) Subword tokenization aims to combine the best of both character- and word-level encoding techniques. We want the `WordTokenizer`’s ability to produce concise output and the `CharTokenizer`’s ability to encode a wide range of inputs with a small vocabulary.
[p414_e10] (body) We can think of the search for the ideal tokenizer as the hunt for an ideal compression of the input data. Reducing token length compresses the overall length of our examples. A small vocabulary reduces the number of bytes we would need to represent each token. If we achieve both, we will be able to feed short, information-rich sequences to our deep learning model.

## Equation candidates (formulas written inline in the text)
- [p414_e03] vocabulary = compute_word_vocabulary(moby_dick, max_size=2_000)
word_tokenizer = WordTokenizer(vocabulary)

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

## Previous page (PDF 413) text, for continuity
We can also substitute this new split rule into our vocabulary function.

Listing 14.4 Computing a word-level vocabulary

Let’s try out our tokenizers on some real-world input—the full text of Moby Dick by Herman Melville. We will first build a vocabulary for both tokenizers and then use it to tokenize some text:

Let’s inspect what our character-level tokenizer has computed:

Now what about the word-level tokenizer?

## Next page (PDF 415) text, for continuity
This analogy between compression and tokenization was not always obvious, but it turns out to be powerful. One of the most practically effective tricks found in the last decade of NLP research was repurposing a 1990s algorithm for lossless compression called byte-pair encoding1 for tokenization. It is used by ChatGPT and many other models to this day. In this section, we will build a tokenizer that uses the byte-pair encoding algorithm. The idea with byte-pair encoding is to start with a basic vocabulary of characters and progressively “merge” common pairings into longer and longer sequences of characters. Let’s say we start with the following input text:

Like the `WordTokenizer`, we will start by computing word counts for all the words in the text. As we create our dictionary of word counts, we will split all our text into characters and join the characters with a space. This will make it easier to consider pairs of characters in our next step.

Listing 14.5 Initializing state for the byte-pair encoding algorithm

Let’s try this out on our data:

1 Phillip Gage, “A New Algorithm for Data Compression,” The C Users Journal Archive (1994), https://dl.acm.org/ doi/10.5555/177910.177914.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.418 — 14 Text classification › 14.2 Preparing text data
- PDF p.412 — 14 Text classification › 14.2 Preparing text data
- PDF p.411 — 14 Text classification › 14.2 Preparing text data
- PDF p.448 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.426 — 14 Text classification › 14.4 Set models
- PDF p.417 — 14 Text classification › 14.2 Preparing text data
- PDF p.409 — 14 Text classification › 14.2 Preparing text data
- PDF p.419 — 14 Text classification › 14.3 Sets vs. sequences

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.480 (printed 456) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
The `Tokenizer` maps from text to sequences of integers, as we would expect. Remember the `SubWordTokenizer` we built in the last chapter? RoBERTa’s tokenizer is almost the same as that tokenizer, with minor tweaks to handle Unicode characters from any language.

Given the size of RoBERTa’s pretraining dataset, subword tokenization is a must. Using a character-level tokenizer would make input sequences way too long, making the model much more expensive to train. Using a word-level tokenizer would require a massive vocabulary to attempt to cover all the distinct words in the millions of documents of text used from across the web. Getting good coverage of words would blow up our vocabulary size and make the `Embedding` layer at the front of the Transformer unworkably large. Using a subword tokenizer allows the model to handle any word with only a 50,000-term vocabulary:

### PDF p.419 (printed 395) — 14 Text classification › 14.3 Sets vs. sequences › 
tokenization used during pretraining or throw out the useful representations of input tokens contained in the model weights. If you are building a model from scratch, you can tailor your tokenization to the problem at hand. In general, the compression offered by word and subword tokenizers is too important to pass up. The shorter the length of your inputs on average, the better a model will be able to track long-range dependencies in the text, improving its overall performance. This has made subword the dominantly popular choice for modern language models. They can handle rare or misspelled words without inflating token length for common inputs. However, there is no one-size-fits-all approach. Some problems in NLP, such as spelling correction, might benefit from low-level character tokenization of input text. On the other hand, a word-level approach is both simple to work with and easily understandable—each model input corresponds to a word a human would read. This would make ranking tokens by importance to a prediction easy to interpret. We will use all three types of tokenizers throughout the text chapters of this book.

### PDF p.418 (printed 394) — 14 Text classification › 14.2 Preparing text data › 
The `SubWordTokenizer` has a slightly longer length for our test sentence than the `Word` `Tokenizer` (16 versus 13 tokens), but unlike the `WordTokenizer`, it can tokenize every word in Moby Dick without using the `"[UNK]"` token. The vocabulary contains every character in our source text, so the worst-case performance will be tokenizing a word into individual characters. We have achieved a short average token length while handling rare words with a small vocabulary. This is the advantage of subword tokenizers.

You might notice that running this code is noticeably slower than the word and character tokenizers; it takes about a minute on our reference hardware. Learning merge rules is much more complex than simply counting the words in an input dataset. While this is a downside to subword tokenization, it is rarely an important concern in practice. You only need to learn a vocabulary once per model, and the cost of learning a subword vocabulary is generally negligible compared to model training. We have now seen three separate approaches for tokenizing input. Now that we can translate from text to numeric input, we can move on to training a model. One final note on tokenization—wh

### PDF p.411 (printed 387) — 14 Text classification › 14.2 Preparing text data › 14.2.1 Character and word tokenization
To start, let’s build a character-level tokenizer that maps each character in an input string to an integer. To keep things simple, we will use only one standardization step— we lowercase all input.

Listing 14.1 A basic character-level tokenizer

### PDF p.411 (printed 387) — 14 Text classification › 14.2 Preparing text data › 14.2.1 Character and word tokenization
Listing 14.1 A basic character-level tokenizer
class CharTokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.unk_id = vocabulary["[UNK]"]
    def standardize(self, inputs):
        return inputs.lower()
    def split(self, inputs):
        return re.findall(r".", inputs)
    def index(self, tokens):
        return [self.vocabulary.get(t, self.unk_id) for t in tokens]
    def __call__(self, inputs):
        inputs = self.standardize(inputs)
        tokens = self.split(inputs)

### PDF p.412 (printed 388) — 14 Text classification › 14.2 Preparing text data › 
We can now do the same for a word-level tokenizer. We can use the same code as our character-level tokenizer with a different splitting step.

Listing 14.3 A basic word-level tokenizer

### PDF p.443 (printed 419) — 14 Text classification › 14.5 Sequence models › Summary
All text modeling problems involve a preprocessing step where text is broken up and transformed to integer data, called tokenization.

Tokenization can be divided into three steps: standardization, splitting, and indexing. Standardization normalizes text, splitting breaks text up into tokens, and indexing assigns each token a unique integer ID.

There are three main types of tokenization: character, word, and subword tokenization. With an expressive-enough model and sufficient training data, the subword tokenization is usually the most effective.

NLP models differ primarily in handling the order of input tokens: – Set models discard most order information and learn simple and fast models based solely on the presence or absence of tokens in the input. Bigram or trigram models consider the presence or absence of two or three consecutive tokens. Set models are incredibly fast to train and deploy. – Sequence models attempt to learn with the ordered sequence of tokens in the input data. Sequence models need large amounts of data to learn effectively.

### PDF p.495 (printed 471) — 16 Text generation › 16.2 Training a mini-GPT › Running the code in this chapter
We will need to preprocess a lot of data to run pretraining for an LLM, even a miniature one like the one we are training. Using a fast tokenization routine to preprocess our source documents into integer tokens can simplify our lives.

We will use SentencePiece, a library for subword tokenization of text data. The actual tokenization technique is the same as the byte-pair encoding tokenization we built ourselves in chapter 14, but the library is written in C++ for speed and adds a `detokenize()` function that will reverse integers to strings and join them together. We will use a premade vocabulary with 32,000 vocabulary terms stored in a particular format needed by the SentencePiece library.
