# Generation bundle for page_412  (PDF page 412, printed page 388)

Page kind: content
Chapter: 14 Text classification
Section: 14.2 Preparing text data
Subsection: 14.2.1 Character and word tokenization
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_412.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_412.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p412_e03] (code)
```
        indices = self.index(tokens)
        return indices
```
[p412_e04] (body) Pretty simple. Before using this, we also need to build a function that computes a vocabulary of tokens based on some input text. Rather than simply mapping all characters to a unique index, let’s give ourselves the ability to limit our vocabulary size to only the most common tokens in our input data. When we get into the modeling side of things, limiting the vocabulary size will be an important way to limit the number of parameters in a model.
[p412_e05] (listing_caption) Listing 14.2 Computing a character-level vocabulary
[p412_e06] (code)
```
import collections
def compute_char_vocabulary(inputs, max_size):
    char_counts = collections.Counter()
    for x in inputs:
        x = x.lower()
        tokens = re.findall(r".", x)
        char_counts.update(tokens)
    vocabulary = ["[UNK]"]
    most_common = char_counts.most_common(max_size - len(vocabulary))
    for token, count in most_common:
        vocabulary.append(token)
    return dict((token, i) for i, token in enumerate(vocabulary))
```
[p412_e07] (body) We can now do the same for a word-level tokenizer. We can use the same code as our character-level tokenizer with a different splitting step.
[p412_e08] (listing_caption) Listing 14.3 A basic word-level tokenizer
[p412_e09] (code)
```
class WordTokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.unk_id = vocabulary["[UNK]"]
    def standardize(self, inputs):
        return inputs.lower()
    def split(self, inputs):
        return re.findall(r"[\w]+|[.,!?;]", inputs)
    def index(self, tokens):
        return [self.vocabulary.get(t, self.unk_id) for t in tokens]
    def __call__(self, inputs):
        inputs = self.standardize(inputs)
        tokens = self.split(inputs)
```

## Listings
- Listing 14.2: Computing a character-level vocabulary (code elements p412_e06)
- Listing 14.3: A basic word-level tokenizer (code elements p412_e09)

## Equation candidates (formulas written inline in the text)
- [p412_e03]         indices = self.index(tokens)
        return indices

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

## Previous page (PDF 411) text, for continuity
as models have grown more expressive, this type of standardization tends to do more harm than good. The tense and plurality of a word are necessary signals to its meaning. For the larger models used today, most standardization is as light as possible—for example, converting all inputs to a standard character encoding before further processing.

With standardization, we have now seen three distinct stages for preprocessing text (figure 14.1):

Figure 14.1 The text preprocessing pipeline

1 Standardization—Where we nor

malize input with basic text-to-text transformations 2 Splitting—Where we split our text into sequences of tokens 3 Indexing—Where we map our tokens to indices using a vocabulary People often refer to the entire process as tokenization, and to an object that maps text to sequence of token indices as a tokenizer. Let’s try building a few.

14.2.1 Character and word tokenization

To start, let’s build a character-level tokenizer that maps each character in an input string to an integer. To keep things simple, we will use only one standardization step— we lowercase all input.

Listing 14.1 A basic character-level tokenizer

## Next page (PDF 413) text, for continuity
We can also substitute this new split rule into our vocabulary function.

Listing 14.4 Computing a word-level vocabulary

Let’s try out our tokenizers on some real-world input—the full text of Moby Dick by Herman Melville. We will first build a vocabulary for both tokenizers and then use it to tokenize some text:

Let’s inspect what our character-level tokenizer has computed:

Now what about the word-level tokenizer?

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.414 — 14 Text classification › 14.2 Preparing text data
- PDF p.417 — 14 Text classification › 14.2 Preparing text data
- PDF p.418 — 14 Text classification › 14.2 Preparing text data
- PDF p.409 — 14 Text classification › 14.2 Preparing text data
- PDF p.448 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.419 — 14 Text classification › 14.3 Sets vs. sequences
- PDF p.410 — 14 Text classification › 14.2 Preparing text data
- PDF p.480 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.414 (printed 390) — 14 Text classification › 14.2 Preparing text data › 
We can already see some of the strengths and weaknesses of both tokenization techniques. A character-level tokenizer needs only 64 vocabulary terms to cover the entire book but will encode each input as a very long sequence. A word-level tokenizer quickly fills a 2,000-term vocabulary (you would need a dictionary with 17,000 terms to index every word in the book!), but the outputs of the word-level tokenizer are much shorter.

As machine learning practitioners have scaled models up with more and more data and parameters, the downsides of both word and character tokenization have become apparent. The “compression” offered by word-level tokenization turns out to be very important—it allows feeding longer sequences into a model. However, if you attempt to build a word-level vocabulary for a large dataset (today, you might see a dataset with trillions of words), you would have an unworkably large vocabulary with hundreds of millions of terms. If you aggressively restrict your word-level vocabulary size, you will encode a lot of text to the `"[UNK]"` token, throwing out valuable information. These issues have led to the rise in popularity of a third type of tokenization, called subword 

### PDF p.409 (printed 385) — 14 Text classification › 14.2 Preparing text data › 
Splitting takes us from a single string to a token sequence, but we still need to transform our string tokens into numeric inputs. By far the most common approach is to map each token to a unique integer index, often called indexing our input. This is a flexible and reversible representation of our tokenized input that can work with a wide range of modeling approaches. Later on, we can decide how to map from token indices into a latent space ingested by the model.

For character tokens, we could use ASCII lookups to index each token—for example, `ord('A') `→` 65` and `ord('z') `→` 122`. However, this can scale poorly when you start to consider other languages—there are over a million characters in the Unicode specification! A more robust technique is to build a mapping from specific tokens in our training data to indices that occur in the data we care about, which in NLP is called a vocabulary. This has the nice property of working for word-level tokens as easily as for character-level tokens. Let’s take a look at how we might use a vocabulary to transform text. We will build a simple Python dictionary that maps tokens to indices, split our input into tokens, and finally index our 

### PDF p.480 (printed 456) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
The `Tokenizer` maps from text to sequences of integers, as we would expect. Remember the `SubWordTokenizer` we built in the last chapter? RoBERTa’s tokenizer is almost the same as that tokenizer, with minor tweaks to handle Unicode characters from any language.

Given the size of RoBERTa’s pretraining dataset, subword tokenization is a must. Using a character-level tokenizer would make input sequences way too long, making the model much more expensive to train. Using a word-level tokenizer would require a massive vocabulary to attempt to cover all the distinct words in the millions of documents of text used from across the web. Getting good coverage of words would blow up our vocabulary size and make the `Embedding` layer at the front of the Transformer unworkably large. Using a subword tokenizer allows the model to handle any word with only a 50,000-term vocabulary:

### PDF p.414 (printed 390) — 14 Text classification › 14.2 Preparing text data › 14.2.2 Subword tokenization
Subword tokenization aims to combine the best of both character- and word-level encoding techniques. We want the `WordTokenizer`’s ability to produce concise output and the `CharTokenizer`’s ability to encode a wide range of inputs with a small vocabulary.

We can think of the search for the ideal tokenizer as the hunt for an ideal compression of the input data. Reducing token length compresses the overall length of our examples. A small vocabulary reduces the number of bytes we would need to represent each token. If we achieve both, we will be able to feed short, information-rich sequences to our deep learning model.

### PDF p.419 (printed 395) — 14 Text classification › 14.3 Sets vs. sequences › 
tokenization used during pretraining or throw out the useful representations of input tokens contained in the model weights. If you are building a model from scratch, you can tailor your tokenization to the problem at hand. In general, the compression offered by word and subword tokenizers is too important to pass up. The shorter the length of your inputs on average, the better a model will be able to track long-range dependencies in the text, improving its overall performance. This has made subword the dominantly popular choice for modern language models. They can handle rare or misspelled words without inflating token length for common inputs. However, there is no one-size-fits-all approach. Some problems in NLP, such as spelling correction, might benefit from low-level character tokenization of input text. On the other hand, a word-level approach is both simple to work with and easily understandable—each model input corresponds to a word a human would read. This would make ranking tokens by importance to a prediction easy to interpret. We will use all three types of tokenizers throughout the text chapters of this book.

### PDF p.418 (printed 394) — 14 Text classification › 14.2 Preparing text data › 
The `SubWordTokenizer` has a slightly longer length for our test sentence than the `Word` `Tokenizer` (16 versus 13 tokens), but unlike the `WordTokenizer`, it can tokenize every word in Moby Dick without using the `"[UNK]"` token. The vocabulary contains every character in our source text, so the worst-case performance will be tokenizing a word into individual characters. We have achieved a short average token length while handling rare words with a small vocabulary. This is the advantage of subword tokenizers.

You might notice that running this code is noticeably slower than the word and character tokenizers; it takes about a minute on our reference hardware. Learning merge rules is much more complex than simply counting the words in an input dataset. While this is a downside to subword tokenization, it is rarely an important concern in practice. You only need to learn a vocabulary once per model, and the cost of learning a subword vocabulary is generally negligible compared to model training. We have now seen three separate approaches for tokenizing input. Now that we can translate from text to numeric input, we can move on to training a model. One final note on tokenization—wh

### PDF p.447 (printed 423) — 15 Language models and the Transformer › 15.1 The language model › 
To build a language model from this input, we will need to massage our source text. First, we will split our data into equal-length chunks that we can batch and use for model training, much as we did for weather measurements in the timeseries chapters. Because we will be using a character-level tokenizer here, we can do this chunking directly on the string input. A 100-character string will map to a 100-integer sequence.

We will also split each input into two separate feature and label sequences, with each label sequence simply being the input sequence offset by a single character.

Listing 15.2 Splitting text into chunks for language model training

### PDF p.417 (printed 393) — 14 Text classification › 14.2 Preparing text data › 
We can now extend this to a full function for computing a byte-pair encoding vocabulary. We start our vocabulary with all characters found in the input text, and we will progressively add merged symbols (larger and larger subwords) to our vocabulary until it reaches our desired length. We also keep a separate dictionary of our merge rules, including a rank order in which we applied them. Next, we will see how to use these merge rules to tokenize new input text.

Listing 14.7 Computing a byte-pair encoding vocabulary
