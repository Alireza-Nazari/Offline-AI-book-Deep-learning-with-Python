# Generation bundle for page_415  (PDF page 415, printed page 391)

Page kind: content
Chapter: 14 Text classification
Section: 14.2 Preparing text data
Subsection: 14.2.2 Subword tokenization
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_415.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_415.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p415_e03] (body) This analogy between compression and tokenization was not always obvious, but it turns out to be powerful. One of the most practically effective tricks found in the last decade of NLP research was repurposing a 1990s algorithm for lossless compression called byte-pair encoding1 for tokenization. It is used by ChatGPT and many other models to this day. In this section, we will build a tokenizer that uses the byte-pair encoding algorithm. The idea with byte-pair encoding is to start with a basic vocabulary of characters and progressively “merge” common pairings into longer and longer sequences of characters. Let’s say we start with the following input text:
[p415_e04] (code)
```
data = [
    "the quick brown fox",
    "the slow brown fox",
    "the quick brown foxhound",
]
```
[p415_e05] (body) Like the `WordTokenizer`, we will start by computing word counts for all the words in the text. As we create our dictionary of word counts, we will split all our text into characters and join the characters with a space. This will make it easier to consider pairs of characters in our next step.
[p415_e06] (listing_caption) Listing 14.5 Initializing state for the byte-pair encoding algorithm
[p415_e07] (code)
```
def count_and_split_words(data):
    counts = collections.Counter()
    for line in data:
        line = line.lower()
        for word in re.findall(r"[\w]+|[.,!?;]", line):
            chars = re.findall(r".", word)
            split_word = " ".join(chars)
            counts[split_word] += 1
    return dict(counts)
counts = count_and_split_words(data)
```
[p415_e08] (body) Let’s try this out on our data:
[p415_e09] (code)
```
>>> counts
{"t h e": 3,
 "q u i c k": 2,
 "b r o w n": 3,
 "f o x": 2,
 "s l o w": 1,
 "f o x h o u n d": 1}
```
[p415_e10] (footnote) 1 Phillip Gage, “A New Algorithm for Data Compression,” The C Users Journal Archive (1994), https://dl.acm.org/ doi/10.5555/177910.177914.

## Listings
- Listing 14.5: Initializing state for the byte-pair encoding algorithm (code elements p415_e07)

## Equation candidates (formulas written inline in the text)
- [p415_e04] data = [
    "the quick brown fox",
    "the slow brown fox",
    "the quick brown foxhound",
]

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

## Previous page (PDF 414) text, for continuity
We can print out the same data for our word-level tokenizer:

We can already see some of the strengths and weaknesses of both tokenization techniques. A character-level tokenizer needs only 64 vocabulary terms to cover the entire book but will encode each input as a very long sequence. A word-level tokenizer quickly fills a 2,000-term vocabulary (you would need a dictionary with 17,000 terms to index every word in the book!), but the outputs of the word-level tokenizer are much shorter.

As machine learning practitioners have scaled models up with more and more data and parameters, the downsides of both word and character tokenization have become apparent. The “compression” offered by word-level tokenization turns out to be very important—it allows feeding longer sequences into a model. However, if you attempt to build a word-level vocabulary for a large dataset (today, you might see a dataset with trillions of words), you would have an unworkably large vocabulary with hundreds of millions of terms. If you aggressively restrict your word-level vocabulary size, you will encode a lot of text to the `"[UNK]"` token, throwing out valuable information. These issues have led to the rise in popularity of a third type of tokenization, called subword tokenization, which attempts to bridge the gap between word- and character-level approaches.

14.2.2 Subword tokenization

Subword tokenization aims to combine the best of both character- and word-level encoding techniques. We want the `WordTokenizer`’s ability to produce concise output and the `CharTokenizer`’s ability to encode a wide range of inputs with a small vocabulary.

We can think of the search for the ideal tokenizer as the hunt for an ideal compression of the input data. Reducing token length compresses the overall lengt …

## Next page (PDF 416) text, for continuity
To apply byte-pair encoding to our split word counts, we will find two characters and merge them into a new symbol. We consider all pairs of characters in all words and only merge the most common one we find. In the previous example, the most common character pair is `("o", "w")`, in both the word `"brown"` (which occurs three times in our data) and `"slow"` (which occurs once). We combine this pair into a new symbol `"ow"` and merge all occurrences of `"o w"`.

Then we continue, counting pairs and merging pairs, except now `"ow"` will be a single unit that could merge with, say, `"l"` to form `"low"`. By progressively merging the most frequent symbol pair, we build up a vocabulary of larger and larger subwords. Let’s try this out on our toy dataset.

Listing 14.6 Running a few steps of byte-pair merging

Here’s what we get:

We can see how common words are merged entirely, whereas less common words are only partially merged.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.417 — 14 Text classification › 14.2 Preparing text data
- PDF p.410 — 14 Text classification › 14.2 Preparing text data
- PDF p.409 — 14 Text classification › 14.2 Preparing text data
- PDF p.408 — 14 Text classification › 14.2 Preparing text data
- PDF p.413 — 14 Text classification › 14.2 Preparing text data
- PDF p.418 — 14 Text classification › 14.2 Preparing text data
- PDF p.429 — 14 Text classification › 14.5 Sequence models
- PDF p.423 — 14 Text classification › 14.4 Set models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.417 (printed 393) — 14 Text classification › 14.2 Preparing text data › 
We can now extend this to a full function for computing a byte-pair encoding vocabulary. We start our vocabulary with all characters found in the input text, and we will progressively add merged symbols (larger and larger subwords) to our vocabulary until it reaches our desired length. We also keep a separate dictionary of our merge rules, including a rank order in which we applied them. Next, we will see how to use these merge rules to tokenize new input text.

Listing 14.7 Computing a byte-pair encoding vocabulary

### PDF p.495 (printed 471) — 16 Text generation › 16.2 Training a mini-GPT › Running the code in this chapter
We will need to preprocess a lot of data to run pretraining for an LLM, even a miniature one like the one we are training. Using a fast tokenization routine to preprocess our source documents into integer tokens can simplify our lives.

We will use SentencePiece, a library for subword tokenization of text data. The actual tokenization technique is the same as the byte-pair encoding tokenization we built ourselves in chapter 14, but the library is written in C++ for speed and adds a `detokenize()` function that will reverse integers to strings and join them together. We will use a premade vocabulary with 32,000 vocabulary terms stored in a particular format needed by the SentencePiece library.

### PDF p.480 (printed 456) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
The `Tokenizer` maps from text to sequences of integers, as we would expect. Remember the `SubWordTokenizer` we built in the last chapter? RoBERTa’s tokenizer is almost the same as that tokenizer, with minor tweaks to handle Unicode characters from any language.

Given the size of RoBERTa’s pretraining dataset, subword tokenization is a must. Using a character-level tokenizer would make input sequences way too long, making the model much more expensive to train. Using a word-level tokenizer would require a massive vocabulary to attempt to cover all the distinct words in the millions of documents of text used from across the web. Getting good coverage of words would blow up our vocabulary size and make the `Embedding` layer at the front of the Transformer unworkably large. Using a subword tokenizer allows the model to handle any word with only a 50,000-term vocabulary:

### PDF p.417 (printed 393) — 14 Text classification › 14.2 Preparing text data › 
Listing 14.7 Computing a byte-pair encoding vocabulary
def compute_sub_word_vocabulary(dataset, vocab_size):
    counts = count_and_split_words(dataset)
    char_counts = collections.Counter()
    for word in counts:
        for char in word.split():
            char_counts[char] += counts[word]
    most_common = char_counts.most_common()
    vocab = ["[
<
<!\S)\S+ \S+(?!\S)", word, overlapped=True)
            if not pairs:
                break
            best = min(pairs, key=lambda pair: self.merges.get(pair, 1e9))
            if best not in self.merges:
                break
            first, second = best.split()
            split = re.compile(f"(?<!\S){first} {second}(?!\S)")
            merged = f"{first}{second}"
            word = split.sub(merged, word)
        return word
    def split(self, inputs):
        tokens = []
        for word in re.findall(r"[
            word=self.bpe_merge(word)
            tokens.extend(word.split())
        return tokens
    def index(self, tokens):
        return [self.vocabulary.get(t, self.unk_id) for t in tokens]
    def __call__(self, inputs):
        inputs = self.standardize(inputs)
        tokens = self.split(inputs)
        ind

### PDF p.418 (printed 394) — 14 Text classification › 14.2 Preparing text data › 
The `SubWordTokenizer` has a slightly longer length for our test sentence than the `Word` `Tokenizer` (16 versus 13 tokens), but unlike the `WordTokenizer`, it can tokenize every word in Moby Dick without using the `"[UNK]"` token. The vocabulary contains every character in our source text, so the worst-case performance will be tokenizing a word into individual characters. We have achieved a short average token length while handling rare words with a small vocabulary. This is the advantage of subword tokenizers.

You might notice that running this code is noticeably slower than the word and character tokenizers; it takes about a minute on our reference hardware. Learning merge rules is much more complex than simply counting the words in an input dataset. While this is a downside to subword tokenization, it is rarely an important concern in practice. You only need to learn a vocabulary once per model, and the cost of learning a subword vocabulary is generally negligible compared to model training. We have now seen three separate approaches for tokenizing input. Now that we can translate from text to numeric input, we can move on to training a model. One final note on tokenization—wh

### PDF p.419 (printed 395) — 14 Text classification › 14.3 Sets vs. sequences › 
tokenization used during pretraining or throw out the useful representations of input tokens contained in the model weights. If you are building a model from scratch, you can tailor your tokenization to the problem at hand. In general, the compression offered by word and subword tokenizers is too important to pass up. The shorter the length of your inputs on average, the better a model will be able to track long-range dependencies in the text, improving its overall performance. This has made subword the dominantly popular choice for modern language models. They can handle rare or misspelled words without inflating token length for common inputs. However, there is no one-size-fits-all approach. Some problems in NLP, such as spelling correction, might benefit from low-level character tokenization of input text. On the other hand, a word-level approach is both simple to work with and easily understandable—each model input corresponds to a word a human would read. This would make ranking tokens by importance to a prediction easy to interpret. We will use all three types of tokenizers throughout the text chapters of this book.

### PDF p.409 (printed 385) — 14 Text classification › 14.2 Preparing text data › 
Splitting takes us from a single string to a token sequence, but we still need to transform our string tokens into numeric inputs. By far the most common approach is to map each token to a unique integer index, often called indexing our input. This is a flexible and reversible representation of our tokenized input that can work with a wide range of modeling approaches. Later on, we can decide how to map from token indices into a latent space ingested by the model.

For character tokens, we could use ASCII lookups to index each token—for example, `ord('A') `→` 65` and `ord('z') `→` 122`. However, this can scale poorly when you start to consider other languages—there are over a million characters in the Unicode specification! A more robust technique is to build a mapping from specific tokens in our training data to indices that occur in the data we care about, which in NLP is called a vocabulary. This has the nice property of working for word-level tokens as easily as for character-level tokens. Let’s take a look at how we might use a vocabulary to transform text. We will build a simple Python dictionary that maps tokens to indices, split our input into tokens, and finally index our 

### PDF p.411 (printed 387) — 14 Text classification › 14.2 Preparing text data › 14.2.1 Character and word tokenization
To start, let’s build a character-level tokenizer that maps each character in an input string to an integer. To keep things simple, we will use only one standardization step— we lowercase all input.

Listing 14.1 A basic character-level tokenizer
