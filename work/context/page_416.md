# Generation bundle for page_416  (PDF page 416, printed page 392)

Page kind: content
Chapter: 14 Text classification
Section: 14.2 Preparing text data
Subsection: 14.2.2 Subword tokenization
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_416.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_416.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p416_e03] (body) To apply byte-pair encoding to our split word counts, we will find two characters and merge them into a new symbol. We consider all pairs of characters in all words and only merge the most common one we find. In the previous example, the most common character pair is `("o", "w")`, in both the word `"brown"` (which occurs three times in our data) and `"slow"` (which occurs once). We combine this pair into a new symbol `"ow"` and merge all occurrences of `"o w"`.
[p416_e04] (body) Then we continue, counting pairs and merging pairs, except now `"ow"` will be a single unit that could merge with, say, `"l"` to form `"low"`. By progressively merging the most frequent symbol pair, we build up a vocabulary of larger and larger subwords. Let’s try this out on our toy dataset.
[p416_e05] (listing_caption) Listing 14.6 Running a few steps of byte-pair merging
[p416_e06] (code)
```
def count_pairs(counts):
    pairs = collections.Counter()
    for word, freq in counts.items():
        symbols = word.split()
        for pair in zip(symbols[:-1], symbols[1:]):
            pairs[pair] += freq
    return pairs
def merge_pair(counts, first, second):
    split = re.compile(f"(?<!\S){first} {second}(?!\S)")
    merged = f"{first}{second}"
    return {split.sub(merged, word): count for word, count in counts.items()}
for i in range(10):
    pairs = count_pairs(counts)
    first, second = max(pairs, key=pairs.get)
    counts = merge_pair(counts, first, second)
    print(list(counts.keys()))
```
[p416_e07] (body) Here’s what we get:
[p416_e08] (code)
```
["t h e", "q u i c k", "b r ow n", "f o x", "s l ow", "f o x h o u n d"]
["th e", "q u i c k", "b r ow n", "f o x", "s l ow", "f o x h o u n d"]
["the", "q u i c k", "b r ow n", "f o x", "s l ow", "f o x h o u n d"]
["the", "q u i c k", "br ow n", "f o x", "s l ow", "f o x h o u n d"]
["the", "q u i c k", "brow n", "f o x", "s l ow", "f o x h o u n d"]
["the", "q u i c k", "brown", "f o x", "s l ow", "f o x h o u n d"]
["the", "q u i c k", "brown", "fo x", "s l ow", "fo x h o u n d"]
["the", "q u i c k", "brown", "fox", "s l ow", "fox h o u n d"]
["the", "qu i c k", "brown", "fox", "s l ow", "fox h o u n d"]
["the", "qui c k", "brown", "fox", "s l ow", "fox h o u n d"]
```
[p416_e09] (body) We can see how common words are merged entirely, whereas less common words are only partially merged.

## Code annotations printed next to listings (callouts, in order)
- Matches an
- unmerged pair
- Replaces all occurances
- with a merged version

## Listings
- Listing 14.6: Running a few steps of byte-pair merging (code elements p416_e06)

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

## Previous page (PDF 415) text, for continuity
This analogy between compression and tokenization was not always obvious, but it turns out to be powerful. One of the most practically effective tricks found in the last decade of NLP research was repurposing a 1990s algorithm for lossless compression called byte-pair encoding1 for tokenization. It is used by ChatGPT and many other models to this day. In this section, we will build a tokenizer that uses the byte-pair encoding algorithm. The idea with byte-pair encoding is to start with a basic vocabulary of characters and progressively “merge” common pairings into longer and longer sequences of characters. Let’s say we start with the following input text:

Like the `WordTokenizer`, we will start by computing word counts for all the words in the text. As we create our dictionary of word counts, we will split all our text into characters and join the characters with a space. This will make it easier to consider pairs of characters in our next step.

Listing 14.5 Initializing state for the byte-pair encoding algorithm

Let’s try this out on our data:

1 Phillip Gage, “A New Algorithm for Data Compression,” The C Users Journal Archive (1994), https://dl.acm.org/ doi/10.5555/177910.177914.

## Next page (PDF 417) text, for continuity
We can now extend this to a full function for computing a byte-pair encoding vocabulary. We start our vocabulary with all characters found in the input text, and we will progressively add merged symbols (larger and larger subwords) to our vocabulary until it reaches our desired length. We also keep a separate dictionary of our merge rules, including a rank order in which we applied them. Next, we will see how to use these merge rules to tokenize new input text.

Listing 14.7 Computing a byte-pair encoding vocabulary

Let’s try out our tokenizer on the full text of Moby Dick:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.410 — 14 Text classification › 14.2 Preparing text data
- PDF p.409 — 14 Text classification › 14.2 Preparing text data
- PDF p.408 — 14 Text classification › 14.2 Preparing text data
- PDF p.429 — 14 Text classification › 14.5 Sequence models
- PDF p.413 — 14 Text classification › 14.2 Preparing text data
- PDF p.418 — 14 Text classification › 14.2 Preparing text data
- PDF p.496 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.412 — 14 Text classification › 14.2 Preparing text data

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.418 (printed 394) — 14 Text classification › 14.2 Preparing text data › 
The `SubWordTokenizer` has a slightly longer length for our test sentence than the `Word` `Tokenizer` (16 versus 13 tokens), but unlike the `WordTokenizer`, it can tokenize every word in Moby Dick without using the `"[UNK]"` token. The vocabulary contains every character in our source text, so the worst-case performance will be tokenizing a word into individual characters. We have achieved a short average token length while handling rare words with a small vocabulary. This is the advantage of subword tokenizers.

You might notice that running this code is noticeably slower than the word and character tokenizers; it takes about a minute on our reference hardware. Learning merge rules is much more complex than simply counting the words in an input dataset. While this is a downside to subword tokenization, it is rarely an important concern in practice. You only need to learn a vocabulary once per model, and the cost of learning a subword vocabulary is generally negligible compared to model training. We have now seen three separate approaches for tokenizing input. Now that we can translate from text to numeric input, we can move on to training a model. One final note on tokenization—wh

### PDF p.495 (printed 471) — 16 Text generation › 16.2 Training a mini-GPT › Running the code in this chapter
We will need to preprocess a lot of data to run pretraining for an LLM, even a miniature one like the one we are training. Using a fast tokenization routine to preprocess our source documents into integer tokens can simplify our lives.

We will use SentencePiece, a library for subword tokenization of text data. The actual tokenization technique is the same as the byte-pair encoding tokenization we built ourselves in chapter 14, but the library is written in C++ for speed and adds a `detokenize()` function that will reverse integers to strings and join them together. We will use a premade vocabulary with 32,000 vocabulary terms stored in a particular format needed by the SentencePiece library.

### PDF p.414 (printed 390) — 14 Text classification › 14.2 Preparing text data › 
We can already see some of the strengths and weaknesses of both tokenization techniques. A character-level tokenizer needs only 64 vocabulary terms to cover the entire book but will encode each input as a very long sequence. A word-level tokenizer quickly fills a 2,000-term vocabulary (you would need a dictionary with 17,000 terms to index every word in the book!), but the outputs of the word-level tokenizer are much shorter.

As machine learning practitioners have scaled models up with more and more data and parameters, the downsides of both word and character tokenization have become apparent. The “compression” offered by word-level tokenization turns out to be very important—it allows feeding longer sequences into a model. However, if you attempt to build a word-level vocabulary for a large dataset (today, you might see a dataset with trillions of words), you would have an unworkably large vocabulary with hundreds of millions of terms. If you aggressively restrict your word-level vocabulary size, you will encode a lot of text to the `"[UNK]"` token, throwing out valuable information. These issues have led to the rise in popularity of a third type of tokenization, called subword 

### PDF p.480 (printed 456) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
The `Tokenizer` maps from text to sequences of integers, as we would expect. Remember the `SubWordTokenizer` we built in the last chapter? RoBERTa’s tokenizer is almost the same as that tokenizer, with minor tweaks to handle Unicode characters from any language.

Given the size of RoBERTa’s pretraining dataset, subword tokenization is a must. Using a character-level tokenizer would make input sequences way too long, making the model much more expensive to train. Using a word-level tokenizer would require a massive vocabulary to attempt to cover all the distinct words in the millions of documents of text used from across the web. Getting good coverage of words would blow up our vocabulary size and make the `Embedding` layer at the front of the Transformer unworkably large. Using a subword tokenizer allows the model to handle any word with only a 50,000-term vocabulary:

### PDF p.409 (printed 385) — 14 Text classification › 14.2 Preparing text data › 
Splitting takes us from a single string to a token sequence, but we still need to transform our string tokens into numeric inputs. By far the most common approach is to map each token to a unique integer index, often called indexing our input. This is a flexible and reversible representation of our tokenized input that can work with a wide range of modeling approaches. Later on, we can decide how to map from token indices into a latent space ingested by the model.

For character tokens, we could use ASCII lookups to index each token—for example, `ord('A') `→` 65` and `ord('z') `→` 122`. However, this can scale poorly when you start to consider other languages—there are over a million characters in the Unicode specification! A more robust technique is to build a mapping from specific tokens in our training data to indices that occur in the data we care about, which in NLP is called a vocabulary. This has the nice property of working for word-level tokens as easily as for character-level tokens. Let’s take a look at how we might use a vocabulary to transform text. We will build a simple Python dictionary that maps tokens to indices, split our input into tokens, and finally index our 

### PDF p.414 (printed 390) — 14 Text classification › 14.2 Preparing text data › 14.2.2 Subword tokenization
Subword tokenization aims to combine the best of both character- and word-level encoding techniques. We want the `WordTokenizer`’s ability to produce concise output and the `CharTokenizer`’s ability to encode a wide range of inputs with a small vocabulary.

We can think of the search for the ideal tokenizer as the hunt for an ideal compression of the input data. Reducing token length compresses the overall length of our examples. A small vocabulary reduces the number of bytes we would need to represent each token. If we achieve both, we will be able to feed short, information-rich sequences to our deep learning model.

### PDF p.409 (printed 385) — 14 Text classification › 14.2 Preparing text data › 
Regex can easily be applied to split our text into words instead. The `"[\w]+"` regular expression will grab consecutive non-whitespace characters, and the `"[.,!?;]"` can match the punctuation marks between the brackets. We can combine the two to achieve a regular expression that splits each word and punctuation mark into a token:

### PDF p.410 (printed 386) — 14 Text classification › 14.2 Preparing text data › 
We introduce a special token called `"[UNK]"` to our vocabulary, which represents a token that is unknown to the vocabulary. This way, we can index all input we come across, even if some terms only occur in our test data. In the previous example `"lazy"` maps to the `"[UNK]"` index 0, as it was not included in our vocabulary.

With these simple text transformations, we are well on our way to building a text preprocessing pipeline. However, there is one more common type of text manipulation we should consider—standardization. Consider these two sentences:

“sunset came. i was staring at the Mexico sky. Isnt nature splendid??”

“Sunset came; I stared at the México sky. Isn’t nature splendid?”

They are very similar—in fact, they are almost identical. Yet, if you were to convert them to indices as previously described, you would end up with very different representations because “i” and “I” are two distinct characters, “Mexico” and “México” are two distinct words, “isnt” isn’t “isn’t,” and so on. A machine learning model doesn’t know a priori that “i” and “I” are the same letter, that “é” is an “e” with an accent, or that “staring” and “stared” are two forms of the same verb. Standard
