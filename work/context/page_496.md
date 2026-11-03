# Generation bundle for page_496  (PDF page 496, printed page 472)

Page kind: content
Chapter: 16 Text generation
Section: 16.2 Training a mini-GPT
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_496.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_496.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p496_e03] (body) As in the last chapter, we can use the KerasHub library to access some extra functions for working with large language models. KerasHub wraps the SentencePiece library as a Keras layer. Let’s try it out.
[p496_e04] (listing_caption) Listing 16.2 Downloading a SentencePiece vocabulary and instantiating a tokenizer
[p496_e05] (code)
```
import keras_hub
import numpy as np
vocabulary_file = keras.utils.get_file(
    origin="https://hf.co/mattdangerw/spiece/resolve/main/vocabulary.proto",
)
tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(vocabulary_file)
```
[p496_e06] (body) We can use this tokenizer to map from text to int sequences bidirectionally:
[p496_e07] (code)
```
>>> tokenizer.tokenize("The quick brown fox.")
array([  450,  4996, 17354,  1701, 29916, 29889], dtype=int32)
>>> tokenizer.detokenize([450, 4996, 17354, 1701, 29916, 29889])
"The quick brown fox."
```
[p496_e08] (body) Let’s use this layer to tokenize our input text and then use `tf.data` to window our input into sequences of length 256.
[p496_e09] (body) When training GPT, the developers chose to keep things simple and make no attempt to keep document boundaries from occurring in the middle of a sample. Instead, they marked a document boundary with a special `<|endoftext|>` token. We will do the same here. Once again, we will use `tf.data` for the input data pipeline and train with any backend. We will load each file shard individually and interleave the output data into a single dataset. This keeps our data loading fast, and we don’t need to worry about text lining up across sample boundaries—each is independent. With interleaving, each processor on our CPU can read and tokenize a separate file simultaneously.
[p496_e10] (listing_caption) Listing 16.3 Preprocessing text input for Transformer pretraining
[p496_e11] (code)
```
import tensorflow as tf
batch_size = 128
sequence_length = 256
suffix = np.array([tokenizer.token_to_id("<|endoftext|>")])
def read_file(filename):
    ds = tf.data.TextLineDataset(filename)
    ds = ds.map(lambda x: tf.strings.regex_replace(x, r"\\n", "\n"))
    ds = ds.map(tokenizer, num_parallel_calls=8)
    return ds.map(lambda x: tf.concat([x, suffix], -1))
```

## Code annotations printed next to listings (callouts, in order)
- Adds the <|endoftext|> token
- Tokenizes data
- Restores newlines

## Listings
- Listing 16.2: Downloading a SentencePiece vocabulary and instantiating a tokenizer (code elements p496_e05)
- Listing 16.3: Preprocessing text input for Transformer pretraining (code elements p496_e11)

## Chapter outline
Chapter: 16 Text generation (PDF pages 490-531)
  - 16.1 A brief history of sequence generation (p.492)
  - 16.2 Training a mini-GPT (p.494)
      - 16.2.1 Building the model (p.497)
      - 16.2.2 Pretraining the model (p.500)
      - 16.2.3 Generative decoding (p.502)
      - 16.2.4 Sampling strategies (p.504)
  - 16.3 Using a pretrained LLM (p.508)
      - 16.3.1 Text generation with the Gemma model (p.509)
      - 16.3.2 Instruction fine-tuning (p.512)
      - 16.3.3 Low-Rank Adaptation (LoRA) (p.514)
  - 16.4 Going further with LLMs (p.519)
      - 16.4.1 Reinforcement Learning with Human Feedback (RLHF) (p.519)
      - 16.4.2 Multimodal LLMs (p.522)
      - 16.4.3 Retrieval Augmented Generation (RAG) (p.525)
      - 16.4.4 “Reasoning” models (p.526)
  - 16.5 Where are LLMs heading next? (p.528)

## Previous page (PDF 495) text, for continuity
Running the code in this chapter

Generative language models are big and take a lot of compute to run. While we’ve taken pains to make the code in this chapter accessible, this is still the most compute-intensive chapter in this book. If you’d like, you can run everything on the free Colab GPU runtime (a T4 GPU as of this writing), but be prepared to wait! This mini-GPT example will take about 6 hours to train, and you will need to restart your Colab runtime in the middle of the notebook to free up GPU memory before loading a larger pretrained model. A larger GPU will make quicker work of these examples; we developed this example on an A100, which can run this chapter’s code end-to-end in a little over an hour. You can always read through the expensive `fit()` calls and edit down the number of training steps for quick experimentation. And if you are running this a few years down the line, there is a good chance these examples are mere child’s play to modern hardware!

We have 50 shards of text data, each with about 75 MB of raw text. Each line contains a document in the crawl with newlines escaped. Let’s look at a document in our first shard:

We will need to preprocess a lot of data to run pretraining for an LLM, even a miniature one like the one we are training. Using a fast tokenization routine to preprocess our source documents into integer tokens can simplify our lives.

We will use SentencePiece, a library for subword tokenization of text data. The actual tokenization technique is the same as the byte-pair encoding tokenization we built ourselves in chapter 14, but the library is written in C++ for speed and adds a `detokenize()` function that will reverse integers to strings and join them together. We will use a premade vocabulary with 32,000 vocabulary terms sto …

## Next page (PDF 497) text, for continuity
As we first did in chapter 8, we will end our `tf.data` pipeline with a call to `prefetch()`. This will make sure we always have some batches loaded onto our GPU and ready for the model.

We have 29,373 batches. You could count this yourself if you would like—the line `ds.reduce(0, lambda c, _: c + 1)` will iterate over the entire dataset and increment a counter. But simply tokenizing a dataset of this size will take a few minutes on a decently fast CPU. At 128 samples per batch and 256 tokens per sample, this is just under a billion tokens of data. Let’s split off 500 batches as a quick validation set, and we are ready to start pretraining:

16.2.1 Building the model

The original GPT model simplifies the sequence-to-sequence Transformer we saw in the last chapter. Rather than take in a source and target sequence with an encoder and decoder, as we did for our translation model, the GPT approach does away with the encoder entirely and only uses the decoder. This means that information can only travel from left to right in a sequence.

This was an interesting bet on the part of the GPT developers. A decoder-only model can still handle sequence-to-sequence problems like question-answering. However, rather than feeding in the question and answer as separate inputs, we must combine both into a single sequence to feed it to our model. So, unlike the original Transformer, the question tokens would not be handled any differently than answer tokens. All tokens are embedded into the same latent space with the same set of parameters. The other consequence of this approach is that the information flow is no longer bidirectional, even for input sequences. Given an input, such as “Where is the capital of

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.448 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.513 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.426 — 14 Text classification › 14.4 Set models
- PDF p.418 — 14 Text classification › 14.2 Preparing text data
- PDF p.483 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.480 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.413 — 14 Text classification › 14.2 Preparing text data
- PDF p.414 — 14 Text classification › 14.2 Preparing text data

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.426 (printed 402) — 14 Text classification › 14.4 Set models › Preprocessing text efficiently
use it throughout this chapter—`keras.utils.text_dataset_from_directory()` will

load a `tf.data.Dataset`, and `map()` will transform our input data, for example, by applying a `TextVectorization` layer. `tf.data` works by running text preprocessing in parallel multiple CPU cores, which is generally sufficient to avoid bottlenecking accelerators during a training run. It is important to note that the code in this chapter is still multi-backend (in fact, we generated the outputs for this chapter using Jax). You can use `tf.data` with PyTorch, JAX, or TensorFlow itself—Keras will automatically convert input Tensors to the correct format for a given backend.

### PDF p.423 (printed 399) — 14 Text classification › 14.4 Set models › 14.4.1 Training a bag-of-words model
To do this text processing in code, it would be easy enough to extend our `WordTokenizer` from earlier in the chapter. An even easier solution is to use the `TextVectorization` layer built into Keras. The `TextVectorization` handles word and character tokenization and comes with several additional features, including multi-hot encoding of the layer output.

The `TextVectorization` layer, like many preprocessing layers in Keras, has an `adapt()` method to learn a layer state from input data. In the case of `TextVectorization`, `adapt()` will learn a vocabulary for a dataset on the fly by iterating over an input dataset. Let’s use it to tokenize and encode our input data. We will build a vocabulary of 20,000 words, a good starting place for text classification problems.

Listing 14.12 Applying a bag-of-words encoding to the IMDb reviews

### PDF p.447 (printed 423) — 15 Language models and the Transformer › 15.1 The language model › 
To build a language model from this input, we will need to massage our source text. First, we will split our data into equal-length chunks that we can batch and use for model training, much as we did for weather measurements in the timeseries chapters. Because we will be using a character-level tokenizer here, we can do this chunking directly on the string input. A 100-character string will map to a 100-integer sequence.

We will also split each input into two separate feature and label sequences, with each label sequence simply being the input sequence offset by a single character.

Listing 15.2 Splitting text into chunks for language model training

### PDF p.483 (printed 459) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
It’s important to match the token ordering used for pretraining as closely as possible; the model will train more quickly and accurately if we do. KerasHub provides a layer for this type of token packing called the `StartEndPacker`. The layer appends start, end, and padding tokens, trimming long sequences to a given sequence length if necessary. Let’s use it along with our tokenizer.

Listing 15.21 Preprocessing IMDb movie reviews with RoBERTa’s tokenizer

### PDF p.480 (printed 456) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
The `Tokenizer` maps from text to sequences of integers, as we would expect. Remember the `SubWordTokenizer` we built in the last chapter? RoBERTa’s tokenizer is almost the same as that tokenizer, with minor tweaks to handle Unicode characters from any language.

Given the size of RoBERTa’s pretraining dataset, subword tokenization is a must. Using a character-level tokenizer would make input sequences way too long, making the model much more expensive to train. Using a word-level tokenizer would require a massive vocabulary to attempt to cover all the distinct words in the millions of documents of text used from across the web. Getting good coverage of words would blow up our vocabulary size and make the `Embedding` layer at the front of the Transformer unworkably large. Using a subword tokenizer allows the model to handle any word with only a 50,000-term vocabulary:

### PDF p.480 (printed 456) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
reviews map to different token indices than they would have during pretraining, we cannot use the learned representations of each token in the model.

A matching model architecture—To use the pretrained model, we need to recreate the math used internally by the model for pretraining exactly.

The pretrained weights—These weights were created by training the model for about a day on 1,024 GPUs and billions of input words.

Recreating the tokenizer and architecture code ourselves would not be too hard. The model internals almost exactly match the `TransformerEncoder` we built previously. However, matching a model implementation is a time-consuming process, and as we have done earlier in this book, we can instead use the KerasHub library to access pretrained model implementations for Keras.

Let’s use KerasHub to load a RoBERTa tokenizer and model. We can use the special constructor `from_preset()` to load a pretrained model’s weights, configuration, and tokenizer assets from disk. We will load RoBERTa’s base model, which is the smallest of the few pretrained checkpoints released with the RoBERTa paper.

Listing 15.20 Loading the RoBERTa pretrained model with KerasHub

### PDF p.418 (printed 394) — 14 Text classification › 14.2 Preparing text data › 
The `SubWordTokenizer` has a slightly longer length for our test sentence than the `Word` `Tokenizer` (16 versus 13 tokens), but unlike the `WordTokenizer`, it can tokenize every word in Moby Dick without using the `"[UNK]"` token. The vocabulary contains every character in our source text, so the worst-case performance will be tokenizing a word into individual characters. We have achieved a short average token length while handling rare words with a small vocabulary. This is the advantage of subword tokenizers.

You might notice that running this code is noticeably slower than the word and character tokenizers; it takes about a minute on our reference hardware. Learning merge rules is much more complex than simply counting the words in an input dataset. While this is a downside to subword tokenization, it is rarely an important concern in practice. You only need to learn a vocabulary once per model, and the cost of learning a subword vocabulary is generally negligible compared to model training. We have now seen three separate approaches for tokenizing input. Now that we can translate from text to numeric input, we can move on to training a model. One final note on tokenization—wh

### PDF p.494 (printed 470) — 16 Text generation › 16.2 Training a mini-GPT › 16.2 Training a mini-GPT
To begin pretraining our mini-GPT, we will need a lot of text data. GPT-1 used a dataset called BooksCorpus, which contained a number of free, self-published books added to the dataset without the explicit permission of the authors. The dataset has since been taken down by its publishers.

We will use a more recent pretraining dataset called the “Colossal Clean Crawled Corpus” (C4), released by Google in 2020. At 750 GB, it’s far bigger than we could reasonably train on for a book example, so we will use less than 1% of the overall corpus. Let’s start by downloading and extracting our data.

Listing 16.1 Downloading a portion of the C4 dataset
