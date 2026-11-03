# Generation bundle for page_447  (PDF page 447, printed page 423)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.1 The language model
Subsection: 15.1.1 Training a Shakespeare language model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_447.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_447.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p447_e03] (body) Let’s take a look at some of the data:
[p447_e04] (code)
```
>>> shakespeare[:250]
First Citizen:
Before we proceed any further, hear me speak.
All:
Speak, speak.
First Citizen:
You are all resolved rather to die than to famish?
All:
Resolved. resolved.
First Citizen:
First, you know Caius Marcius is chief enemy to the people.
```
[p447_e05] (body) To build a language model from this input, we will need to massage our source text. First, we will split our data into equal-length chunks that we can batch and use for model training, much as we did for weather measurements in the timeseries chapters. Because we will be using a character-level tokenizer here, we can do this chunking directly on the string input. A 100-character string will map to a 100-integer sequence.
[p447_e06] (body) We will also split each input into two separate feature and label sequences, with each label sequence simply being the input sequence offset by a single character.
[p447_e07] (listing_caption) Listing 15.2 Splitting text into chunks for language model training
[p447_e08] (code)
```
import tensorflow as tf
sequence_length = 100
def split_input(input, sequence_length):
    for i in range(0, len(input), sequence_length):
        yield input[i : i + sequence_length]
features = list(split_input(shakespeare[:-1], sequence_length))
labels = list(split_input(shakespeare[1:], sequence_length))
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```
[p447_e09] (body) Let’s look at an `(x, y)` input sample. Our label at each position in the sequence is the next character in the sequence:
[p447_e10] (code)
```
>>> x, y = next(dataset.as_numpy_iterator())
>>> x[:50], y[:50]
(b"First Citizen:\nBefore we proceed any further, hear",
 b"irst Citizen:\nBefore we proceed any further, hear ")
```

## Code annotations printed next to listings (callouts, in order)
- The chunk size we will use during
- training. We only train on sequences
- of 100 characters at a time.

## Listings
- Listing 15.2: Splitting text into chunks for language model training (code elements p447_e08)

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

## Previous page (PDF 446) text, for continuity
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences. Attempting to represent every output sequence as a unique classifier output would overwhelm compute resources no matter how we design our model. A practical approach for making such a prediction problem feasible is to build a model that only predicts a single token output at a time. A language model is a model that, in its simplest form, learns a straightforward but deep probability distribution: `p(token|past tokens)`. Given a sequence of all tokens observed up to a point, a language model will attempt to output a probability distribution over all possible tokens that could come next. A 20, …

## Next page (PDF 448) text, for continuity
To map this input to a sequence of integers, we can again use the `TextVectorization` layer we saw in the last chapter. To learn a character-level vocabulary instead of a word-level vocabulary, we can change our `split` argument. Rather than the default `"whitespace"` splitting, we instead split by `"character"`. We will do no standardization here—to keep things simple, we will preserve case and pass punctuation through unaltered.

Listing 15.3 Learning a character-level vocabulary with the `TextVectorization` layer

Let’s inspect the vocabulary:

We need only 67 characters to handle the full source text.

Next, we can apply our tokenization layer to our input text. And finally, we can shuffle, batch, and cache our dataset so we don’t need to recompute it every epoch:

With that, we are ready to start modeling.

To build our simple language model, we want to predict the probability of a character given all past characters. Of all the modeling possibilities we have seen so far in this book, an RNN is the most natural fit, as the recurrent state of each cell allows the model to propagate information about past characters when predicting the label of the current character. We can also use an `Embedding`, as we saw in the previous chapter, to embed each input character as a unique 256-dimensional vector. We will use only a single recurrent layer to keep this model small and easy to train. Any recurrent layer would do here, but to keep things simple, we will use a `GRU`, which is fast and has a simpler internal state than an `LSTM`.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.452 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.449 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.450 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.483 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.484 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.451 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.453 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.488 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.488 (printed 464) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › Summary
A language model is a model that learns a specific probability distribution— `p(token|past tokens)`: – Language models have broad applications, but the most important is that you can generate text by calling them in a loop, where the output token at one time step becomes the input token in the next. – A masked language model learns a related probability distribution `p(tokens|surrounding` `tokens)` and can be helpful for classifying text and individual tokens. – A sequence-to-sequence language model learns to predict the next token given both past tokens in a target sequence and an entirely separate, fixed source sequence. Sequence-to-sequence models are useful for problems like translation and question answering. – A sequence-to-sequence model usually has two separate components. An encoder computes a representation of the source sequence, and a decoder takes this representation as input and predicts the next token in a target sequence based on past tokens.

Attention is a mechanism that allows a model to pull information from anywhere in a sequence selectively based on the context of the token currently being processed: – Attention avoids the problems RNNs have with long-range de

### PDF p.453 (printed 429) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
The general template behind sequence-to-sequence models is described in figure 15.1. During training the following happen:

An encoder model turns the source sequence into an intermediate representation.

A decoder is trained using the language modeling setup we saw previously. It will recursively predict the next token in the target sequence by looking at all previous target tokens and our encoder’s representation of the source sequence.

During inference, we don’t have access to the target sequence—we’re trying to predict it from scratch. We will generate it one token at a time, just as we did with our Shakespeare generator:

We obtain the encoded source sequence from the encoder.

The decoder starts by looking at the encoded source sequence as well as an initial “seed” token (such as the string `"[start]"`) and uses them to predict the first real token in the sequence.

The predicted sequence so far is fed back into the decoder, in a loop, until it generates a stop token (such as the string `"[end]"`).

Figure 15.1 Sequence-to-sequence learning: the source sequence is processed by the encoder and is then sent to the decoder. The decoder looks at the target sequence so far and pr

### PDF p.452 (printed 428) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
We have yet to produce the next great tragedy, but this is not terrible for two minutes of training on a minimal dataset. The goal of this toy example is to show the power of the language model setup. We trained the model on the narrow problem of guessing a single character at a time but still use it for a much broader problem, generating an open-ended, Shakespearean-like text response.

It’s important to notice that this training setup only works because a recurrent neural network only passes information forward in the sequence. If you’d like, try replacing the `GRU` layer with a `Bidirectional(GRU(...))`. The training accuracy will zoom to above 99% immediately, and generation will stop working entirely. During training, our model sees the entire sequence each train step. If we “cheat” by letting information from the next token in the sequence affect the current token’s prediction, we’ve made our problem trivially easy. This language modeling setup is fundamental to countless problems in the text domain. It is also somewhat unique compared to other modeling problems we have seen so far in this book. We cannot simply call `model.predict()` to get the desired output. There is an en

### PDF p.452 (printed 428) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 15.2 Sequence-to-sequence learning
Let’s take the language model idea and extend it to tackle an important problem— machine translation. Translation belongs to a class of modeling problems often called sequence-to-sequence modeling (or seq2seq if you are trying to save keystrokes). We seek to build a model that can take in a source text as a fixed input sequence and generate the translated text sequence as a result. Question answering is another classic sequence-to-sequence problem.

### PDF p.409 (printed 385) — 14 Text classification › 14.2 Preparing text data › 
Splitting takes us from a single string to a token sequence, but we still need to transform our string tokens into numeric inputs. By far the most common approach is to map each token to a unique integer index, often called indexing our input. This is a flexible and reversible representation of our tokenized input that can work with a wide range of modeling approaches. Later on, we can decide how to map from token indices into a latent space ingested by the model.

For character tokens, we could use ASCII lookups to index each token—for example, `ord('A') `→` 65` and `ord('z') `→` 122`. However, this can scale poorly when you start to consider other languages—there are over a million characters in the Unicode specification! A more robust technique is to build a mapping from specific tokens in our training data to indices that occur in the data we care about, which in NLP is called a vocabulary. This has the nice property of working for word-level tokens as easily as for character-level tokens. Let’s take a look at how we might use a vocabulary to transform text. We will build a simple Python dictionary that maps tokens to indices, split our input into tokens, and finally index our 

### PDF p.450 (printed 426) — 15 Language models and the Transformer › 15.1 The language model › 15.1.2 Generating Shakespeare
Now that we have trained a model that can predict the next individual tokens with some accuracy, we would like to use it to extrapolate an entire predicted sequence. We can do this by calling the model in a loop, where the model’s predicted output at one time step becomes the model’s input at the next time step. A model built for this kind of feedback loop is sometimes called an autoregressive model.

To run such a loop, we need to perform a slight surgery on the model we just trained. During training, our model handled only a fixed sequence length of 100 tokens, and the `GRU` cell’s state was handled implicitly when calling the layer. During generation, we would like to predict a single output token at a time and explicitly output the state of the `GRU`’s cell. We need to propagate that state, which contains all information the model has encoded about past input characters, the next time we call the model. Let’s make a model that handles a single input character at a time and allows explicitly passing the RNN state. Because this model will have the same computational structure, with slightly modified inputs and outputs, we can assign weights from one model to another.

Listing 15.

### PDF p.449 (printed 425) — 15 Language models and the Transformer › 15.1 The language model › 
This model outputs a softmax probability for every possible character in our vocabulary, and we will `compile()` it with a crossentropy loss. Note that our model is still training on a classification problem, it’s just that we will make one classification prediction for every token in our sequence. For our batch of 64 samples with 100 characters each, we will predict 6,400 individual labels. Loss and accuracy metrics reported by Keras during training will be averaged first across each sequence and, second, across each batch.

Let’s go ahead and train our language model.

Listing 15.5 Training a miniature language model

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
