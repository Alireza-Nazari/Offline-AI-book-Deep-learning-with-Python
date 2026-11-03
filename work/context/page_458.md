# Generation bundle for page_458  (PDF page 458, printed page 434)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.2 Sequence-to-sequence learning
Subsection: 15.2.2 Sequence-to-sequence learning with RNNs
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_458.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_458.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p458_e03] (body) idea, as we want a rich representation of the source sequence. We can achieve this with a `Bidirectional` layer.
[p458_e04] (listing_caption) Listing 15.11 Building a sequence-to-sequence encoder
[p458_e05] (code)
```
embed_dim = 256
hidden_dim = 1024
source = keras.Input(shape=(None,), dtype="int32", name="english")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
rnn_layer = layers.GRU(hidden_dim)
rnn_layer = layers.Bidirectional(rnn_layer, merge_mode="sum")
encoder_output = rnn_layer(x)
```
[p458_e06] (body) Next, let’s add the decoder—a simple `GRU` layer that takes as its initial state the encoded source sentence. On top of it, we add a `Dense` layer that produces a probability distribution over the Spanish vocabulary for each output step. Here, we want to predict the next tokens based only on what came before, so a `Bidirectional` RNN would break training by making the loss function trivially easy.
[p458_e07] (listing_caption) Listing 15.12 Building a sequence-to-sequence decoder
[p458_e08] (code)
```
target = keras.Input(shape=(None,), dtype="int32", name="spanish")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(target)
rnn_layer = layers.GRU(hidden_dim, return_sequences=True)
x = rnn_layer(x, initial_state=encoder_output)
x = layers.Dropout(0.5)(x)
target_predictions = layers.Dense(vocab_size, activation="softmax")(x)
seq2seq_rnn = keras.Model([source, target], target_predictions)
```
[p458_e09] (body) Let’s take a look at the seq2seq model in full:
[p458_e10] (code)
```
>>> seq2seq_rnn.summary()
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)          ┃ Output Shape      ┃     Param # ┃ Connected to       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ english (InputLayer)  │ (None, None)      │           0 │ -                  │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ spanish (InputLayer)  │ (None, None)      │           0 │ -                  │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ embedding_1           │ (None, None, 256) │   3,840,000 │ english[0][0]      │
│ (Embedding)           │                   │             │                    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ not_equal (NotEqual)  │ (None, None)      │           0 │ english[0][0]      │
```

## Code annotations printed next to listings (callouts, in order)
- Predicts the next word of the
- translation, given the current word

## Listings
- Listing 15.11: Building a sequence-to-sequence encoder (code elements p458_e05)
- Listing 15.12: Building a sequence-to-sequence decoder (code elements p458_e08)

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

## Previous page (PDF 457) text, for continuity
into another is to keep the output of the RNN at each time step and predict an output token from it. In Keras, it would look like this:

However, there is a critical issue with this approach. Due to the step-by-step nature of RNNs, the model will only look at tokens `0...N` in the source sequence to predict token `N` in the target sequence. Consider translating the sentence, “I will bring the bag to you.” In Spanish, that would be “Te traeré la bolsa,” where “Te,” the first word of the translation, corresponds to “you” in the English source text. There’s simply no way to output the first word of the translation without seeing the last word of the source English text!

If you’re a human translator, you’d start by reading the entire source sentence before beginning to translate it. This is especially important if you’re dealing with languages with wildly different word ordering. And that’s precisely what standard sequence-to-sequence models do. In a proper sequence-to-sequence setup (see figure 15.2), you would first use an encoder RNN to turn the entire source sequence into a single representation of the source text. This could be the last output of the RNN or, alternatively, its final internal state vectors. We can use this representation as the initial state of a decoder RNN in the language model setup instead of an initial state of zeros, which we used in our Shakespeare generator. This decoder learns to predict the next word of the Spanish translation given the current word of the translation, with all information about the English sequence coming from that initial RNN state.

Figure 15.2 A sequence-to-sequence RNN: an RNN encoder is used to produce a vector that encodes the entire source sequence, which is used as the initial state for an RNN decoder.

Let’s impleme …

## Next page (PDF 459) text, for continuity
Our model and data are both ready. We can now begin training our translation model:

We picked accuracy as a crude way to monitor validation set performance during training. We get to 65% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 65% of the time. However, in practice, next-token accuracy isn’t a great metric for machine translation models, in particular because it makes the assumption that the correct target tokens from `0` to `N` are already known when predicting token `N + 1`. In reality, during inference, you’re generating the target sentence from scratch, and you can’t rely on previously generated tokens being 100% correct. When working on a real-world machine translation system, metrics must be more carefully designed. There are standard metrics, such as a BLEU score, that measure the similarity of the machine-translated text to a set of high-quality reference translations and can tolerate slightly misaligned sequences.

At last, let’s use our model for inference. We’ll pick a few sentences in the test set and check how our model translates them. We’ll start from the seed token `"[start]"` and feed it into the decoder model, together with the encoded English source sentence. We’ll retrieve a next-token prediction, and we’ll re-inject it into the decoder repeatedly, sampling one new target token at each iteration, until we get to `"[end]"` or reach the maximum sentence length.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.473 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.476 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.449 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.453 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.474 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.436 — 14 Text classification › 14.5 Sequence models
- PDF p.431 — 14 Text classification › 14.5 Sequence models
- PDF p.471 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.453 (printed 429) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
The general template behind sequence-to-sequence models is described in figure 15.1. During training the following happen:

An encoder model turns the source sequence into an intermediate representation.

A decoder is trained using the language modeling setup we saw previously. It will recursively predict the next token in the target sequence by looking at all previous target tokens and our encoder’s representation of the source sequence.

During inference, we don’t have access to the target sequence—we’re trying to predict it from scratch. We will generate it one token at a time, just as we did with our Shakespeare generator:

We obtain the encoded source sequence from the encoder.

The decoder starts by looking at the encoded source sequence as well as an initial “seed” token (such as the string `"[start]"`) and uses them to predict the first real token in the sequence.

The predicted sequence so far is fed back into the decoder, in a loop, until it generates a stop token (such as the string `"[end]"`).

Figure 15.1 Sequence-to-sequence learning: the source sequence is processed by the encoder and is then sent to the decoder. The decoder looks at the target sequence so far and pr

### PDF p.461 (printed 437) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Note that this inference setup, while very simple, is inefficient, since we reprocess the entire source sentence and the entire generated target sentence every time we sample a new word. In a practical application, you’d want to be careful not to recompute any state that has not changed. All we really need to predict a new token in the decoder is the current token and the previous RNN state, which we could cache before each loop iteration. There are many ways this toy model could be improved. We could use a deep stack of recurrent layers for both the encoder and the decoder, we could try other RNN layers like `LSTM`, and so on. Beyond such tweaks, however, the RNN approach to sequence-to-sequence learning has a few fundamental limitations:

The source sequence representation has to be held entirely in the encoder state vector, which significantly limits the size and complexity of the sentences you can translate.

RNNs have trouble dealing with very long sequences since they tend to progressively forget about the past—by the time you’ve reached the 100th token in either sequence, little information remains about the start of the sequence.

Recurrent neural networks dominated sequenc

### PDF p.488 (printed 464) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › Summary
A language model is a model that learns a specific probability distribution— `p(token|past tokens)`: – Language models have broad applications, but the most important is that you can generate text by calling them in a loop, where the output token at one time step becomes the input token in the next. – A masked language model learns a related probability distribution `p(tokens|surrounding` `tokens)` and can be helpful for classifying text and individual tokens. – A sequence-to-sequence language model learns to predict the next token given both past tokens in a target sequence and an entirely separate, fixed source sequence. Sequence-to-sequence models are useful for problems like translation and question answering. – A sequence-to-sequence model usually has two separate components. An encoder computes a representation of the source sequence, and a decoder takes this representation as input and predicts the next token in a target sequence based on past tokens.

Attention is a mechanism that allows a model to pull information from anywhere in a sequence selectively based on the context of the token currently being processed: – Attention avoids the problems RNNs have with long-range de

### PDF p.456 (printed 432) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 15.2.2 Sequence-to-sequence learning with RNNs
Before we try the twin encoder/decoder setup we previously mentioned, let’s think through simpler options. The easiest, naive way to use RNNs to turn one sequence

### PDF p.431 (printed 407) — 14 Text classification › 14.5 Sequence models › 
We can build this layer directly into a model and use a bidirectional LSTM to allow information to propagate both forward and backward along the token sequence. Later, when we look at generation, we will see the need for unidirectional sequence models (where a token state only depends on the token state before it). For classification tasks, a bidirectional LSTM is a good fit.

Let’s build our model.

Listing 14.21 Building an LSTM sequence model

### PDF p.497 (printed 473) — 16 Text generation › 16.2 Training a mini-GPT › 16.2.1 Building the model
The original GPT model simplifies the sequence-to-sequence Transformer we saw in the last chapter. Rather than take in a source and target sequence with an encoder and decoder, as we did for our translation model, the GPT approach does away with the encoder entirely and only uses the decoder. This means that information can only travel from left to right in a sequence.

This was an interesting bet on the part of the GPT developers. A decoder-only model can still handle sequence-to-sequence problems like question-answering. However, rather than feeding in the question and answer as separate inputs, we must combine both into a single sequence to feed it to our model. So, unlike the original Transformer, the question tokens would not be handled any differently than answer tokens. All tokens are embedded into the same latent space with the same set of parameters. The other consequence of this approach is that the information flow is no longer bidirectional, even for input sequences. Given an input, such as “Where is the capital of

### PDF p.452 (printed 428) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 15.2 Sequence-to-sequence learning
Let’s take the language model idea and extend it to tackle an important problem— machine translation. Translation belongs to a class of modeling problems often called sequence-to-sequence modeling (or seq2seq if you are trying to save keystrokes). We seek to build a model that can take in a source text as a fixed input sequence and generate the translated text sequence as a result. Question answering is another classic sequence-to-sequence problem.

### PDF p.489 (printed 465) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The Transformer is a sequence modeling architecture that uses attention as the only mechanism to pass information across a sequence: – The Transformer works by stacking blocks of alternating attention and two-layer feedforward networks. – The Transformer can scale to many parameters and lots of training data while still improving accuracy in the language modeling problem. – Unlike RNNs, the Transformer involves no sequence-length loops at training time, making the model much easier to train in parallel across many machines. – A Transformer encoder uses bidirectional attention to build a rich representation of a sequence. – A Transformer decoder uses causal attention to predict the next word in a language model setup.
