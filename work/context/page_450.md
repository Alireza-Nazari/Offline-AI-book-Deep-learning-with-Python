# Generation bundle for page_450  (PDF page 450, printed page 426)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.1 The language model
Subsection: 15.1.2 Generating Shakespeare
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_450.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_450.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p450_e03] (code)
```
    metrics=["sparse_categorical_accuracy"],
)
model.fit(training_data, epochs=20)
```
[p450_e04] (body) After 20 epochs, our model can eventually predict the next character in our input sequences around 70% of the time.
[p450_e05] (heading_subsection) 15.1.2 Generating Shakespeare
[p450_e06] (body) Now that we have trained a model that can predict the next individual tokens with some accuracy, we would like to use it to extrapolate an entire predicted sequence. We can do this by calling the model in a loop, where the model’s predicted output at one time step becomes the model’s input at the next time step. A model built for this kind of feedback loop is sometimes called an autoregressive model.
[p450_e07] (body) To run such a loop, we need to perform a slight surgery on the model we just trained. During training, our model handled only a fixed sequence length of 100 tokens, and the `GRU` cell’s state was handled implicitly when calling the layer. During generation, we would like to predict a single output token at a time and explicitly output the state of the `GRU`’s cell. We need to propagate that state, which contains all information the model has encoded about past input characters, the next time we call the model. Let’s make a model that handles a single input character at a time and allows explicitly passing the RNN state. Because this model will have the same computational structure, with slightly modified inputs and outputs, we can assign weights from one model to another.
[p450_e08] (listing_caption) Listing 15.6 Modifying the language model for autoregressive inference
[p450_e09] (code)
```
inputs = keras.Input(shape=(1,), dtype="int", name="token_ids")
input_state = keras.Input(shape=(hidden_dim,), name="state")
x = layers.Embedding(vocabulary_size, embedding_dim)(inputs)
x, output_state = layers.GRU(hidden_dim, return_state=True)(
    x, initial_state=input_state
)
outputs = layers.Dense(vocabulary_size, activation="softmax")(x)
generation_model = keras.Model(
    inputs=(inputs, input_state),
    outputs=(outputs, output_state),
)
generation_model.set_weights(model.get_weights())
```
[p450_e10] (body) With this, we can call the model to predict an output sequence in a loop. Before we do, we will make explicit lookup tables so we switch from characters to integers and choose a prompt—a snippet of text we will feed as input to the model before we begin predicting new tokens:

## Code annotations printed next to listings (callouts, in order)
- Creates a
- model that
- receives and
- outputs the
- RNN state
- Copies the
- parameters from
- the original model

## Listings
- Listing 15.6: Modifying the language model for autoregressive inference (code elements p450_e09)

## Equation candidates (formulas written inline in the text)
- [p450_e03]     metrics=["sparse_categorical_accuracy"],
)
model.fit(training_data, epochs=20)

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

## Previous page (PDF 449) text, for continuity
Listing 15.4 Building a miniature language model

Let’s take a look at our model summary:

This model outputs a softmax probability for every possible character in our vocabulary, and we will `compile()` it with a crossentropy loss. Note that our model is still training on a classification problem, it’s just that we will make one classification prediction for every token in our sequence. For our batch of 64 samples with 100 characters each, we will predict 6,400 individual labels. Loss and accuracy metrics reported by Keras during training will be averaged first across each sequence and, second, across each batch.

Let’s go ahead and train our language model.

Listing 15.5 Training a miniature language model

## Next page (PDF 451) text, for continuity
To begin generation, we first need to “prime” the internal state of the GRU with our prompt. To do this, we will feed the prompt into the model one token at a time. This will compute the exact RNN state the model would see if this prompt had been encountered during training.

When we feed the very last character of the prompt into the model, our state output will capture information about the entire prompt sequence. We can save the final output prediction to later select the first character of our generated response.

Listing 15.7 Using a fixed prompt to compute a language model’s starting state

Now we are ready to let the model predict a new output sequence. In a loop, up to a desired length, we will continually select the most likely next character predicted by the model, feed that to the model, and persist the new RNN state. In this way, we can predict an entire sequence, a token at time.

Listing 15.8 Predicting with the language model a token at a time

Let’s convert our output integer sequence to a string to see what the model predicted. To detokenize our input, we simply map all token IDs to strings and join them together:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.452 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.448 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.446 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.488 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.457 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.458 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.453 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.447 — 15 Language models and the Transformer › 15.1 The language model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.452 (printed 428) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
We have yet to produce the next great tragedy, but this is not terrible for two minutes of training on a minimal dataset. The goal of this toy example is to show the power of the language model setup. We trained the model on the narrow problem of guessing a single character at a time but still use it for a much broader problem, generating an open-ended, Shakespearean-like text response.

It’s important to notice that this training setup only works because a recurrent neural network only passes information forward in the sequence. If you’d like, try replacing the `GRU` layer with a `Bidirectional(GRU(...))`. The training accuracy will zoom to above 99% immediately, and generation will stop working entirely. During training, our model sees the entire sequence each train step. If we “cheat” by letting information from the next token in the sequence affect the current token’s prediction, we’ve made our problem trivially easy. This language modeling setup is fundamental to countless problems in the text domain. It is also somewhat unique compared to other modeling problems we have seen so far in this book. We cannot simply call `model.predict()` to get the desired output. There is an en

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences

### PDF p.502 (printed 478) — 16 Text generation › 16.2 Training a mini-GPT › 16.2.3 Generative decoding
To sample some output from our model, we can follow the approach we used to generate Shakespeare or Spanish translations in chapter 15. We feed a prompt of fixed tokens into the model. For each position in the input sequence, the model outputs a probability distribution over the entire vocabulary for the next token. By selecting the most likely next token at the last location, adding it to our sequence, and then repeating this process, we are able to generate a new sequence, one token at a time.

Listing 16.9 A simple generation function for the mini-GPT model

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

### PDF p.448 (printed 424) — 15 Language models and the Transformer › 15.1 The language model › 
With that, we are ready to start modeling.

To build our simple language model, we want to predict the probability of a character given all past characters. Of all the modeling possibilities we have seen so far in this book, an RNN is the most natural fit, as the recurrent state of each cell allows the model to propagate information about past characters when predicting the label of the current character. We can also use an `Embedding`, as we saw in the previous chapter, to embed each input character as a unique 256-dimensional vector. We will use only a single recurrent layer to keep this model small and easy to train. Any recurrent layer would do here, but to keep things simple, we will use a `GRU`, which is fast and has a simpler internal state than an `LSTM`.

### PDF p.504 (printed 480) — 16 Text generation › 16.2 Training a mini-GPT › Cached generation
There’s still one more major inefficiency in the generation function we just built. Can you spot it? Each time we call our model, we call it for an entire sequence and then throw away everything but the predictions for a single position. This is wasteful—our sequence only changes by a single token between generation steps. When we did generation with an RNN in chapter 15, we could keep around our RNN state and only compute outputs for a single token at each step. This state vector contained all the information the model needed about the past sequence. Transformers that use a causal attention, like GPT, actually have a similar notion of state. If we think through the entire model we just built, you’ll note that attention is the only place where the model passes information from position to position. The feedforward blocks of a transformer only modify the hidden representation of each token position in isolation. Inside attention, we incorporate information about past tokens through the `key` and `value` vectors. For a given `query` at a position, we compute attention scores by dotting the `query` with all previous `key` vectors and combining all previous `value` vectors. These `key`

### PDF p.505 (printed 481) — 16 Text generation › 16.2 Training a mini-GPT › 
This isn’t so much a bug as it’s a direct consequence of our training objective. Our model is trying to predict the most likely next token in a sequence across about a billion words on many, many topics. If there’s no obvious choice for where a sequence of text should head next, an effective strategy is to guess common words or repeated patterns of words. Unsurprisingly, our model learns to do this during training almost immediately. If you were to stop training our model very early on, it would likely generate the word `"the"` incessantly, as `"the"` is the most common word in the English language. During our generative loop, we have always chosen the most likely predicted token in our model’s output. But our output is not just a single predicted token; it is a probability distribution across all 32,000 tokens in our vocabulary. Using the most likely output at each generation step is called greedy search. It’s the most straightforward approach to using model predictions, but it is hardly the only one. If we instead add some randomness to the process, we can explore the probability distribution learned by the model more broadly. This can keep us from getting stuck in loops of high-
